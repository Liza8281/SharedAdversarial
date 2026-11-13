
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from gan_gpt2_bert_train import (
    bert_batch,
    load_train_urls,
    looks_like_url,
    sample_generator,
    save_models,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = SCRIPT_DIR / "CleanedDATA" / "Train_2025.csv"
DEFAULT_GPT2_CHECKPOINT = SCRIPT_DIR / "PreTrainedGenerator"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "GAN_PPO_output_warm_seed45"

DEFAULT_DISC_CHECKPOINT = str(SCRIPT_DIR / "WarmedDiscriminator")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2/BERT GAN: PPO-Clip actor-critic")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--url-column", default="url")
    p.add_argument("--gpt2-checkpoint", default=str(DEFAULT_GPT2_CHECKPOINT))
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--bert-checkpoint", default=DEFAULT_DISC_CHECKPOINT,)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--bert-max-length", type=int, default=128)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr-generator", type=float, default=1e-6)
    p.add_argument("--lr-discriminator", type=float, default=2e-5)
    p.add_argument("--lr-value", type=float, default=1e-4)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    if min(args.steps, args.max_new_tokens, args.save_every, args.log_every, args.ppo_epochs) < 1:
        p.error("steps, max-new-tokens, save-every, log-every, ppo-epochs must be positive")
    if args.batch_size < 2 or args.top_k < 2 or args.temperature <= 0:
        p.error("batch-size >= 2, top-k >= 2, temperature > 0 required")
    if not 0 < args.clip_epsilon < 1 or args.entropy_coef < 0 or args.target_kl < 0:
        p.error("clip-epsilon must be in (0,1), entropy-coef and target-kl >= 0")
    return args


class ValueHead(nn.Module):
    """Estimate discriminator reward"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, prefix_features: torch.Tensor) -> torch.Tensor:
        return self.network(prefix_features).squeeze(-1)


def policy_token_stats(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    active: torch.Tensor,
    top_k: int,
    temperature: float,
    pad_id: int,
    eos_id: int,
    strict_support: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:

    if pad_id != eos_id:
        logits = logits.clone()
        logits[..., pad_id] = -float("inf")
    values, indices = logits.topk(k=min(top_k, logits.shape[-1]), dim=-1)
    matches = indices.eq(target_tokens.unsqueeze(-1))
    is_supported = matches.any(dim=-1)
    if strict_support and not bool((is_supported | ~active).all().item()):
        raise RuntimeError("The sampling policy and its original log-probabilities disagree")

    chosen_index = matches.int().argmax(dim=-1).unsqueeze(-1)
    log_probs = F.log_softmax(values / temperature, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=chosen_index).squeeze(-1)
    if not strict_support:
        token_log_probs = torch.where(
            is_supported, token_log_probs, torch.full_like(token_log_probs, -30.0),
        )
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return token_log_probs, entropy


@torch.no_grad()
def collect_old_policy(
    generator,
    sample_ids: torch.Tensor,
    active: torch.Tensor,
    prompt_len: int,
    top_k: int,
    temperature: float,
    pad_id: int,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Freeze old action probabilities for one PPO rollout."""
    generator.eval()
    result = generator(input_ids=sample_ids, output_hidden_states=True, use_cache=False)
    # A token at position prompt_len is chosen using the logits at prompt_len-1.
    prefix_slice = slice(prompt_len - 1, -1)
    old_logp, _ = policy_token_stats(
        result.logits[:, prefix_slice, :], sample_ids[:, prompt_len:],
        active, top_k, temperature, pad_id, eos_id,
    )
    old_prefix_features = result.hidden_states[-1][:, prefix_slice, :].detach()
    return old_logp.detach(), old_prefix_features


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    real_urls = load_train_urls(args.train_csv, args.url_column)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(real_urls):,} URLs from TRAIN CSV ONLY; device={device}")

    g_tok = AutoTokenizer.from_pretrained(args.gpt2_checkpoint)
    generator = AutoModelForCausalLM.from_pretrained(args.gpt2_checkpoint).to(device)
    if g_tok.eos_token_id is None:
        raise ValueError("The fine-tuned GPT-2 tokenizer must define an EOS token")
    if g_tok.pad_token_id is None:
        g_tok.pad_token = g_tok.eos_token
    if len(g_tok) != generator.get_input_embeddings().num_embeddings:
        raise ValueError("GPT-2 tokenizer and embedding sizes differ")

    d_tok = AutoTokenizer.from_pretrained(args.bert_checkpoint)
    discriminator = AutoModelForSequenceClassification.from_pretrained(
        args.bert_checkpoint, num_labels=2,
    ).to(device)
    discriminator.config.id2label = {0: "GENERATED", 1: "REAL"}
    discriminator.config.label2id = {"GENERATED": 0, "REAL": 1}

    value_head = ValueHead(generator.config.n_embd).to(device)
    actor_optimizer = AdamW(generator.parameters(), lr=args.lr_generator)
    discriminator_optimizer = AdamW(discriminator.parameters(), lr=args.lr_discriminator)
    critic_optimizer = AdamW(value_head.parameters(), lr=args.lr_value)

    prompt_ids = g_tok.encode("https://", add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Cannot tokenize https:// prompt")
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if len(prompt_ids) + args.max_new_tokens > getattr(generator.config, "n_positions", 1024):
        raise ValueError("GPT-2 context window is shorter than prompt + max-new-tokens")

    config = {**vars(args), "train_csv": str(args.train_csv),
              "output_dir": str(args.output_dir), "num_train_urls": len(real_urls),
              "gan_labels": {"0": "generated", "1": "real phishing"},
              "ppo_variant": "token-level PPO-Clip + prefix-value-head + Monte Carlo terminal reward"}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    fields = ["step", "d_loss", "d_accuracy", "mean_log_p_real", "actor_loss", "value_loss",
              "approx_kl", "clip_fraction", "actual_ppo_epochs", "unique_rate",
              "syntax_valid_rate", "avg_new_tokens"]
    metrics_path = args.output_dir / "training_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fields)
        writer.writeheader()

        for step in trange(1, args.steps + 1, desc="PPO-GAN training"):

            ids, active, generated_urls = sample_generator(
                generator, g_tok, prompt, args.batch_size,
                args.max_new_tokens, args.top_k, args.temperature,
            )
            real_batch = random.choices(real_urls, k=args.batch_size)


            discriminator.train()
            d_batch = bert_batch(d_tok, real_batch + generated_urls, args.bert_max_length, device)
            targets = torch.cat((torch.ones(args.batch_size, device=device, dtype=torch.long),
                                 torch.zeros(args.batch_size, device=device, dtype=torch.long)))
            discriminator_optimizer.zero_grad(set_to_none=True)
            d_logits = discriminator(**d_batch).logits
            d_loss = F.cross_entropy(d_logits, targets)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            discriminator_optimizer.step()
            d_accuracy = (d_logits.argmax(dim=-1) == targets).float().mean().item()


            discriminator.eval()
            with torch.no_grad():
                fake_batch = bert_batch(d_tok, generated_urls, args.bert_max_length, device)
                fake_logits = discriminator(**fake_batch).logits
                rewards = F.log_softmax(fake_logits, dim=-1)[:, 1].clamp(min=-10.0)


            old_logp, old_states = collect_old_policy(
                generator, ids, active, len(prompt_ids),
                args.top_k, args.temperature, g_tok.pad_token_id, g_tok.eos_token_id,
            )
            with torch.no_grad():
                old_values = value_head(old_states)
                advantages = rewards[:, None] - old_values

                selected_adv = advantages[active]
                advantages = (advantages - selected_adv.mean()) / (
                    selected_adv.std(unbiased=False).clamp_min(1e-6)
                )

            weights = active.to(dtype=torch.float32)
            count = weights.sum().clamp_min(1)
            actor_loss_value = value_loss_value = approx_kl_value = clip_fraction_value = 0.0
            actual_epochs = 0


            for epoch in range(args.ppo_epochs):
                generator.eval()  # Gradients work in eval mode; dropout must match OLD policy.
                actor_optimizer.zero_grad(set_to_none=True)
                lm_logits = generator(input_ids=ids, use_cache=False).logits[:, len(prompt_ids) - 1:-1, :]
                new_logp, entropy = policy_token_stats(
                    lm_logits, ids[:, len(prompt_ids):], active,
                    args.top_k, args.temperature, g_tok.pad_token_id, g_tok.eos_token_id,
                    strict_support=False,
                )
                log_ratio = new_logp - old_logp
                ratio = log_ratio.clamp(min=-20.0, max=20.0).exp()
                unclipped = ratio * advantages
                clipped = ratio.clamp(1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
                surrogate_loss = -(torch.minimum(unclipped, clipped) * weights).sum() / count
                mean_entropy = (entropy * weights).sum() / count
                actor_loss = surrogate_loss - args.entropy_coef * mean_entropy
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                actor_optimizer.step()


                critic_optimizer.zero_grad(set_to_none=True)
                predicted_values = value_head(old_states)
                value_loss = (((predicted_values - rewards[:, None]) ** 2) * weights).sum() / count
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
                critic_optimizer.step()

                with torch.no_grad():

                    approx_kl = (((ratio - 1) - log_ratio) * weights).sum() / count
                    clip_fraction = (((ratio - 1).abs() > args.clip_epsilon) * weights).sum() / count
                actor_loss_value = actor_loss.item()
                value_loss_value = value_loss.item()
                approx_kl_value = approx_kl.item()
                clip_fraction_value = clip_fraction.item()
                actual_epochs = epoch + 1
                if args.target_kl > 0 and approx_kl_value > args.target_kl:
                    break

            row = {
                "step": step,
                "d_loss": round(d_loss.item(), 6),
                "d_accuracy": round(d_accuracy, 4),
                "mean_log_p_real": round(rewards.mean().item(), 6),
                "actor_loss": round(actor_loss_value, 6),
                "value_loss": round(value_loss_value, 6),
                "approx_kl": round(approx_kl_value, 6),
                "clip_fraction": round(clip_fraction_value, 4),
                "actual_ppo_epochs": actual_epochs,
                "unique_rate": round(len(set(generated_urls)) / len(generated_urls), 4),
                "syntax_valid_rate": round(sum(looks_like_url(u) for u in generated_urls) / len(generated_urls), 4),
                "avg_new_tokens": round(active.sum(dim=1).float().mean().item(), 2),
            }
            writer.writerow(row)
            if step % args.log_every == 0 or step == 1:
                log_file.flush()
                print(f"step={step} D_acc={row['d_accuracy']:.3f} "
                      f"R={row['mean_log_p_real']:.3f} KL={row['approx_kl']:.4f} "
                      f"ppo_epochs={actual_epochs} valid={row['syntax_valid_rate']:.3f}", flush=True)

            if step % args.save_every == 0 or step == args.steps:
                save_models(generator, g_tok, discriminator, d_tok, args.output_dir, step)
                checkpoint = args.output_dir / f"checkpoint_step_{step:06d}"
                torch.save(value_head.state_dict(), checkpoint / "value_head.pt")
                (args.output_dir / f"examples_step_{step:06d}.txt").write_text(
                    "\n".join(generated_urls) + "\n", encoding="utf-8",
                )

    print(f"Done.")


if __name__ == "__main__":
    main()
