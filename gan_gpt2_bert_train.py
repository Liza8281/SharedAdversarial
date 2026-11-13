

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 / BERT URL GAN (train split only)")
    p.add_argument("--train-csv", type=Path, required=True, help="Existing phishing TRAIN CSV, not full dataset")
    p.add_argument("--url-column", default="url")
    p.add_argument("--gpt2-checkpoint", required=True, help="Existing fine-tuned GPT-2 folder")
    p.add_argument("--output-dir", type=Path, default=Path("results/gan_gpt2_bert"))
    p.add_argument("--bert-checkpoint", default="bert-base-uncased", help="Use pretrained, not phishing-fine-tuned BERT")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--bert-max-length", type=int, default=128)
    p.add_argument("--top-k", type=int, default=50, help="Exact top-k distribution used in sampling AND logprob calculation")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr-generator", type=float, default=1e-6)
    p.add_argument("--lr-discriminator", type=float, default=2e-5)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.max_new_tokens < 1:
        p.error("steps >= 1, batch-size >= 2, max-new-tokens >= 1 required")
    if args.top_k < 2 or args.temperature <= 0 or args.save_every < 1 or args.log_every < 1:
        p.error("top-k >= 2, temperature > 0, save-every/log-every >= 1 required")
    return args


def load_train_urls(path: Path, column: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"TRAIN CSV does not exist: {path}")
    df = pd.read_csv(path)
    if column not in df:
        raise ValueError(f"Column {column!r} absent. Available: {df.columns.tolist()}")
    urls = df[column].dropna().astype(str).str.strip()
    urls = urls[urls.str.match(r"^https?://", case=False, na=False)]
    urls = urls.drop_duplicates().tolist()
    if len(urls) < 2:
        raise ValueError("At least two non-empty http(s) URLs are required in the TRAIN CSV")
    return urls


def candidate_url(text: str) -> str:
    """Use the first whitespace-delimited URL candidate, never append training URLs."""
    parts = text.strip().split(maxsplit=1)
    return parts[0] if parts else ""


def looks_like_url(text: str) -> bool:
    # A syntax proxy, NOT proof that an actual phishing site exists.
    return bool(re.match(r"^https?://[^\s/?#:.]+(?:\.[^\s/?#:.]+)+(?::\d+)?(?:[/?#][^\s]*)?$", text))


@torch.no_grad()
def sample_generator(
    generator, g_tokenizer, prompt: torch.Tensor, batch_size: int,
    max_new_tokens: int, top_k: int, temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    generator.eval()  # Keep dropout disabled in sampling and later scoring.
    ids = prompt.expand(batch_size, -1).clone()
    ended = torch.zeros(batch_size, dtype=torch.bool, device=ids.device)
    active_columns: list[torch.Tensor] = []
    eos = g_tokenizer.eos_token_id
    pad = g_tokenizer.pad_token_id
    assert eos is not None and pad is not None

    for _ in range(max_new_tokens):
        logits = generator(input_ids=ids).logits[:, -1, :]
        if pad != eos:
            logits[:, pad] = -float("inf")  # Pad is never a URL character.
        top_values, top_ids = logits.topk(k=min(top_k, logits.shape[-1]), dim=-1)
        probabilities = F.softmax(top_values / temperature, dim=-1)
        choices = torch.multinomial(probabilities, num_samples=1)
        next_ids = top_ids.gather(1, choices).squeeze(1)
        active_columns.append(~ended)
        next_ids = torch.where(ended, torch.full_like(next_ids, eos), next_ids)
        ids = torch.cat((ids, next_ids.unsqueeze(1)), dim=1)
        ended = ended | next_ids.eq(eos)
        if ended.all():
            break

    active = torch.stack(active_columns, dim=1)
    raw_strings = g_tokenizer.batch_decode(ids.detach().cpu(), skip_special_tokens=True)
    return ids, active, [candidate_url(s) for s in raw_strings]


def sampled_sequence_logprobs(
    generator, ids: torch.Tensor, active: torch.Tensor,
    prompt_len: int, top_k: int, temperature: float, pad_token_id: int, eos_token_id: int,
) -> torch.Tensor:

    generator.eval()  # Gradients remain enabled; eval only disables dropout.
    logits = generator(input_ids=ids).logits[:, prompt_len - 1:-1, :]
    if pad_token_id != eos_token_id:
        logits = logits.clone()
        logits[..., pad_token_id] = -float("inf")
    top_values, top_ids = logits.topk(k=min(top_k, logits.shape[-1]), dim=-1)
    targets = ids[:, prompt_len:]
    matches = top_ids.eq(targets.unsqueeze(-1))
    if not (matches.any(dim=-1) | ~active).all():
        raise RuntimeError("Sample/scorer top-k mismatch; check that GPT-2 was not updated between them")
    selected = matches.int().argmax(dim=-1).unsqueeze(-1)
    log_probs = F.log_softmax(top_values / temperature, dim=-1)
    chosen_logprobs = log_probs.gather(-1, selected).squeeze(-1)
    return (chosen_logprobs * active.to(chosen_logprobs.dtype)).sum(dim=1)


def bert_batch(tokenizer, texts: list[str], max_length: int, device: torch.device) -> dict:
    result = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {k: v.to(device) for k, v in result.items()}


def save_models(generator, g_tok, discriminator, d_tok, dest: Path, step: int) -> None:
    step_dir = dest / f"checkpoint_step_{step:06d}"
    (step_dir / "generator").mkdir(parents=True, exist_ok=True)
    (step_dir / "discriminator").mkdir(parents=True, exist_ok=True)
    generator.save_pretrained(step_dir / "generator")
    g_tok.save_pretrained(step_dir / "generator")
    discriminator.save_pretrained(step_dir / "discriminator")
    d_tok.save_pretrained(step_dir / "discriminator")
    print(f"Saved: {step_dir}", flush=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_urls = load_train_urls(args.train_csv, args.url_column)
    print(f"Loaded {len(real_urls):,} TRAIN URLs. No validation/test CSV is accessed.")
    print(f"Device: {device}")

    g_tok = AutoTokenizer.from_pretrained(args.gpt2_checkpoint)
    generator = AutoModelForCausalLM.from_pretrained(args.gpt2_checkpoint).to(device)
    if g_tok.eos_token_id is None:
        raise ValueError("GPT-2 tokenizer must define EOS; use your fine-tuned GPT-2 checkpoint")
    if g_tok.pad_token_id is None:
        g_tok.pad_token = g_tok.eos_token
    if len(g_tok) != generator.get_input_embeddings().num_embeddings:
        raise ValueError("GPT-2 tokenizer size differs from checkpoint's embedding size")

    d_tok = AutoTokenizer.from_pretrained(args.bert_checkpoint)
    discriminator = AutoModelForSequenceClassification.from_pretrained(
        args.bert_checkpoint, num_labels=2,
    ).to(device)
    if discriminator.config.num_labels != 2:
        raise ValueError("Discriminator must output exactly two logits: generated=0, real=1")
    # This explicit mapping prevents confusing this with phishing-vs-benign labels.
    discriminator.config.id2label = {0: "GENERATED", 1: "REAL"}
    discriminator.config.label2id = {"GENERATED": 0, "REAL": 1}

    prompt_ids = g_tok.encode("https://", add_special_tokens=False)
    if not prompt_ids:
        raise RuntimeError("Generator tokenizer could not tokenize the https:// prompt")
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    context_limit = getattr(generator.config, "n_positions", 1024)
    if prompt.shape[1] + args.max_new_tokens > context_limit:
        raise ValueError(f"Prompt + max-new-tokens exceeds GPT-2 context limit {context_limit}")

    g_opt = AdamW(generator.parameters(), lr=args.lr_generator)
    d_opt = AdamW(discriminator.parameters(), lr=args.lr_discriminator)
    baseline: float | None = None

    (args.output_dir / "config.json").write_text(
        json.dumps({**vars(args), "train_csv": str(args.train_csv),
                    "output_dir": str(args.output_dir), "num_train_urls": len(real_urls),
                    "gan_labels": {"0": "generated", "1": "real phishing"}},
                   indent=2), encoding="utf-8",
    )
    log_path = args.output_dir / "training_metrics.csv"
    fields = ["step", "d_loss", "d_accuracy", "g_loss", "mean_log_p_real",
              "unique_rate", "syntax_valid_rate", "avg_new_tokens"]

    with log_path.open("w", newline="", encoding="utf-8") as logfile:
        writer = csv.DictWriter(logfile, fieldnames=fields)
        writer.writeheader()

        for step in trange(1, args.steps + 1, desc="GAN training"):
            # (1) Samples stay offline; only the predefined TRAIN split is used.
            sampled_ids, active, fakes = sample_generator(
                generator, g_tok, prompt, args.batch_size,
                args.max_new_tokens, args.top_k, args.temperature,
            )
            reals = random.choices(real_urls, k=args.batch_size)

            # (2) Train BERT on REAL(1) vs GENERATED(0).
            discriminator.train()
            d_inputs = bert_batch(d_tok, reals + fakes, args.bert_max_length, device)
            labels = torch.cat((torch.ones(args.batch_size, dtype=torch.long, device=device),
                                torch.zeros(args.batch_size, dtype=torch.long, device=device)))
            d_opt.zero_grad(set_to_none=True)
            d_logits = discriminator(**d_inputs).logits
            d_loss = F.cross_entropy(d_logits, labels)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            d_opt.step()
            d_accuracy = (d_logits.argmax(dim=-1) == labels).float().mean().item()

            # (3) BERT is fixed when computing generator rewards.
            discriminator.eval()
            with torch.no_grad():
                fake_inputs = bert_batch(d_tok, fakes, args.bert_max_length, device)
                real_logits = discriminator(**fake_inputs).logits
                # Reward log P_D(real | fake); no static-detector evasion reward.
                rewards = F.log_softmax(real_logits, dim=-1)[:, 1].clamp(min=-10.0)

            # (4) Update GPT-2 with a sequence-level policy gradient.
            # Running baseline lowers variance; it is NOT a separate classifier.
            current_mean = rewards.mean().item()
            if baseline is None:
                baseline = current_mean
            advantage = (rewards - baseline).detach()
            baseline = 0.95 * baseline + 0.05 * current_mean

            g_opt.zero_grad(set_to_none=True)
            seq_logprobs = sampled_sequence_logprobs(
                generator, sampled_ids, active, prompt.shape[1],
                args.top_k, args.temperature, g_tok.pad_token_id, g_tok.eos_token_id,
            )
            g_loss = -(advantage * seq_logprobs).mean()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            g_opt.step()

            row = {
                "step": step,
                "d_loss": round(d_loss.item(), 6),
                "d_accuracy": round(d_accuracy, 4),
                "g_loss": round(g_loss.item(), 6),
                "mean_log_p_real": round(current_mean, 6),
                "unique_rate": round(len(set(fakes)) / len(fakes), 4),
                "syntax_valid_rate": round(sum(looks_like_url(s) for s in fakes) / len(fakes), 4),
                "avg_new_tokens": round(active.sum(dim=1).float().mean().item(), 2),
            }
            writer.writerow(row)
            if step % args.log_every == 0 or step == 1:
                logfile.flush()
                print(f"step={step} D_acc={row['d_accuracy']:.3f} "
                      f"D_loss={row['d_loss']:.3f} G_reward={row['mean_log_p_real']:.3f} "
                      f"valid={row['syntax_valid_rate']:.3f}", flush=True)

            if step % args.save_every == 0 or step == args.steps:
                save_models(generator, g_tok, discriminator, d_tok, args.output_dir, step)
                with (args.output_dir / f"examples_step_{step:06d}.txt").open(
                    "w", encoding="utf-8",
                ) as examples_file:
                    examples_file.write("\n".join(fakes) + "\n")

    print(f"Done. Metrics: {log_path}")
    print("IMPORTANT: Compare trained generator vs original GPT-2 on a separate, untouched evaluation set.")


if __name__ == "__main__":
    main()
