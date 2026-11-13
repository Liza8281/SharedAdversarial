from __future__ import annotations
import csv
import json
import os
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GPT2Tokenizer
from transformers import logging as hf_logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor, set_random_seed
from gan_gpt2_bert_train import sample_generator, bert_batch, looks_like_url, load_train_urls
from gan_gpt2_bert_ppo import policy_token_stats, collect_old_policy, ValueHead
from drl_defer_v1 import (
    PhishingEnv, URLCnnExtractor, encode_batch, load_urls, normalize_url,
    REWARD_CONFIG as DRL_REWARD_V1,
    MAX_URL_LENGTH, EMBEDDING_DIM, FEATURES_DIM,
    ACTION_LEGIT, ACTION_PHISH, ACTION_DEFER,
    N_STEPS, BATCH_SIZE, N_EPOCHS, LR, GAMMA, GAE_LAMBDA, CLIP_RANGE,
    ENTROPY_COEF, VF_COEF, MAX_GRAD_NORM,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# 3-label discriminator mapping. R_BERT is P(REAL_PHISH). BERT is a reward signal, not a labeller.
D_GENERATED, D_REAL_PHISH, D_BENIGN = 0, 1, 2



# Config
@dataclass
class Config:
    # warm-start checkpoints
    gen_ckpt: Path = SCRIPT_DIR / "PreTrainedGenerator"
    disc_ckpt: Path = SCRIPT_DIR / "WarmedDiscriminator"
    detector_ckpt: Path = SCRIPT_DIR / "DRL_defer_v1" / "ppo_phish_detector.zip"
    data_dir: Path = SCRIPT_DIR / "CleanedDATA"
    output_dir: Path = SCRIPT_DIR / "AdversarialLoop_v1"
    n_iterations: int = 30
    n_warm: int = 5
    alpha_max: float = 0.5       # DRL-evasion weight at convergence
    lam: float = 0.1             # KL-to-reference penalty coefficient
    gen_rollouts_per_round: int = 50
    gen_batch_size: int = 8
    max_new_tokens: int = 64
    top_k: int = 50
    temperature: float = 1.0
    lr_generator: float = 1e-6
    lr_value: float = 1e-4
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    target_kl: float = 0.02
    bert_max_length: int = 96
    disc_batch_size: int = 16
    disc_updates_per_round: int = 20
    lr_discriminator: float = 2e-5
    detector_timesteps_per_round: int = 20_000
    archive_cap: int = 20_000
    attacks_per_round: int = 256
    seed: int = 42
    device: str = "auto"

def schedule_weights(t: int, cfg: Config) -> tuple[float, float]:
    if cfg.n_warm <= 0:
        alpha = cfg.alpha_max
    else:
        alpha = cfg.alpha_max * min(1.0, t / cfg.n_warm)
    return alpha, 1.0 - alpha


# Independent sample-validation gate
def _domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def build_benign_filter(benign_urls: list[str]) -> tuple[set[str], set[str]]:
    """Known-benign URL set and benign-domain set, used to reject fake 'evasions'."""
    url_set = {normalize_url(u) for u in benign_urls}
    dom_set = {_domain(u) for u in benign_urls if _domain(u)}
    return url_set, dom_set


def validate_generated(
    urls: list[str], benign_url_set: set[str], benign_dom_set: set[str],
) -> tuple[list[str], dict]:
    kept, n_invalid, n_benign = [], 0, 0
    for u in urls:
        if not looks_like_url(u):
            n_invalid += 1
            continue
        if normalize_url(u) in benign_url_set or _domain(u) in benign_dom_set:
            n_benign += 1
            continue
        kept.append(u)
    stats = {
        "n_generated": len(urls), "n_valid_phishing": len(kept),
        "n_dropped_invalid": n_invalid, "n_dropped_known_benign": n_benign,
        "valid_rate": round(len(kept) / len(urls), 4) if urls else 0.0,
    }
    return kept, stats


@torch.no_grad()
def detector_phish_prob(detector: PPO, g_tokenizer, urls: list[str]) -> np.ndarray:
    """P(action == phish) under the detector's (frozen) policy, in [0,1]."""
    if not urls:
        return np.zeros(0, dtype=np.float32)
    out = []
    for i in range(0, len(urls), 1024):
        obs = encode_batch(urls[i:i + 1024], g_tokenizer, MAX_URL_LENGTH)
        obs_t = obs_as_tensor(obs, detector.policy.device)
        dist = detector.policy.get_distribution(obs_t)
        probs = dist.distribution.probs[:, ACTION_PHISH]
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out)


def compute_R_DRL(detector: PPO, g_tokenizer, urls: list[str]) -> np.ndarray:
    """R_DRL = 1 - P_phish  (high when the detector is fooled). In [0,1]."""
    return 1.0 - detector_phish_prob(detector, g_tokenizer, urls)


@torch.no_grad()
def compute_R_BERT(discriminator, d_tokenizer, urls: list[str], max_length: int, device) -> np.ndarray:
    """R_BERT = P(class == REAL_PHISH) under the 3-label discriminator. In [0,1]. NEVER a label."""
    if not urls:
        return np.zeros(0, dtype=np.float32)
    out = []
    discriminator.eval()
    for i in range(0, len(urls), 256):
        batch = bert_batch(d_tokenizer, urls[i:i + 256], max_length, device)
        logits = discriminator(**batch).logits
        probs = F.softmax(logits, dim=-1)[:, D_REAL_PHISH]
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out)


# 3-label discriminator update
def update_discriminator_3label(
    discriminator, d_tokenizer, d_opt, cfg: Config, device,
    benign: list[str], real_phish: list[str], generated: list[str],
) -> dict:
    discriminator.train()
    bs = cfg.disc_batch_size
    losses, accs = [], []
    if not generated:
        return {"d_loss": float("nan"), "d_accuracy": float("nan")}
    for _ in range(cfg.disc_updates_per_round):
        g = random.choices(generated, k=bs)
        r = random.choices(real_phish, k=bs)
        b = random.choices(benign, k=bs)
        texts = g + r + b
        labels = torch.tensor(
            [D_GENERATED] * bs + [D_REAL_PHISH] * bs + [D_BENIGN] * bs,
            dtype=torch.long, device=device,
        )
        batch = bert_batch(d_tokenizer, texts, cfg.bert_max_length, device)
        d_opt.zero_grad(set_to_none=True)
        logits = discriminator(**batch).logits
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        d_opt.step()
        losses.append(loss.item())
        accs.append((logits.argmax(dim=-1) == labels).float().mean().item())
    return {"d_loss": round(float(np.mean(losses)), 6), "d_accuracy": round(float(np.mean(accs)), 4)}



# Generator PPO step
def generator_policy_update(
    generator, g_tokenizer, prompt, value_head, actor_opt, critic_opt, pi_ref,
    reward_drl_fn, reward_bert_fn, alpha_t: float, beta_t: float, lam: float, cfg: Config, device,
) -> dict:
    pad_id, eos_id = g_tokenizer.pad_token_id, g_tokenizer.eos_token_id
    prompt_len, top_k, temp = prompt.shape[1], cfg.top_k, cfg.temperature
    acc = {"actor_loss": [], "value_loss": [], "approx_kl": [], "kl_ref": [],
           "mean_R_DRL": [], "mean_R_BERT": [], "unique_rate": []}

    for _ in range(cfg.gen_rollouts_per_round):
        ids, active, urls = sample_generator(
            generator, g_tokenizer, prompt, cfg.gen_batch_size, cfg.max_new_tokens, top_k, temp)
        r_drl = torch.as_tensor(reward_drl_fn(urls), dtype=torch.float32, device=device)
        r_bert = torch.as_tensor(reward_bert_fn(urls), dtype=torch.float32, device=device)
        if getattr(cfg, "reward_mode", "add") == "mul":
            rewards = r_drl * r_bert
        else:
            rewards = alpha_t * r_drl + beta_t * r_bert
        old_logp, old_states = collect_old_policy(
            generator, ids, active, prompt_len, top_k, temp, pad_id, eos_id)
        with torch.no_grad():
            pi_ref.eval()
            ref_dev = next(pi_ref.parameters()).device
            ids_ref = ids.to(ref_dev)
            ref_logits = pi_ref(input_ids=ids_ref, use_cache=False).logits[:, prompt_len - 1:-1, :]
            ref_logp, _ = policy_token_stats(
                ref_logits, ids_ref[:, prompt_len:], active.to(ref_dev), top_k, temp, pad_id, eos_id,
                strict_support=False)
            ref_logp = ref_logp.to(device)
            old_values = value_head(old_states)
            advantages = rewards[:, None] - old_values
            sel = advantages[active]
            advantages = (advantages - sel.mean()) / sel.std(unbiased=False).clamp_min(1e-6)

        weights = active.to(torch.float32)
        count = weights.sum().clamp_min(1)
        last = {"actor_loss": 0.0, "value_loss": 0.0, "approx_kl": 0.0, "kl_ref": 0.0}

        for _epoch in range(cfg.ppo_epochs):
            generator.eval()                                             # dropout off; grads on
            actor_opt.zero_grad(set_to_none=True)
            lm_logits = generator(input_ids=ids, use_cache=False).logits[:, prompt_len - 1:-1, :]
            new_logp, entropy = policy_token_stats(
                lm_logits, ids[:, prompt_len:], active, top_k, temp, pad_id, eos_id,
                strict_support=False)
            log_ratio = new_logp - old_logp
            ratio = log_ratio.clamp(min=-20.0, max=20.0).exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantages
            surrogate = -(torch.minimum(unclipped, clipped) * weights).sum() / count
            mean_entropy = (entropy * weights).sum() / count
            kl_ref = ((new_logp - ref_logp) * weights).sum() / count     # approx KL(pi || pi_ref)
            actor_loss = surrogate - cfg.entropy_coef * mean_entropy + lam * kl_ref
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            actor_opt.step()

            # critic
            critic_opt.zero_grad(set_to_none=True)
            pred_values = value_head(old_states)
            value_loss = (((pred_values - rewards[:, None]) ** 2) * weights).sum() / count
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
            critic_opt.step()

            with torch.no_grad():
                approx_kl = (((ratio - 1) - log_ratio) * weights).sum() / count
            last = {"actor_loss": actor_loss.item(), "value_loss": value_loss.item(),
                    "approx_kl": approx_kl.item(), "kl_ref": kl_ref.item()}
            if cfg.target_kl > 0 and approx_kl.item() > cfg.target_kl:
                break

        acc["actor_loss"].append(last["actor_loss"]);   acc["value_loss"].append(last["value_loss"])
        acc["approx_kl"].append(last["approx_kl"]);      acc["kl_ref"].append(last["kl_ref"])
        acc["mean_R_DRL"].append(float(r_drl.mean()));   acc["mean_R_BERT"].append(float(r_bert.mean()))
        acc["unique_rate"].append(len(set(urls)) / len(urls) if urls else 0.0)

    return {k: round(float(np.mean(v)), 6) if v else float("nan") for k, v in acc.items()}



# Detector hardening
def harden_detector(
    detector: PPO, g_tokenizer, cfg: Config,
    benign: list[str], phish_plus_archive: list[str],
) -> None:

    def make_env():
        return Monitor(PhishingEnv(
            g_tokenizer, benign, phish_plus_archive,
            max_len=MAX_URL_LENGTH, seed=cfg.seed, reward_config=DRL_REWARD_V1.copy(),
        ))
    detector.set_env(DummyVecEnv([make_env]))
    detector.learn(total_timesteps=cfg.detector_timesteps_per_round, reset_num_timesteps=False)



# Independent real-data evaluation
@torch.no_grad()
def _predict_actions(detector: PPO, g_tokenizer, urls: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(urls), 1024):
        obs = encode_batch(urls[i:i + 1024], g_tokenizer, MAX_URL_LENGTH)
        actions, _ = detector.predict(obs, deterministic=True)
        out.append(np.asarray(actions).reshape(-1))
    return np.concatenate(out) if out else np.empty(0, int)


def evaluate_detector(detector: PPO, g_tokenizer, data_dir: Path) -> dict:
    """Committed metrics + coverage on the held-out REAL sets."""
    benign = load_urls(data_dir / "Test_Benign.csv", 0)
    ben_act = _predict_actions(detector, g_tokenizer, benign)
    tn = int((ben_act == ACTION_LEGIT).sum())
    fp = int((ben_act == ACTION_PHISH).sum())
    defer_b = int((ben_act == ACTION_DEFER).sum())

    results = {}
    for name, fn in [("Test_2025", "Test_2025.csv"), ("Test_2026", "Test_2026.csv")]:
        p = data_dir / fn
        if not p.exists():
            continue
        phish = load_urls(p, 1)
        ph_act = _predict_actions(detector, g_tokenizer, phish)
        tp = int((ph_act == ACTION_PHISH).sum())
        fn_ = int((ph_act == ACTION_LEGIT).sum())
        defer_p = int((ph_act == ACTION_DEFER).sum())

        committed = tp + fn_ + tn + fp
        recall = tp / (tp + fn_) if (tp + fn_) else float("nan")
        benign_acc = tn / (tn + fp) if (tn + fp) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else float("nan")
        results[name] = {
            "recall": round(recall, 4), "benign_acc": round(benign_acc, 4),
            "precision": round(precision, 4), "f1": round(f1, 4),
            "coverage": round(committed / (len(phish) + len(benign)), 4),
            "recall_strict": round(tp / len(phish), 4),
        }
    return results



# Attack archive

class AttackArchive:
    def __init__(self, cap: int, rng: random.Random):
        self.cap = cap
        self.rng = rng
        self.items: list[str] = []
        self._seen: set[str] = set()
        self._n_offered = 0

    def add(self, urls: list[str]) -> None:
        for u in urls:
            if u in self._seen:
                continue
            self._n_offered += 1
            if len(self.items) < self.cap:
                self.items.append(u)
                self._seen.add(u)
            else:  # reservoir replacement keeps a uniform sample of all attacks seen
                j = self.rng.randint(0, self._n_offered - 1)
                if j < self.cap:
                    self._seen.discard(self.items[j])
                    self.items[j] = u
                    self._seen.add(u)

    def all(self) -> list[str]:
        return list(self.items)



# Main loop.
def main(cfg: Config | None = None) -> None:
    cfg = cfg or Config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    set_random_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(
        "cuda" if (cfg.device == "auto" and torch.cuda.is_available())
        else "cpu" if cfg.device == "auto" else cfg.device
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "config.json").write_text(
        json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2), encoding="utf-8"
    )
    rng = random.Random(cfg.seed)
    print(f"Device: {device}  |  output: {cfg.output_dir.name}")

    # warm starts
    g_tok = AutoTokenizer.from_pretrained(str(cfg.gen_ckpt))
    if g_tok.pad_token_id is None:
        g_tok.pad_token = g_tok.eos_token
    generator = AutoModelForCausalLM.from_pretrained(str(cfg.gen_ckpt)).to(device)

    pi_ref = AutoModelForCausalLM.from_pretrained(str(cfg.gen_ckpt)).to("cpu")  # frozen KL anchor (CPU)
    pi_ref.eval()
    for p in pi_ref.parameters():
        p.requires_grad_(False)

    d_tok = AutoTokenizer.from_pretrained(str(cfg.disc_ckpt))
    print("Discriminator: warm BERT body + fresh 3-class head "
          "(2->3 classifier reinit is EXPECTED; the head trains in-loop).")
    _hf_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        discriminator = AutoModelForSequenceClassification.from_pretrained(
            str(cfg.disc_ckpt), num_labels=3, ignore_mismatched_sizes=True,
        ).to(device)
    finally:
        hf_logging.set_verbosity(_hf_verbosity)
    discriminator.config.id2label = {D_GENERATED: "GENERATED", D_REAL_PHISH: "REAL_PHISH", D_BENIGN: "BENIGN"}
    discriminator.config.label2id = {"GENERATED": D_GENERATED, "REAL_PHISH": D_REAL_PHISH, "BENIGN": D_BENIGN}

    # Detectors: hardened (trained) + frozen baseline (control). Detector encodes with GPT-2 tokenizer.
    det_tok = GPT2Tokenizer.from_pretrained("gpt2")
    det_tok.pad_token = det_tok.eos_token
    detector = PPO.load(str(cfg.detector_ckpt), device=device)
    detector.tensorboard_log = None
    baseline = PPO.load(str(cfg.detector_ckpt), device="cpu")  # never trained; the control

    from torch.optim import AdamW
    value_head = ValueHead(generator.config.n_embd).to(device)
    actor_opt = AdamW(generator.parameters(), lr=cfg.lr_generator)
    critic_opt = AdamW(value_head.parameters(), lr=cfg.lr_value)
    disc_opt = AdamW(discriminator.parameters(), lr=cfg.lr_discriminator)
    prompt = torch.tensor([g_tok.encode("https://", add_special_tokens=False)], dtype=torch.long, device=device)

    # data
    real_phish = load_train_urls(cfg.data_dir / "Train_2025.csv", "url")
    benign = load_urls(cfg.data_dir / "Train_Benign.csv", 0)
    benign_url_set, benign_dom_set = build_benign_filter(benign)
    archive = AttackArchive(cfg.archive_cap, rng)
    print(f"real_phish={len(real_phish):,}  benign={len(benign):,}")

    reward_drl_fn = lambda urls: compute_R_DRL(detector, det_tok, urls)
    reward_bert_fn = lambda urls: compute_R_BERT(discriminator, d_tok, urls, cfg.bert_max_length, device)

    log_path = cfg.output_dir / "loop_metrics.csv"
    fields = [
        "round", "alpha_t", "beta_t",
        "gen_actor_loss", "gen_approx_kl", "mean_R_DRL", "mean_R_BERT",
        "valid_rate", "n_dropped_invalid", "n_dropped_known_benign", "archive_size",
        "d_loss", "d_accuracy",
        "H_2025_recall", "H_2025_benign", "H_2025_f1",
        "H_2026_recall", "H_2026_benign", "H_2026_f1",
        "B_2025_recall", "B_2025_f1", "B_2026_recall", "B_2026_f1",
    ]
    logfile = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(logfile, fieldnames=fields)
    writer.writeheader()

    # Baseline (control) on the held-out real sets — computed once (it never changes).
    base_eval = evaluate_detector(baseline, det_tok, cfg.data_dir)
    (cfg.output_dir / "baseline_eval.json").write_text(json.dumps(base_eval, indent=2), encoding="utf-8")
    print(f"Baseline (non-adversarial) held-out: {base_eval}")

    for t in range(1, cfg.n_iterations + 1):
        alpha_t, beta_t = schedule_weights(t, cfg)


        detector.policy.set_training_mode(False)
        gen_stats = generator_policy_update(
            generator, g_tok, prompt, value_head, actor_opt, critic_opt, pi_ref,
            reward_drl_fn, reward_bert_fn, alpha_t, beta_t, cfg.lam, cfg, device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()  # release the generator-PPO graph before the detector PPO

        fakes: list[str] = []
        for _ in range(0, cfg.attacks_per_round, cfg.gen_batch_size):
            _, _, batch_urls = sample_generator(
                generator, g_tok, prompt, cfg.gen_batch_size,
                cfg.max_new_tokens, cfg.top_k, cfg.temperature,
            )
            fakes.extend(batch_urls)
        valid_attacks, vstats = validate_generated(fakes, benign_url_set, benign_dom_set)
        archive.add(valid_attacks)

        # 3-label discriminator update
        d_stats = update_discriminator_3label(
            discriminator, d_tok, disc_opt, cfg, device,
            benign=benign, real_phish=real_phish, generated=valid_attacks or fakes,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

        phish_train = real_phish + archive.all()
        harden_detector(detector, det_tok, cfg, benign, phish_train)
        if device.type == "cuda":
            torch.cuda.empty_cache()  # release the detector-PPO rollout/optim buffers

        hard_eval = evaluate_detector(detector, det_tok, cfg.data_dir)

        # diagnostics
        mean_rdrl = float(np.mean(reward_drl_fn(valid_attacks))) if valid_attacks else float("nan")
        mean_rbert = float(np.mean(reward_bert_fn(valid_attacks))) if valid_attacks else float("nan")
        row = {
            "round": t, "alpha_t": round(alpha_t, 4), "beta_t": round(beta_t, 4),
            "gen_actor_loss": gen_stats.get("actor_loss"),
            "gen_approx_kl": gen_stats.get("approx_kl"),
            "mean_R_DRL": round(mean_rdrl, 4), "mean_R_BERT": round(mean_rbert, 4),
            "valid_rate": vstats["valid_rate"],
            "n_dropped_invalid": vstats["n_dropped_invalid"],
            "n_dropped_known_benign": vstats["n_dropped_known_benign"],
            "archive_size": len(archive.all()),
            "d_loss": d_stats["d_loss"], "d_accuracy": d_stats["d_accuracy"],
            "H_2025_recall": hard_eval.get("Test_2025", {}).get("recall"),
            "H_2025_benign": hard_eval.get("Test_2025", {}).get("benign_acc"),
            "H_2025_f1": hard_eval.get("Test_2025", {}).get("f1"),
            "H_2026_recall": hard_eval.get("Test_2026", {}).get("recall"),
            "H_2026_benign": hard_eval.get("Test_2026", {}).get("benign_acc"),
            "H_2026_f1": hard_eval.get("Test_2026", {}).get("f1"),
            "B_2025_recall": base_eval.get("Test_2025", {}).get("recall"),
            "B_2025_f1": base_eval.get("Test_2025", {}).get("f1"),
            "B_2026_recall": base_eval.get("Test_2026", {}).get("recall"),
            "B_2026_f1": base_eval.get("Test_2026", {}).get("f1"),
        }
        writer.writerow(row)
        logfile.flush()
        print(f"[round {t:02d}] a={alpha_t:.2f} validRate={vstats['valid_rate']:.2f} "
              f"archive={len(archive.all())} D_acc={d_stats['d_accuracy']} "
              f"HARD 2025 R/F1={row['H_2025_recall']}/{row['H_2025_f1']} "
              f"(base {row['B_2025_recall']}/{row['B_2025_f1']})", flush=True)

        ckpt = cfg.output_dir / f"round_{t:02d}"
        ckpt.mkdir(parents=True, exist_ok=True)
        detector.save(str(ckpt / "detector"))
        generator.save_pretrained(ckpt / "generator")
        g_tok.save_pretrained(ckpt / "generator")
        discriminator.save_pretrained(ckpt / "discriminator")
        d_tok.save_pretrained(ckpt / "discriminator")
        (ckpt / "eval.json").write_text(
            json.dumps({"hardened": hard_eval, "baseline": base_eval}, indent=2), encoding="utf-8")

    logfile.close()
    print(f"Done. Per-round metrics: {log_path}")
    print("Headline = hardened vs baseline recall/F1 on held-out REAL sets (B_* vs H_* columns).")


if __name__ == "__main__":
    main()
