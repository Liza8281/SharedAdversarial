
from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging
import CLeanURLNet as U
from gan_gpt2_bert_train import sample_generator, load_train_urls
from adversarial_loop import (
    generator_policy_update, compute_R_BERT, update_discriminator_3label,
    validate_generated, build_benign_filter, AttackArchive, schedule_weights, ValueHead,
    D_GENERATED, D_REAL_PHISH, D_BENIGN,
)

SCRIPT_DIR = Path(__file__).resolve().parent

@dataclass
class Config:
    # warm-start checkpoints
    gen_ckpt: Path = SCRIPT_DIR / "PreTrainedGenerator"
    disc_ckpt: Path = SCRIPT_DIR / "WarmedDiscriminator"
    urlnet_ckpt: Path = SCRIPT_DIR / "CleanURLNET"       # trained URLNet (emb_mode 5)
    data_dir: Path = SCRIPT_DIR / "CleanedDATA"
    output_dir: Path = SCRIPT_DIR / "AdversarialLoop_URLNet"

    # schedule (beta_t = 1 - alpha_t; see adversarial_loop.schedule_weights)
    n_iterations: int = 30
    n_warm: int = 5              # alpha_t ramps 0 -> alpha_max over the first n_warm rounds
    alpha_max: float = 0.5       # URLNet-evasion weight at convergence; lower it (e.g. 0.3) to
                                 # keep realism DOMINANT at convergence (beta stays >= 0.7)
    lam: float = 0.1             # KL-to-reference penalty coefficient
    reward_mode: str = "add"     # "add" = alpha*R_evade + beta*R_realism (original v2);
                                 # "mul" = R_evade * R_realism (both required; anti reward-hacking)
    freeze_generator: bool = False  # True = AUGMENTED arm: generator stays frozen (no PPO step);
                                    # synthetic attacks come from the fixed pretrained generator.

    # generator phase  (consumed by the reused generator_policy_update)
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

    # 3-label discriminator update
    bert_max_length: int = 96
    disc_batch_size: int = 16
    disc_updates_per_round: int = 20
    lr_discriminator: float = 2e-5

    # attack archive + validation / realism gate
    archive_cap: int = 20_000
    attacks_per_round: int = 256
    realism_min: float = 0.3
    lr_urlnet: float = 1e-4            # fine-tune LR
    urlnet_epochs_per_round: int = 1
    harden_batch_size: int = 128
    harden_phish_per_round: int = 4000
    harden_benign_per_round: int = 4000
    decision_threshold: float = 0.5

    seed: int = 42
    device: str = "auto"


@torch.no_grad()
def urlnet_phish_prob(model: "U.URLNet", urls: list[str], batch: int = 512) -> np.ndarray:
    """P(class == phishing) under URLNet. Uses URLNet's OWN encoder/vocabs (U._encode)."""
    if not urls:
        return np.zeros(0, dtype=np.float32)
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(urls), batch):
        enc = [U._encode(u) for u in urls[i:i + batch]]
        x_char = torch.as_tensor(np.stack([e[0] for e in enc]), device=dev)
        x_word = torch.as_tensor(np.stack([e[1] for e in enc]), device=dev)
        x_wc   = torch.as_tensor(np.stack([e[2] for e in enc]), device=dev)
        probs = torch.softmax(model(x_char, x_word, x_wc), dim=1)[:, 1]
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out)


def compute_R_URLNET(model: "U.URLNet", urls: list[str]) -> np.ndarray:
    """R_URLNET = 1 - P_phish  (high when URLNet is fooled). In [0,1]."""
    return 1.0 - urlnet_phish_prob(model, urls)


def _prf(pp: np.ndarray, pb: np.ndarray, thr: float = 0.5) -> dict:
    """recall / benign_acc / precision / f1 from phish-set and benign-set phish-probs."""
    tp = int((pp >= thr).sum()); fn = int((pp < thr).sum())
    fp = int((pb >= thr).sum()); tn = int((pb < thr).sum())
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    benignacc = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"recall": round(recall, 4), "benign_acc": round(benignacc, 4),
            "precision": round(precision, 4), "f1": round(f1, 4)}


def evaluate_urlnet(model: "U.URLNet", data_dir: Path, thr: float = 0.5) -> dict:
    """Held-out evaluation on the REAL test sets at the fixed operating threshold."""
    benign, _ = U._load_urls(data_dir / "Test_Benign.csv", 0)
    pb = urlnet_phish_prob(model, benign)
    res = {}
    for name, fname in [("Test_2025", "Test_2025.csv"), ("Test_2026", "Test_2026.csv")]:
        phish, _ = U._load_urls(data_dir / fname, 1)
        pp = urlnet_phish_prob(model, phish)
        res[name] = _prf(pp, pb, thr)
    return res


def resolve_threshold(cfg: Config) -> float:
    """URLNet operating threshold: cfg value if >=0, else the checkpoint's optimal_threshold.txt,
    else 0.5."""
    if cfg.decision_threshold is not None and cfg.decision_threshold >= 0:
        return float(cfg.decision_threshold)
    tf = cfg.urlnet_ckpt / "optimal_threshold.txt"
    if tf.exists():
        try:
            return float(tf.read_text().strip())
        except Exception:
            pass
    return 0.5


def harden_urlnet(model: "U.URLNet", optimizer, cfg: Config, rng: random.Random,
                  benign: list[str], real_phish: list[str], archive_attacks: list[str]) -> dict:

    phish = list(archive_attacks) + rng.sample(real_phish, min(cfg.harden_phish_per_round, len(real_phish)))
    n_benign = min(len(benign), max(cfg.harden_benign_per_round, len(phish)))
    benign_s = rng.sample(benign, n_benign)

    urls   = benign_s + phish
    labels = [0] * len(benign_s) + [1] * len(phish)
    ds = U.URLNetDataset(urls, labels)                       # uses URLNet's encoder + current vocabs
    loader = DataLoader(ds, batch_size=cfg.harden_batch_size, shuffle=True, num_workers=0)
    loss_fn = nn.CrossEntropyLoss()
    dev = next(model.parameters()).device

    model.train()
    last_loss, correct, seen = 0.0, 0, 0
    for _ in range(cfg.urlnet_epochs_per_round):
        for x_char, x_word, x_wc, y in loader:
            x_char, x_word, x_wc, y = x_char.to(dev), x_word.to(dev), x_wc.to(dev), y.to(dev)
            optimizer.zero_grad()
            logits = model(x_char, x_word, x_wc)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
            correct += (logits.argmax(1) == y).sum().item(); seen += len(y)
    return {"h_loss": round(last_loss, 4), "h_acc": round(correct / seen, 4) if seen else float("nan"),
            "n_phish": len(phish), "n_benign": len(benign_s)}


# Realism admission gate (requirement from the DRL run)
def realism_filter(urls: list[str], discriminator, d_tok, cfg: Config, device) -> tuple[list[str], int]:

    if cfg.realism_min <= 0 or not urls:
        return urls, 0
    r = compute_R_BERT(discriminator, d_tok, urls, cfg.bert_max_length, device)
    keep = r >= cfg.realism_min
    kept = [u for u, k in zip(urls, keep) if k]
    return kept, int((~keep).sum())


# Main loop

def main(cfg: Config = Config()):
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(
        "cuda" if (cfg.device == "auto" and torch.cuda.is_available())
        else "cpu" if cfg.device == "auto" else cfg.device)
    U.DEVICE = device  # make URLNet's module use the same device
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "config.json").write_text(
        json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2), encoding="utf-8")
    rng = random.Random(cfg.seed)
    print(f"Device: {device}  |  output: {cfg.output_dir.name}")

    # generator (SFT) + frozen reference (CPU, VRAM saving)
    g_tok = AutoTokenizer.from_pretrained(str(cfg.gen_ckpt))
    if g_tok.pad_token_id is None:
        g_tok.pad_token = g_tok.eos_token
    generator = AutoModelForCausalLM.from_pretrained(str(cfg.gen_ckpt)).to(device)
    pi_ref = AutoModelForCausalLM.from_pretrained(str(cfg.gen_ckpt)).to("cpu")  # frozen KL anchor (CPU)
    pi_ref.eval()
    for p in pi_ref.parameters():
        p.requires_grad_(False)

    #3-label discriminator (warm BERT body + fresh 3-way head)
    d_tok = AutoTokenizer.from_pretrained(str(cfg.disc_ckpt))
    print("Discriminator: warm BERT body + fresh 3-class head "
          "(2->3 classifier reinit is EXPECTED; the head trains in-loop).")
    _v = hf_logging.get_verbosity(); hf_logging.set_verbosity_error()
    try:
        discriminator = AutoModelForSequenceClassification.from_pretrained(
            str(cfg.disc_ckpt), num_labels=3, ignore_mismatched_sizes=True).to(device)
    finally:
        hf_logging.set_verbosity(_v)
    discriminator.config.id2label = {D_GENERATED: "GENERATED", D_REAL_PHISH: "REAL_PHISH", D_BENIGN: "BENIGN"}
    discriminator.config.label2id = {"GENERATED": D_GENERATED, "REAL_PHISH": D_REAL_PHISH, "BENIGN": D_BENIGN}

    #URLNet detector: hardened (trained) + frozen baseline (control)
    urlnet = U.load_model(str(cfg.urlnet_ckpt))                 # sets U.char/word/ngram dicts
    urlnet.to(device)
    baseline = U.load_model(str(cfg.urlnet_ckpt))               # frozen control (same weights)
    baseline.to(device).eval()
    for p in baseline.parameters():
        p.requires_grad_(False)

    # optimizers
    value_head = ValueHead(generator.config.n_embd).to(device)
    actor_opt  = AdamW(generator.parameters(), lr=cfg.lr_generator)
    critic_opt = AdamW(value_head.parameters(), lr=cfg.lr_value)
    disc_opt   = AdamW(discriminator.parameters(), lr=cfg.lr_discriminator)
    urlnet_opt = Adam(urlnet.parameters(), lr=cfg.lr_urlnet)
    prompt = torch.tensor([g_tok.encode("https://", add_special_tokens=False)], dtype=torch.long, device=device)

    # data (known provenance)
    real_phish = load_train_urls(cfg.data_dir / "Train_2025.csv", "url")
    benign, _  = U._load_urls(cfg.data_dir / "Train_Benign.csv", 0)
    benign_url_set, benign_dom_set = build_benign_filter(benign)
    archive = AttackArchive(cfg.archive_cap, rng)
    thr = resolve_threshold(cfg)
    print(f"real_phish={len(real_phish):,}  benign={len(benign):,}  realism_gate>={cfg.realism_min}"
          f"  URLNet_threshold={thr:.4f}")

    # reward  (URLNet + discriminator FROZEN inside the generator hook)
    reward_urlnet_fn = lambda urls: compute_R_URLNET(urlnet, urls)
    reward_bert_fn   = lambda urls: compute_R_BERT(discriminator, d_tok, urls, cfg.bert_max_length, device)

    # --- logging ---
    log_path = cfg.output_dir / "loop_metrics.csv"
    fields = ["round", "alpha_t", "beta_t",
              "gen_actor_loss", "gen_approx_kl", "mean_R_URLNET", "mean_R_BERT", "fool_rate",
              "valid_rate", "n_dropped_invalid", "n_dropped_known_benign", "n_dropped_unrealistic",
              "archive_size", "d_loss", "d_accuracy", "h_loss", "h_acc",
              "H_2025_recall", "H_2025_benign", "H_2025_f1",
              "H_2026_recall", "H_2026_benign", "H_2026_f1",
              "B_2025_recall", "B_2025_f1", "B_2026_recall", "B_2026_f1"]
    logfile = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(logfile, fieldnames=fields); writer.writeheader()

    base_eval = evaluate_urlnet(baseline, cfg.data_dir, thr)
    (cfg.output_dir / "baseline_eval.json").write_text(json.dumps(base_eval, indent=2), encoding="utf-8")
    print(f"Baseline URLNet held-out (thr={thr:.4f}): {base_eval}")

    # iterative best-response
    for t in range(1, cfg.n_iterations + 1):
        alpha_t, beta_t = schedule_weights(t, cfg)

        if cfg.freeze_generator:
            gen_stats = {}          # AUGMENTED arm: generator stays frozen, no PPO update
        else:
            urlnet.eval()
            gen_stats = generator_policy_update(
                generator, g_tok, prompt, value_head, actor_opt, critic_opt, pi_ref,
                reward_urlnet_fn, reward_bert_fn, alpha_t, beta_t, cfg.lam, cfg, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        #Sample attacks -> validation gate -> REALISM gate -> archive.
        fakes: list[str] = []
        for _ in range(0, cfg.attacks_per_round, cfg.gen_batch_size):
            _, _, batch_urls = sample_generator(
                generator, g_tok, prompt, cfg.gen_batch_size, cfg.max_new_tokens, cfg.top_k, cfg.temperature)
            fakes.extend(batch_urls)
        valid_attacks, vstats = validate_generated(fakes, benign_url_set, benign_dom_set)
        realistic_attacks, n_unreal = realism_filter(valid_attacks, discriminator, d_tok, cfg, device)
        archive.add(realistic_attacks)

        # 3-label discriminator update
        d_stats = update_discriminator_3label(
            discriminator, d_tok, disc_opt, cfg, device,
            benign=benign, real_phish=real_phish, generated=realistic_attacks or valid_attacks or fakes)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # supervised adversarial fine-tuning with clean replay.
        h_stats = harden_urlnet(urlnet, urlnet_opt, cfg, rng, benign, real_phish, archive.all())
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # hardened vs frozen baseline (observational only).
        hard_eval = evaluate_urlnet(urlnet, cfg.data_dir, thr)

        mean_ru  = float(np.mean(reward_urlnet_fn(realistic_attacks))) if realistic_attacks else float("nan")
        mean_rb  = float(np.mean(reward_bert_fn(realistic_attacks))) if realistic_attacks else float("nan")

        if realistic_attacks:
            pp_att = urlnet_phish_prob(urlnet, realistic_attacks)
            fool_rate = round(float((pp_att < thr).mean()), 4)
        else:
            fool_rate = float("nan")
        row = {
            "round": t, "alpha_t": round(alpha_t, 4), "beta_t": round(beta_t, 4),
            "gen_actor_loss": gen_stats.get("actor_loss"), "gen_approx_kl": gen_stats.get("approx_kl"),
            "mean_R_URLNET": round(mean_ru, 4), "mean_R_BERT": round(mean_rb, 4), "fool_rate": fool_rate,
            "valid_rate": vstats["valid_rate"],
            "n_dropped_invalid": vstats["n_dropped_invalid"],
            "n_dropped_known_benign": vstats["n_dropped_known_benign"],
            "n_dropped_unrealistic": n_unreal,
            "archive_size": len(archive.all()),
            "d_loss": d_stats["d_loss"], "d_accuracy": d_stats["d_accuracy"],
            "h_loss": h_stats["h_loss"], "h_acc": h_stats["h_acc"],
            "H_2025_recall": hard_eval["Test_2025"]["recall"], "H_2025_benign": hard_eval["Test_2025"]["benign_acc"],
            "H_2025_f1": hard_eval["Test_2025"]["f1"],
            "H_2026_recall": hard_eval["Test_2026"]["recall"], "H_2026_benign": hard_eval["Test_2026"]["benign_acc"],
            "H_2026_f1": hard_eval["Test_2026"]["f1"],
            "B_2025_recall": base_eval["Test_2025"]["recall"], "B_2025_f1": base_eval["Test_2025"]["f1"],
            "B_2026_recall": base_eval["Test_2026"]["recall"], "B_2026_f1": base_eval["Test_2026"]["f1"],
        }
        writer.writerow(row); logfile.flush()
        print(f"[round {t:02d}] a={alpha_t:.2f} foolR_URLNET={row['mean_R_URLNET']} "
              f"valid={vstats['valid_rate']} dropped_unreal={n_unreal} archive={row['archive_size']} "
              f"HARD 2026 R/F1={row['H_2026_recall']}/{row['H_2026_f1']} "
              f"(base {row['B_2026_recall']}/{row['B_2026_f1']})", flush=True)

        # checkpoint
        ckpt = cfg.output_dir / f"round_{t:02d}"; ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(urlnet.state_dict(), ckpt / "urlnet.pt")
        generator.save_pretrained(ckpt / "generator"); g_tok.save_pretrained(ckpt / "generator")
        discriminator.save_pretrained(ckpt / "discriminator"); d_tok.save_pretrained(ckpt / "discriminator")
        (ckpt / "eval.json").write_text(
            json.dumps({"hardened": hard_eval, "baseline": base_eval}, indent=2), encoding="utf-8")

    logfile.close()
    print(f"Done. Per-round metrics: {log_path}")
    print("Headline = hardened URLNet vs frozen baseline recall/F1 on held-out REAL sets (H_* vs B_*).")


if __name__ == "__main__":
    main()
