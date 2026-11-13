"""Warm up a BERT discriminator for the URL GAN BEFORE adversarial training.

Task: REAL phishing (label 1) vs GENERATED URLs (label 0), exactly the label
convention gan_gpt2_bert_ppo.py uses. The point is to hand the GAN a
discriminator that already gives a usable reward signal, instead of a fresh head
that sits at 0.5 accuracy and starves the generator of gradient.

Deliberately stops EARLY, at a target validation accuracy (default 0.82), not at
convergence: a discriminator warmed to ~100% would score every generated URL as
fake and collapse the generator's reward. The GAN loop keeps training it from
there.

  REAL      = existing phishing TRAIN split (CleanedDATA/Train_2025.csv)
  GENERATED = your fine-tuned generator's output (PreTrainedGenerator/generated_urls.csv)

Uses bert-base-cased (keeps case and accents that distinguish many URLs), a
domain-disjoint split on the real URLs, and saves a checkpoint you pass to the
GAN via --bert-checkpoint. No test data is touched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from urllib.parse import urlsplit

os.environ.setdefault("USE_TF", "0")

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REAL_CSV = SCRIPT_DIR / "CleanedDATA" / "Train_2025.csv"
DEFAULT_GENERATED_CSV = SCRIPT_DIR / "PreTrainedGenerator" / "generated_urls.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "WarmedDiscriminator"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-csv", type=Path, default=DEFAULT_REAL_CSV)
    p.add_argument("--generated-csv", type=Path, default=DEFAULT_GENERATED_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--url-column", default="url")
    p.add_argument("--bert-checkpoint", default="bert-base-cased")
    p.add_argument("--target-accuracy", type=float, default=0.82,
                   help="Stop once validation accuracy reaches this (keeps the discriminator beatable)")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def registrable_domain(url: str) -> str:
    try:
        import tldextract
        ext = tldextract.TLDExtract(suffix_list_urls=())(url)
        value = getattr(ext, "top_domain_under_public_suffix", None) or ext.registered_domain
        return (value or urlsplit(url).hostname or url).lower()
    except ImportError:
        host = (urlsplit(url).hostname or url).lower()
        return ".".join(host.split(".")[-2:])


def load_column(path: Path, column: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    frame = pd.read_csv(path, dtype={column: "string"}, low_memory=False)
    if column not in frame.columns:
        raise ValueError(f"'{path}' has no '{column}' column")
    urls = [str(u).strip() for u in frame[column].dropna()]
    return list(dict.fromkeys(u for u in urls if u))


def domain_disjoint_split(urls: list[str], frac: float, rng: random.Random):
    groups: dict[str, list[str]] = {}
    for u in urls:
        groups.setdefault(registrable_domain(u), []).append(u)
    domains = sorted(groups)
    rng.shuffle(domains)
    target = max(1, round(len(urls) * frac))
    val, train = [], []
    for d in domains:
        (val if len(val) < target else train).extend(groups[d])
    return train, val


@torch.no_grad()
def evaluate(model, tokenizer, urls, labels, args, device) -> tuple[float, float]:
    model.eval()
    correct = total_loss = 0
    for i in range(0, len(urls), args.batch_size):
        batch = urls[i:i + args.batch_size]
        y = torch.tensor(labels[i:i + args.batch_size], device=device)
        enc = tokenizer(batch, padding=True, truncation=True, max_length=args.max_length,
                        return_tensors="pt").to(device)
        logits = model(**enc).logits
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
    return correct / len(urls), total_loss / len(urls)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          "cpu" if args.device == "auto" else args.device)
    rng = random.Random(args.seed)

    real = load_column(args.real_csv, args.url_column)
    generated = load_column(args.generated_csv, args.url_column)
    print(f"Real phishing: {len(real):,}   Generated: {len(generated):,}")
    if len(generated) < 20:
        raise ValueError("Need at least ~20 generated URLs to warm up a discriminator")

    # Balance to the smaller (generated) class so accuracy is meaningful.
    n = min(len(real), len(generated))
    real = rng.sample(real, n)
    generated = rng.sample(generated, n)

    real_train, real_val = domain_disjoint_split(real, args.val_fraction, rng)
    gen_train, gen_val = generated[int(n * args.val_fraction):], generated[:int(n * args.val_fraction)]

    train = [(u, 1) for u in real_train] + [(u, 0) for u in gen_train]
    val = [(u, 1) for u in real_val] + [(u, 0) for u in gen_val]
    rng.shuffle(train)
    rng.shuffle(val)
    val_urls, val_labels = [u for u, _ in val], [y for _, y in val]
    print(f"Train: {len(train):,}  Val: {len(val):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_checkpoint, num_labels=2).to(device)
    model.config.id2label = {0: "GENERATED", 1: "REAL"}
    model.config.label2id = {"GENERATED": 0, "REAL": 1}
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_acc, step, stop = 0.0, 0, False
    history = []
    while step < args.max_steps and not stop:
        rng.shuffle(train)
        for i in range(0, len(train), args.batch_size):
            if step >= args.max_steps:
                break
            model.train()
            batch = train[i:i + args.batch_size]
            urls = [u for u, _ in batch]
            y = torch.tensor([lbl for _, lbl in batch], device=device)
            enc = tokenizer(urls, padding=True, truncation=True, max_length=args.max_length,
                            return_tensors="pt").to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(**enc).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % args.eval_every == 0 or step == args.max_steps:
                acc, val_loss = evaluate(model, tokenizer, val_urls, val_labels, args, device)
                best_acc = max(best_acc, acc)
                history.append({"step": step, "train_loss": round(loss.item(), 4),
                                "val_accuracy": round(acc, 4), "val_loss": round(val_loss, 4)})
                print(f"step={step}  train_loss={loss.item():.4f}  val_acc={acc:.4f}", flush=True)
                if acc >= args.target_accuracy:
                    print(f"Reached target accuracy {args.target_accuracy}; stopping early.")
                    stop = True
                    break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    final_acc, final_loss = evaluate(model, tokenizer, val_urls, val_labels, args, device)
    (args.output_dir / "warmup_manifest.json").write_text(json.dumps({
        "bert_checkpoint": args.bert_checkpoint,
        "real_csv": str(args.real_csv),
        "generated_csv": str(args.generated_csv),
        "label_convention": {"0": "GENERATED", "1": "REAL"},
        "balanced_class_size": n,
        "steps_trained": step,
        "target_accuracy": args.target_accuracy,
        "final_val_accuracy": final_acc,
        "final_val_loss": final_loss,
        "best_val_accuracy": best_acc,
        "seed": args.seed,
        "history": history,
        "note": "Pass --bert-checkpoint <this dir> to gan_gpt2_bert_ppo.py.",
    }, indent=2), encoding="utf-8")
    print(f"\nFinal val accuracy: {final_acc:.4f}  (perplexity-free; classification)")
    print(f"Saved warmed discriminator to {args.output_dir}")
    print("Use it in the GAN with:  --bert-checkpoint \"%s\"" % args.output_dir)


if __name__ == "__main__":
    main()
