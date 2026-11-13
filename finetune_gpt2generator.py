
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

os.environ.setdefault("USE_TF", "0")

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = SCRIPT_DIR / "CleanedDATA" / "Train_2025.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "PreTrainedGenerator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--validation-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--phish-label", default="1")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--max-per-domain", type=int, default=20,
                        help="Cap on URLs per registrable domain (0 disables)")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--num-generate", type=int, default=1000)
    parser.add_argument("--generate-batch-size", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def is_well_formed_url(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if re.search(r"[\s\x00-\x1f\x7f]", value):
        return False
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.hostname)
            and "." in parsed.hostname
        )
    except ValueError:
        return False


def make_domain_fn():
    """Return (url -> registrable domain, method name)."""
    try:
        import tldextract

        # Bundled suffix-list snapshot only
        extractor = tldextract.TLDExtract(suffix_list_urls=())

        def domain_of(url: str) -> str:
            ext = extractor(url)
            return (ext.registered_domain or ext.domain or url).lower()

        return domain_of, "tldextract"
    except ImportError:
        def domain_of(url: str) -> str:
            host = (urlsplit(url).hostname or url).lower()
            return ".".join(host.split(".")[-2:])

        return domain_of, "last-two-labels (tldextract not installed)"


def load_urls(
    path: Path, url_column: str, label_column: str, phish_label: str
) -> tuple[list[str], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    frame = pd.read_csv(path, dtype={url_column: "string"}, low_memory=False)
    if url_column not in frame.columns:
        raise ValueError(f"'{path}' has no '{url_column}' column")

    raw_count = len(frame)
    if label_column in frame.columns:
        frame = frame[frame[label_column].astype(str).str.strip() == phish_label]
    phish_count = len(frame)

    # Preserve URL structure, including original protocol and query text.
    candidates = [str(item).strip() for item in frame[url_column].dropna()]
    filtered = [item for item in candidates if is_well_formed_url(item)]
    unique = list(dict.fromkeys(filtered))
    stats = {
        "raw_rows": raw_count,
        "phishing_label_rows": phish_count,
        "syntax_valid_rows": len(filtered),
        "unique_valid_rows": len(unique),
    }
    return unique, stats


def cap_per_domain(
    urls: list[str], domain_of, cap: int, rng: random.Random
) -> list[str]:
    if cap <= 0:
        return urls
    groups: dict[str, list[str]] = defaultdict(list)
    for url in urls:
        groups[domain_of(url)].append(url)
    kept = []
    for members in groups.values():
        if len(members) > cap:
            members = rng.sample(members, cap)
        kept.extend(members)
    rng.shuffle(kept)
    return kept


def split_by_domain(
    urls: list[str], domain_of, fraction: float, rng: random.Random
) -> tuple[list[str], list[str]]:
    """Domain-disjoint split: every registrable domain lands wholly in one side."""
    groups: dict[str, list[str]] = defaultdict(list)
    for url in urls:
        groups[domain_of(url)].append(url)
    domains = sorted(groups)
    rng.shuffle(domains)
    target = max(1, round(len(urls) * fraction))
    val, train = [], []
    for domain in domains:
        group = groups[domain]
        # A single huge domain (e.g. google.com) must not overshoot the val target.
        fits = len(val) < target and len(val) + len(group) <= target * 1.1
        (val if fits else train).extend(group)
    return train, val


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def encode_examples(
    urls: list[str], tokenizer, max_tokens: int
) -> tuple[list[dict[str, list[int]]], int]:
    # <|endoftext|> doubles as BOS so the model learns the first-token distribution.
    bos = eos = tokenizer.eos_token_id
    encoded = []
    overlong = 0
    for url in urls:
        ids = tokenizer.encode(url, add_special_tokens=False)
        if len(ids) + 2 > max_tokens:
            overlong += 1
            continue
        ids = [bos] + ids + [eos]
        encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return encoded, overlong


def version_compatible_kwargs(cls, kwargs: dict, renames: dict[str, str]) -> dict:
    """Map new-style argument names to old ones when the installed version predates them."""
    params = inspect.signature(cls.__init__).parameters
    out = {}
    for key, value in kwargs.items():
        if key not in params and key in renames:
            key = renames[key]
        out[key] = value
    return out


@torch.no_grad()
def generate_urls(model, tokenizer, args) -> list[str]:
    model.eval()
    device = next(model.parameters()).device
    set_seed(args.seed)
    urls = []
    remaining = args.num_generate
    while remaining > 0:
        n = min(args.generate_batch_size, remaining)
        prompt = torch.full((n, 1), tokenizer.eos_token_id, dtype=torch.long, device=device)
        output = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=True,
            top_p=args.top_p,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens - 1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[tokenizer.pad_token_id]],
        )
        urls.extend(
            tokenizer.decode(seq[1:], skip_special_tokens=True).strip() for seq in output
        )
        remaining -= n
    return urls


def main() -> None:
    args = parse_args()
    if args.max_tokens < 16:
        raise ValueError("--max-tokens must be at least 16")
    if not (0.0 < args.validation_fraction < 0.5):
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    if args.batch_size < 1 or args.gradient_accumulation < 1 or args.epochs <= 0:
        raise ValueError("Batch size, accumulation and epochs must be positive")

    set_seed(args.seed)
    rng = random.Random(args.seed)
    domain_of, domain_method = make_domain_fn()

    train_urls, train_stats = load_urls(
        args.train_csv, args.url_column, args.label_column, args.phish_label
    )
    train_urls = cap_per_domain(train_urls, domain_of, args.max_per_domain, rng)
    train_stats["after_domain_cap"] = len(train_urls)

    if args.validation_csv:
        val_urls, val_stats = load_urls(
            args.validation_csv, args.url_column, args.label_column, args.phish_label
        )
        val_urls = cap_per_domain(val_urls, domain_of, args.max_per_domain, rng)
        val_stats["after_domain_cap"] = len(val_urls)
        overlap = set(train_urls) & set(val_urls)
        if overlap:
            raise ValueError(
                f"Training/validation CSVs overlap on {len(overlap)} exact URLs. "
                "Resolve the split before proceeding."
            )
        domain_overlap = {domain_of(u) for u in train_urls} & {domain_of(u) for u in val_urls}
        if domain_overlap:
            raise ValueError(
                f"Training/validation CSVs share {len(domain_overlap)} registrable domains. "
                "Use a domain-disjoint split."
            )
    else:
        train_urls, val_urls = split_by_domain(
            train_urls, domain_of, args.validation_fraction, rng
        )
        val_stats = {"created_from_training_csv": len(val_urls)}

    if len(train_urls) < 2 or len(val_urls) < 1:
        raise ValueError("Insufficient URLs after data filtering/splitting")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("The selected tokenizer must have an EOS token")
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|url_pad|>"})

    train_rows, excluded_train = encode_examples(train_urls, tokenizer, args.max_tokens)
    val_rows, excluded_val = encode_examples(val_urls, tokenizer, args.max_tokens)
    if len(train_rows) < 2 or len(val_rows) < 1:
        raise ValueError(
            "Not enough URLs fit the token limit; increase --max-tokens "
            "or inspect your data."
        )

    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    old_vocab = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight
        embeddings[old_vocab:] = embeddings[:old_vocab].mean(dim=0, keepdim=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = tokenizer.eos_token_id

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs = version_compatible_kwargs(
        TrainingArguments,
        dict(
            output_dir=str(args.output_dir / "checkpoints"),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.05,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            logging_steps=25,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
        ),
        renames={"eval_strategy": "evaluation_strategy"},
    )
    training_args = TrainingArguments(**training_kwargs)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer_kwargs = version_compatible_kwargs(
        Trainer,
        dict(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            processing_class=tokenizer,
        ),
        renames={"processing_class": "tokenizer"},
    )
    trainer = Trainer(**trainer_kwargs)

    total_train = len(train_rows) + excluded_train
    print(f"Train CSV:           {args.train_csv}")
    print(f"Output dir:          {args.output_dir}")
    print(f"Domain grouping:     {domain_method}")
    print(f"Training examples:   {len(train_rows):,}")
    print(f"Validation examples: {len(val_rows):,}")
    print(
        f"Overlong exclusions: {excluded_train:,} train "
        f"({excluded_train / total_train:.1%}), {excluded_val:,} validation"
    )

    trainer.train()
    metrics = trainer.evaluate()
    eval_loss = float(metrics["eval_loss"])
    metrics["perplexity"] = math.exp(eval_loss) if eval_loss < 100 else float("inf")

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    generated = generate_urls(trainer.model, tokenizer, args)
    train_set = set(train_urls)
    gen_frame = pd.DataFrame({
        "url": generated,
        "well_formed": [is_well_formed_url(u) for u in generated],
        "exact_in_train": [u in train_set for u in generated],
    })
    gen_path = args.output_dir / "generated_urls.csv"
    gen_frame.to_csv(gen_path, index=False)

    manifest = {
        "model": args.model_name,
        "seed": args.seed,
        "train_csv": str(args.train_csv),
        "validation_csv": str(args.validation_csv) if args.validation_csv else None,
        "split": "domain-disjoint",
        "domain_grouping": domain_method,
        "max_per_domain": args.max_per_domain,
        "bos_token": "eos_token (<|endoftext|>) prepended to every example",
        "vocab_size": len(tokenizer),
        "max_tokens_including_bos_eos": args.max_tokens,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "train_file_sha256": sha256(args.train_csv),
        "validation_file_sha256": sha256(args.validation_csv) if args.validation_csv else None,
        "source_train_stats": train_stats,
        "source_validation_stats": val_stats,
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "excluded_overlong_train": excluded_train,
        "excluded_overlong_validation": excluded_val,
        "metrics": {
            "eval_loss": eval_loss,
            "perplexity": metrics["perplexity"] if math.isfinite(metrics["perplexity"]) else None,
        },
        "generation": {
            "file": gen_path.name,
            "num_generated": len(generated),
            "top_p": args.top_p,
            "temperature": args.temperature,
            "well_formed_rate": float(gen_frame["well_formed"].mean()),
            "unique_rate": gen_frame["url"].nunique() / max(1, len(gen_frame)),
            "exact_in_train_rate": float(gen_frame["exact_in_train"].mean()),
        },
        "notes": (
            "Held-out validation loss measures next-token prediction only. "
            "Generation rates here are quick sanity checks; near-duplicate, "
            "structural-novelty, distribution and detection metrics require "
            "a separate evaluation."
        ),
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Validation loss: {eval_loss:.4f}")
    print(f"Validation perplexity: {metrics['perplexity']:.3f}")
    print(f"Generated URLs: {len(generated):,} -> {gen_path}")
    print(f"Model/tokenizer/manifest saved: {args.output_dir}")


if __name__ == "__main__":
    main()
