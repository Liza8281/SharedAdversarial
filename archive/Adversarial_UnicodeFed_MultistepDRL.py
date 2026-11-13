"""Archival: earlier co-evolutionary loop (GPT-2 generator vs multistep DRL detector).

Kept for reference only. This script is NOT part of the runnable pipeline in the
repository root: it expects externally trained models (a fine-tuned GPT-2 generator,
a 3-label BERT discriminator, a prebuilt multistep DRL policy, and a char+word URLNet
checkpoint) and the earlier 70/30 data split, none of which are included here. Update
the paths in Config to point at your own artifacts before running.
"""
from __future__ import annotations
import copy
import json
import os
import random
import re
import sys
import time
import traceback
import unicodedata
from collections import Counter, deque, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse, urlsplit, urlunsplit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.optim import Adam
from transformers import BertForSequenceClassification, BertTokenizer, GPT2Tokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import numpy.core as _np_core
sys.modules.setdefault("numpy._core", _np_core)
for _sub in ["multiarray", "umath", "_multiarray_umath", "numeric"]:
    try:
        sys.modules.setdefault(f"numpy._core.{_sub}", __import__(f"numpy.core.{_sub}", fromlist=[_sub]))
    except ImportError:
        pass


@dataclass(frozen=True)
class Config:
    seed: int = 42

    # Edit these paths to point at your own model artifacts.
    pretrained_gen:  str = "models/gpt2_generator"
    pretrained_disc: str = "models/finetunedBERT_discriminator"

    pretrained_multistep_drl: str = "models/DRL_Unicode_Fed/best_multistep_drl.zip"
    pretrained_cnn: str = "models/URLNet/best_model.pt"
    cnn_vocab_path: str = "models/URLNet/word_vocab.json"
    require_cnn_query: bool = True
    cnn_query_fallback: float = 0.5

    aux_dim: int = 20
    mask_dim: int = 3
    max_steps: int = 4

    query_struct_cost: float = -0.01
    query_cnn_cost: float = -0.03
    query_bert_cost: float = -0.03
    repeat_query_penalty: float = -0.10
    unavailable_query_penalty: float = -0.10
    timeout_defer_penalty_extra: float = -0.05
    evidence_dropout_both_available: float = 0.60
    evidence_dropout_cnn_unavailable: float = 0.25
    evidence_dropout_bert_unavailable: float = 0.10
    evidence_dropout_both_unavailable: float = 0.05
    info_gain_weight: float = 0.0
    benign_data:     str = "data/legit_train70.csv"
    phishing_data:   str = "data/phish_train70.csv"
    output_dir:      str = "output/Adversarial_UnicodeFed_output"
    num_iterations: int = 50
    # Gentle adversarial fine-tuning of an already-trained multistep policy.
    drl_timesteps:  int = 1024
    drl_n_steps:    int = 512
    drl_batch_size: int = 128
    drl_ent_coef:   float = 0.005
    drl_n_epochs:    int = 4
    drl_clip_range:  float = 0.10
    drl_target_kl:   float = 0.02
    gan_batch_size: int = 16
    gan_batches:    int = 20
    disc_batches:   int = 10
    gan_pool_size:  int = 200
    eval_size:      int = 500
    holdout:        int = 500
    val_benign_size: int = 100
    max_obs_tokens:     int = 80
    gen_max_new_tokens: int = 48
    prompt:             str = "https://"
    enable_homoglyph_augmentation: bool = True
    homoglyph_probability: float = 0.20
    homoglyph_max_replacements: int = 1
    homoglyph_mutate_tld: bool = False
    homoglyph_require_idna_roundtrip: bool = True
    homoglyph_use_skeleton_novelty: bool = True
    save_homoglyph_examples: bool = True
    lr_gen:  float = 2e-5
    lr_disc: float = 2e-5
    lr_drl:  float = 3e-5
    gan_smooth: float = 0.10   # fixed soft label for generated URLs
    replay_buffer_size: int = 500
    replay_ratio:       float = 0.10
    reward_tp: float = 1.0
    reward_tn: float = 1.0
    reward_fp: float = -1.0   # In another version: was -0.5 -- half of reward_tn's magnitude, never equalized when reward_tn was fixed 0.5->1.0. That gap is exactly what made blind-guess-phishing beat both deferring and guessing-benign under uncertainty
    reward_fn: float = -2.0
    fp_guard_multiplier: float = 2.0   # FP penalty x this when benign acc < threshold
    fp_guard_threshold:  float = 0.85
    defer_phish_dataset:  float = 0.30    # defer on a KNOWN phishing URL (low: should catch it)
    defer_benign_dataset: float = -0.10   # defer on a KNOWN benign URL  (negative: should pass it)
    defer_phish_gan:      float = 0.45    # defer on an UNCERTAIN generated phish-ish URL (rewarded)
    defer_benign_gan:     float = 0.20    # defer on an UNCERTAIN generated benign-ish URL (rewarded)
    reward_mode: str = "static_match"        # "adversarial" or "static_match"
    static_r_tp:          float = 1.0
    static_r_tn:          float = 1.0   # was 0.5 -- undervalued vs static_r_tp for no stated reason
    static_r_fp:          float = -1.0  # was -0.5 -- half of static_r_tn's magnitude, never equalized when static_r_tn was fixed. Confirmed this specific gap made blind-guess-phishing (EV=+0.25) beat both deferring (EV=+0.10) and guess-benign (EV=-0.5) under pure uncertainty -- directly explains the dns.google/cache zero-evidence false positive and the near-0% defer rate across every checkpoint evaluated so far.
    static_r_fn_base:     float = -2.0
    static_fn_scale:      float = 0.0        # was 3.0 (dynamic escalation); fixed FN penalty now
    static_r_abstain:     float = -0.10
    static_disagreement_resolution_bonus: float = 0.30
    static_disagreement_aux17_threshold:  float = 0.30
    reset_fn_rate_each_iter: bool = True
    benign_ratio:   float = 0.50
    phishing_ratio: float = 0.30
    gan_ratio:      float = 0.10
    # replay_ratio is taken from replay_ratio above

    bert_gate:            float = 0.00
    invalid_url_reward:   float = -0.50
    benign_domain_reward: float = -0.30
    evasion_boost:        float = 3.0    # amplify the (1 - drl_detection) evasion term
    plausibility_weight:  float = 0.15   # weight on structural-plausibility term
    reward_clip:          float = 3.0    # clip normalised rewards before PPO
    novelty_ngram: int = 4
    novelty_reference_size: int = 5000
    novelty_veto_threshold: float = 0.72   # if max similarity >= this, reject from GAN pool
    novelty_penalty_reward: float = -0.25  # reward used when a generated URL is too train-like
    novelty_weight: float = 0.25           # reward contribution: 1 - max_train_phish_similarity
    use_novelty_veto: bool = True

    min_gate_phish_det: float = 0.93
    min_gate_benign_acc: float = 0.93
    max_gate_metric_drop: float = 0.03
    max_gate_f1_drop: float = 0.03

    kl_skip_limit:  float = 5.0    # skip a batch's update if its KL exceeds this
    kl_reset_limit: float = 10.0   # roll back generator to iter-start if KL exceeds this

    warmup_steps: int = 200
    nash_window:    int = 5
    nash_fool_tol:  float = 0.02
    nash_disc_tol:  float = 0.01
    nash_min_iter:  int = 15
    nash_min_mean_fool: float = 0.03


CFG = Config()

_dropout_weights = (
    CFG.evidence_dropout_both_available,
    CFG.evidence_dropout_cnn_unavailable,
    CFG.evidence_dropout_bert_unavailable,
    CFG.evidence_dropout_both_unavailable,
)
assert abs(sum(_dropout_weights) - 1.0) < 1e-9, "Evidence dropout weights must sum to 1.0"
EVIDENCE_AVAILABILITY_OUTCOMES = ["both", "cnn_down", "bert_down", "none"]
EVIDENCE_AVAILABILITY_WEIGHTS = list(_dropout_weights)

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(CFG.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


HOSTING_PLATFORM_SUFFIXES = {
    "weeblysite.com", "netlify.app", "webflow.io", "github.io", "glitch.me",
    "vercel.app", "pages.dev", "000webhostapp.com", "sites.google.com",
    # shorteners:
    "zmb.la", "q-r.to", "qrco.de", "ead.me", "u.to", "u9.to", "zim.it",
    "reurl.cc", "hjty.io", "bit.ly", "tinyurl.com", "t.co", "goo.gl",
}

INVALID_CHARS = set("*'\"\\|{}[]^`~<>")

# Common two-level public suffixes
TWO_LEVEL_SUFFIXES = {
    "co.uk", "ac.uk", "gov.uk", "org.uk", "com.au", "net.au", "org.au",
    "co.jp", "com.br", "com.cn", "com.np", "co.in", "co.nz", "co.za",
}

PHISH_NGRAM_REFERENCE: list[tuple[str, set[str]]] = []
PHISH_SKELETON_NGRAM_REFERENCE: list[tuple[str, set[str]]] = []

UNICODE_HOMOGLYPH_MAP: dict[str, tuple[str, ...]] = {
    "a": ("\u0430",),       # Cyrillic small a
    "c": ("\u0441",),       # Cyrillic small es
    "e": ("\u0435",),       # Cyrillic small ie
    "i": ("\u0456",),       # Cyrillic small i
    "j": ("\u0458",),       # Cyrillic small je
    "o": ("\u043e",),       # Cyrillic small o
    "p": ("\u0440",),       # Cyrillic small er
    "s": ("\u0455",),       # Cyrillic small dze
    "x": ("\u0445",),       # Cyrillic small ha
    "y": ("\u0443",),       # Cyrillic small u
}

ASCII_CONFUSABLE_MAP: dict[str, tuple[str, ...]] = {
    "o": ("0",),
    "l": ("1",),
    "i": ("1",),
    "s": ("5",),
    "a": ("4",),
    "e": ("3",),
    "b": ("6",),
    "g": ("9",),
    "t": ("7",),
    "z": ("2",),
}

HOMOGLYPH_MAP: dict[str, tuple[str, ...]] = {
    key: UNICODE_HOMOGLYPH_MAP.get(key, ()) + ASCII_CONFUSABLE_MAP.get(key, ())
    for key in set(UNICODE_HOMOGLYPH_MAP) | set(ASCII_CONFUSABLE_MAP)
}

HOMOGLYPH_REVERSE: dict[str, str] = {
    confusable: ascii_char
    for ascii_char, values in HOMOGLYPH_MAP.items()
    for confusable in values
}

HOMOGLYPH_METADATA: dict[str, dict] = {}
HOMOGLYPH_EXAMPLE_ROWS: list[dict] = []


def confusable_skeleton(text: str) -> str:
    return "".join(HOMOGLYPH_REVERSE.get(ch, ch) for ch in str(text)).casefold()


def _idna_ascii_hostname(hostname: str) -> Optional[str]:
    try:
        ascii_host = hostname.encode("idna").decode("ascii")
        decoded = ascii_host.encode("ascii").decode("idna")
        if not decoded:
            return None
        if len(ascii_host) > 253:
            return None
        if any(len(label) == 0 or len(label) > 63 for label in ascii_host.split(".")):
            return None
        return ascii_host
    except (UnicodeError, ValueError):
        return None


def mutate_url_with_homoglyph(
    url: str,
    rng: random.Random,
    max_replacements: int,
) -> tuple[str, dict]:
    metadata = {
        "source_url": url,
        "final_url": url,
        "is_homoglyph": False,
        "mutation_count": 0,
        "unicode_host": "",
        "punycode_host": "",
        "mutated_positions": "",
    }

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        if not hostname or not parsed.scheme:
            return url, metadata
        if "@" in parsed.netloc:
            return url, metadata

        labels = hostname.split(".")
        if len(labels) < 2:
            return url, metadata

        last_mutable_label = len(labels) if CFG.homoglyph_mutate_tld else len(labels) - 1
        candidates: list[tuple[int, int]] = []
        for label_index in range(max(0, last_mutable_label)):
            for char_index, ch in enumerate(labels[label_index]):
                if ch.casefold() in HOMOGLYPH_MAP:
                    candidates.append((label_index, char_index))

        if not candidates:
            return url, metadata

        rng.shuffle(candidates)
        replacement_budget = min(
            max(1, int(max_replacements)),
            len(candidates),
        )

        mutated_labels = [list(label) for label in labels]
        used_positions = []

        for label_index, char_index in candidates[:replacement_budget]:
            original = mutated_labels[label_index][char_index]
            choices = HOMOGLYPH_MAP.get(original.casefold())
            if not choices:
                continue
            replacement = rng.choice(choices)
            mutated_labels[label_index][char_index] = replacement
            used_positions.append(f"{label_index}:{char_index}:{original}->{replacement}")

        if not used_positions:
            return url, metadata

        unicode_host = ".".join("".join(label) for label in mutated_labels)
        punycode_host = _idna_ascii_hostname(unicode_host)

        if CFG.homoglyph_require_idna_roundtrip and not punycode_host:
            return url, metadata
        netloc = unicode_host
        if parsed.port is not None:
            netloc = f"{unicode_host}:{parsed.port}"

        mutated_url = urlunsplit(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )

        if mutated_url == url or not is_valid_url(mutated_url):
            return url, metadata

        metadata.update(
            {
                "final_url": mutated_url,
                "is_homoglyph": True,
                "mutation_count": len(used_positions),
                "unicode_host": unicode_host,
                "punycode_host": punycode_host or "",
                "mutated_positions": "|".join(used_positions),
            }
        )
        return mutated_url, metadata

    except (ValueError, UnicodeError):
        return url, metadata


def augment_urls_with_homoglyphs(
    urls: list[str],
    probability: Optional[float] = None,
    rng: Optional[random.Random] = None,
    context: str = "generated",
    iteration: Optional[int] = None,
) -> tuple[list[str], list[dict]]:
    if not urls:
        return [], []

    rng = rng or random.Random()
    probability = (
        CFG.homoglyph_probability
        if probability is None
        else float(probability)
    )

    output: list[str] = []
    rows: list[dict] = []
    seen: set[str] = set()

    for source_url in urls:
        final_url = source_url
        metadata = {
            "source_url": source_url,
            "final_url": source_url,
            "is_homoglyph": False,
            "mutation_count": 0,
            "unicode_host": "",
            "punycode_host": "",
            "mutated_positions": "",
        }

        if (
            CFG.enable_homoglyph_augmentation
            and rng.random() < probability
        ):
            final_url, metadata = mutate_url_with_homoglyph(
                source_url,
                rng=rng,
                max_replacements=CFG.homoglyph_max_replacements,
            )

        if final_url in seen:
            final_url = source_url
            metadata.update(
                {
                    "final_url": source_url,
                    "is_homoglyph": False,
                    "mutation_count": 0,
                    "unicode_host": "",
                    "punycode_host": "",
                    "mutated_positions": "",
                }
            )

        seen.add(final_url)
        output.append(final_url)

        metadata["context"] = context
        metadata["iteration"] = iteration
        rows.append(metadata)
        HOMOGLYPH_METADATA[final_url] = metadata

        if metadata["is_homoglyph"]:
            HOMOGLYPH_EXAMPLE_ROWS.append(dict(metadata))

    return output, rows


def is_homoglyph_url(url: str) -> bool:
    return bool(HOMOGLYPH_METADATA.get(url, {}).get("is_homoglyph", False))


def char_ngrams(text: str, n: int) -> set[str]:
    text = re.sub(r"\s+", "", str(text).lower())
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_from_sets(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def max_train_phish_similarity(url: str) -> float:
    if not PHISH_NGRAM_REFERENCE:
        return 0.0

    q_raw = char_ngrams(url, CFG.novelty_ngram)
    raw_similarity = (
        max(jaccard_from_sets(q_raw, ref_set) for _, ref_set in PHISH_NGRAM_REFERENCE)
        if q_raw
        else 0.0
    )

    if not CFG.homoglyph_use_skeleton_novelty or not PHISH_SKELETON_NGRAM_REFERENCE:
        return raw_similarity

    q_skeleton = char_ngrams(confusable_skeleton(url), CFG.novelty_ngram)
    skeleton_similarity = (
        max(
            jaccard_from_sets(q_skeleton, ref_set)
            for _, ref_set in PHISH_SKELETON_NGRAM_REFERENCE
        )
        if q_skeleton
        else 0.0
    )
    return max(raw_similarity, skeleton_similarity)


def novelty_score(url: str) -> float:
    return float(max(0.0, min(1.0, 1.0 - max_train_phish_similarity(url))))


def get_host(url: str) -> str:
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = parsed.netloc.lower().split("@")[-1].split(":")[0]
        return host.strip(".")
    except Exception:
        return ""


def get_registered_domain(url: str) -> str:
    host = get_host(url)
    if not host:
        return ""
    parts = [p for p in host.split(".") if p]
    if len(parts) < 2:
        return host
    if len(parts) >= 3 and ".".join(parts[-2:]) in TWO_LEVEL_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def is_valid_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return False
    if any(c in url for c in INVALID_CHARS):
        return False
    if len(url) < 12 or len(url) > 300:
        return False
    host = get_host(url)
    if "." not in host or len(host) < 4 or not any(c.isalpha() for c in host):
        return False
    return True


# Plausibility scoring (structural phishing cues; literature-based)
PHISH_BRANDS = {
    "paypal", "apple", "google", "microsoft", "amazon", "netflix", "facebook",
    "instagram", "linkedin", "twitter", "dropbox", "wellsfargo", "bankofamerica",
    "chase", "hsbc", "dhl", "fedex", "ups", "usps", "irs", "coinbase", "binance",
}
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online", "live",
    "click", "link", "support", "ru", "cn", "pw",
}


def phishing_plausibility_score(url: str) -> float:
    try:
        low = url.lower()
        no_scheme = re.sub(r"^https?://", "", low)
        parts = no_scheme.split("/")
        host = parts[0]
        path = "/".join(parts[1:]) if len(parts) > 1 else ""
        host_parts = host.split(".")
        tld = host_parts[-1] if host_parts else ""
        registered_base = host_parts[-2] if len(host_parts) >= 2 else host
        subdomains = host_parts[:-2] if len(host_parts) > 2 else []
        sub_str = ".".join(subdomains)

        brand_in_sub_or_path = any(b in sub_str or b in path for b in PHISH_BRANDS)
        brand_is_registered = any(b in registered_base for b in PHISH_BRANDS)
        brand_misuse = 1.0 if (brand_in_sub_or_path and not brand_is_registered) else 0.0
        suspicious_tld = 1.0 if tld in SUSPICIOUS_TLDS else 0.0
        subdomain_depth = 1.0 if len(subdomains) >= 2 else 0.0
        hyphen_in_domain = 1.0 if "-" in registered_base else 0.0
        has_at = 1.0 if "@" in url else 0.0
        long_path = 1.0 if len(path) > 40 else 0.0
        digit_in_domain = 1.0 if any(c.isdigit() for c in registered_base) else 0.0

        return float(
            brand_misuse * 0.35 + suspicious_tld * 0.20 + subdomain_depth * 0.15 +
            hyphen_in_domain * 0.10 + has_at * 0.08 + long_path * 0.07 +
            digit_in_domain * 0.05
        )
    except Exception:
        return 0.0


# The first 10 positions preserve the original multistep observation layout.
# Positions 10-13 add four Unicode-aware structural features:
#   10 = contains non-ASCII character
#   11 = mixed-script hostname
#   12 = contains IDNA/punycode label (xn--)
#   13 = confusable-character ratio

MULTISTEP_PHISH_BRANDS = {
    "paypal", "apple", "google", "microsoft", "amazon", "netflix", "facebook",
    "instagram", "linkedin", "twitter", "dropbox", "wellsfargo", "bankofamerica",
    "chase", "hsbc", "dhl", "fedex", "ups", "usps", "irs", "coinbase", "binance",
    "office", "outlook", "icloud", "verify", "secure", "account", "login",
}

MULTISTEP_SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online", "live",
    "click", "link", "support", "ru", "cn", "pw", "zip", "mov", "icu", "shop",
    "work", "cfd", "quest", "monster", "lol",
}


def _hostname_scripts(hostname: str) -> set[str]:
    scripts: set[str] = set()

    for ch in str(hostname):
        if not ch.isalpha():
            continue

        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            scripts.add("LATIN")
        elif "CYRILLIC" in name:
            scripts.add("CYRILLIC")
        elif "GREEK" in name:
            scripts.add("GREEK")
        elif name:
            scripts.add("OTHER")

    return scripts


def _unicode_structural_features(hostname: str) -> list[float]:
    host = str(hostname or "")
    host_lower = host.casefold()

    contains_non_ascii = 1.0 if any(ord(ch) > 127 for ch in host) else 0.0

    scripts = _hostname_scripts(host)
    mixed_script = 1.0 if len(scripts) > 1 else 0.0

    contains_punycode = 1.0 if any(
        label.startswith("xn--")
        for label in host_lower.split(".")
    ) else 0.0

    confusable_count = sum(
        1
        for ch in host
        if ch in HOMOGLYPH_REVERSE
    )
    confusable_ratio = min(
        confusable_count / max(1, len(host)),
        1.0,
    )

    return [
        contains_non_ascii,
        mixed_script,
        contains_punycode,
        float(confusable_ratio),
    ]


def entropy_confidence(p: float) -> float:

    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    h = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))  # in [0, 1]
    return float(np.clip(1.0 - h, 0.0, 1.0))


def multistep_structural_features(url: str) -> np.ndarray:

    low = str(url).strip().lower()
    host = get_host(low)
    parts = [p for p in host.split(".") if p]
    tld = parts[-1] if parts else ""
    registered_base = parts[-2] if len(parts) >= 2 else host
    subdomains = parts[:-2] if len(parts) > 2 else []
    sub_str = ".".join(subdomains)

    try:
        parsed = urlparse(low if "://" in low else "http://" + low)
        path = (parsed.path or "") + ("?" + parsed.query if parsed.query else "")
    except Exception:
        path = ""

    brand_in_sub_or_path = any(
        b in sub_str or b in path
        for b in MULTISTEP_PHISH_BRANDS
    )
    brand_is_registered = any(
        b in registered_base
        for b in MULTISTEP_PHISH_BRANDS
    )
    brand_misuse = 1.0 if (
        brand_in_sub_or_path and not brand_is_registered
    ) else 0.0

    suspicious_tld = 1.0 if tld in MULTISTEP_SUSPICIOUS_TLDS else 0.0
    subdomain_depth_norm = min(len(subdomains) / 4.0, 1.0)
    hyphen_in_domain = 1.0 if "-" in registered_base else 0.0
    has_at = 1.0 if "@" in low else 0.0
    long_path = 1.0 if len(path) > 40 else 0.0
    digit_in_domain = 1.0 if any(
        c.isdigit()
        for c in registered_base
    ) else 0.0
    ip_host = 1.0 if re.fullmatch(
        r"(?:\d{1,3}\.){3}\d{1,3}",
        host or "",
    ) else 0.0
    length_norm = min(len(low) / 200.0, 1.0)

    structural_score = float(
        brand_misuse * 0.35
        + suspicious_tld * 0.20
        + subdomain_depth_norm * 0.15
        + hyphen_in_domain * 0.10
        + has_at * 0.08
        + long_path * 0.07
        + digit_in_domain * 0.05
    )
    structural_score = float(
        np.clip(structural_score, 0.0, 1.0)
    )

    unicode_features = _unicode_structural_features(host)
    struct_confidence = entropy_confidence(structural_score)

    return np.asarray(
        [
            0.0,                     # 0: BERT hidden until query
            0.0,                     # 1: CNN hidden until query
            structural_score,        # 2
            length_norm,             # 3
            ip_host,                 # 4
            suspicious_tld,          # 5
            subdomain_depth_norm,    # 6
            has_at,                  # 7
            digit_in_domain,         # 8
            long_path,               # 9
            *unicode_features,       # 10-13
            struct_confidence,       # 14: always visible, paired with structural_score
            0.0,                     # 15: cnn_confidence, hidden until query_cnn
            0.0,                     # 16: bert_confidence, hidden until query_bert
            0.0,                     # 17: cnn/bert disagreement, 0.0 until both queried
        ],
        dtype=np.float32,
    )


class LRUEvidenceCache:
    """Separate caches preserve true selective querying."""
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._bert: OrderedDict[str, float] = OrderedDict()
        self._cnn: OrderedDict[str, float] = OrderedDict()
        self._struct: OrderedDict[str, np.ndarray] = OrderedDict()

    def _get(self, cache: OrderedDict, key: str):
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    def _put(self, cache: OrderedDict, key: str, value) -> None:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self.max_size:
            cache.popitem(last=False)

    def get_bert(self, url: str):
        return self._get(self._bert, url)

    def put_bert(self, url: str, value: float) -> None:
        self._put(self._bert, url, float(value))

    def get_cnn(self, url: str):
        return self._get(self._cnn, url)

    def put_cnn(self, url: str, value: float) -> None:
        self._put(self._cnn, url, float(value))

    def get_struct(self, url: str):
        value = self._get(self._struct, url)
        return None if value is None else value.copy()

    def put_struct(self, url: str, value: np.ndarray) -> None:
        self._put(self._struct, url, value.astype(np.float32).copy())

    def clear_bert(self) -> None:
        self._bert.clear()

    def stats(self) -> dict:
        return {
            "bert": len(self._bert),
            "cnn": len(self._cnn),
            "struct": len(self._struct),
        }


EVIDENCE_CACHE = LRUEvidenceCache()


def read_url_csv(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p)
    col = next((c for c in ["url", "URL", "urls", "link", "uri", "domain"] if c in df.columns),
               df.columns[0])
    urls = df[col].dropna().astype(str).str.strip().tolist()
    return [u for u in urls if u]


print("Loading data...")
benign_urls = read_url_csv(CFG.benign_data)
phishing_urls = read_url_csv(CFG.phishing_data)

if len(benign_urls) <= CFG.holdout or len(phishing_urls) <= CFG.holdout:
    raise ValueError(
        f"Need more than holdout={CFG.holdout} rows in both sets. "
        f"benign={len(benign_urls)}, phishing={len(phishing_urls)}."
    )

eval_benign = benign_urls[-CFG.holdout:]
eval_phishing = phishing_urls[-CFG.holdout:]
benign_urls = benign_urls[:-CFG.holdout]
phishing_urls = phishing_urls[:-CFG.holdout]


assert not (set(benign_urls) & set(eval_benign)), "Benign holdout leaked into training!"
assert not (set(phishing_urls) & set(eval_phishing)), "Phishing holdout leaked into training!"


phishing_domain_counts = Counter(get_registered_domain(u) for u in phishing_urls)
benign_domain_counts = Counter(get_registered_domain(u) for u in benign_urls)  # training only

benign_domains = set()
for domain, b_count in benign_domain_counts.items():
    if not domain:
        continue
    if phishing_domain_counts.get(domain, 0) > b_count * 5:  # mostly-phishing domain: so not to be veto
        continue
    benign_domains.add(domain)
benign_domains.update(HOSTING_PLATFORM_SUFFIXES)


val_benign = benign_urls[-min(CFG.val_benign_size, len(benign_urls)):]

# Novelty reference built from TRAINING phishing URLs only. This keeps the generator from getting rewarded for near-copies of the PhishTank training distribution.
_ref_n = min(CFG.novelty_reference_size, len(phishing_urls))
_phish_ref_sample = random.sample(phishing_urls, _ref_n) if _ref_n else []
PHISH_NGRAM_REFERENCE = [
    (u, char_ngrams(u, CFG.novelty_ngram))
    for u in _phish_ref_sample
]
PHISH_SKELETON_NGRAM_REFERENCE = [
    (u, char_ngrams(confusable_skeleton(u), CFG.novelty_ngram))
    for u in _phish_ref_sample
]


def is_benign_domain(url: str) -> bool:
    d = get_registered_domain(url)
    return bool(d) and (d in benign_domains or any(d.endswith("." + b) for b in benign_domains))


print(f"  Train benign       : {len(benign_urls)}")
print(f"  Train phishing     : {len(phishing_urls)}")
print(f"  Eval holdout       : {len(eval_benign)} benign, {len(eval_phishing)} phishing")
print(f"  Benign veto domains: {len(benign_domains)}")
print(f"  Novelty reference  : {len(PHISH_NGRAM_REFERENCE)} training phishing URLs")
print(f"  Homoglyph augment  : enabled={CFG.enable_homoglyph_augmentation}, "
      f"probability={CFG.homoglyph_probability:.0%}, max_replacements={CFG.homoglyph_max_replacements}")
print(f"  DRL auxiliary dim  : {CFG.aux_dim} (includes 4 Unicode structural features)")
print(f"  FP-guard val slice : {len(val_benign)} benign\n")


url_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
url_tokenizer.pad_token = url_tokenizer.eos_token
VOCAB_SIZE = url_tokenizer.vocab_size


def url_to_obs(url: str) -> np.ndarray:
    toks = url_tokenizer.encode(
        url, max_length=CFG.max_obs_tokens, padding="max_length", truncation=True
    )
    return np.asarray(toks, dtype=np.int32)


def batch_to_obs(urls: Iterable) -> np.ndarray:
    urls = list(urls)
    if not urls:
        return np.empty((0, CFG.max_obs_tokens), dtype=np.int32)
    return np.stack([url_to_obs(u) for u in urls])

print("Loading generator...")
gen_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gen_tokenizer.pad_token = gen_tokenizer.eos_token
gen_model = AutoModelForCausalLMWithValueHead.from_pretrained(CFG.pretrained_gen).to(device)

prompt_ids = gen_tokenizer.encode(CFG.prompt, return_tensors="pt", add_special_tokens=False).to(device)
prompt_len = int(prompt_ids.size(1))
if prompt_len == 0:
    prompt_ids = gen_tokenizer.encode("https", return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = int(prompt_ids.size(1))
    print(f"  [WARN] Empty prompt encoding; fallback prompt_len={prompt_len}")
print(f"  Prompt: {CFG.prompt!r} -> {prompt_len} tokens")


ppo_cfg = PPOConfig(
    model_name=CFG.pretrained_gen,
    learning_rate=CFG.lr_gen,
    batch_size=CFG.gan_batch_size,
    mini_batch_size=CFG.gan_batch_size,
    ppo_epochs=1,
    kl_penalty="kl",
    init_kl_coef=0.05,   # [FIX 5]
    target=2.0,          # [FIX 5]
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
    log_with=None,
)
ppo_trainer = PPOTrainer(config=ppo_cfg, model=gen_model, tokenizer=gen_tokenizer)
print("  Generator ready\n")


def clean_generated_text(text: str) -> str:
    return re.split(r"[\s'\"{}]", text.strip())[0].strip()


def generate_urls(n: int, temperature: float = 1.0, allow_fallback: bool = False, verbose: bool = True):
    if n <= 0:
        return [], 0
    gen_model.eval()
    valid = []
    seen = set()
    attempts, max_attempts = 0, max(n * 8, CFG.gan_batch_size * 4)

    while len(valid) < n and attempts < max_attempts:
        bs = min(CFG.gan_batch_size * 2, max((n - len(valid)) * 3, 1))
        with torch.no_grad():
            out = gen_model.pretrained_model.generate(
                input_ids=prompt_ids.repeat(bs, 1),
                max_new_tokens=CFG.gen_max_new_tokens,
                do_sample=True, temperature=temperature, top_k=100, top_p=0.95,
                pad_token_id=gen_tokenizer.eos_token_id, repetition_penalty=1.2,
            )
        # NOTE: do NOT clamp token IDs — GPT-2 IDs are already valid; clamping corrupts text.
        for u in (clean_generated_text(x) for x in gen_tokenizer.batch_decode(out, skip_special_tokens=True)):
            if is_valid_url(u) and u not in seen:
                valid.append(u)
                seen.add(u)
                if len(valid) >= n:
                    break
        attempts += bs

    n_genuine = min(len(valid), n)

    valid, homoglyph_rows = augment_urls_with_homoglyphs(
        valid[:n_genuine],
        probability=CFG.homoglyph_probability,
        rng=random,
        context="generate_urls",
    )

    if n_genuine < n and allow_fallback:
        valid.extend(random.choices(phishing_urls, k=n - n_genuine))
        if verbose:
            print(f"    [gen] genuine={n_genuine}/{n}, fallback={n - n_genuine}")
    else:
        valid = valid[:n]
        if verbose:
            n_homoglyph = sum(
                1 for u in valid[:n_genuine]
                if is_homoglyph_url(u)
            )
            print(
                f"    [gen] genuine={n_genuine}/{n}"
                + (" OK" if n_genuine == n else "")
                + f", homoglyph={n_homoglyph}/{n_genuine}"
            )
    return valid[:n], n_genuine



def gan_veto_reason(url: str) -> Optional[str]:

    if not is_valid_url(url):
        return "invalid"
    if is_benign_domain(url):
        return "benign_domain"
    if CFG.use_novelty_veto:
        sim = max_train_phish_similarity(url)
        if sim >= CFG.novelty_veto_threshold:
            return "near_train_phish"
    return None


def gan_ground_truth_label(url: str):

    if gan_veto_reason(url) is not None:
        return None
    return 1                 # generator is trained to produce phishing -> label phishing


def make_gan_pool(n: int, temperature: float = 1.0, verbose: bool = True):
    raw, raw_genuine = generate_urls(n, temperature=temperature, allow_fallback=False, verbose=verbose)
    kept, reasons = [], Counter()
    for u in raw[:raw_genuine]:
        reason = gan_veto_reason(u)
        if reason is None:
            kept.append(u)
        else:
            reasons[reason] += 1
    vetoed = sum(reasons.values())
    if verbose and vetoed:
        reason_text = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        print(f"    [gan-label] vetoed={vetoed}/{raw_genuine} ({reason_text})")
    return kept, raw_genuine, vetoed



# Discriminator (BERT)
print("Loading discriminator...")
disc_tokenizer = BertTokenizer.from_pretrained(CFG.pretrained_disc)
disc_model = BertForSequenceClassification.from_pretrained(CFG.pretrained_disc, num_labels=3).to(device)
opt_disc = Adam(disc_model.parameters(), lr=CFG.lr_disc)
print("  Discriminator ready\n")


def bert_score(urls: list) -> torch.Tensor:

    if not urls:
        return torch.empty(0, device=device)
    was_training = disc_model.training
    disc_model.eval()
    enc = disc_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = disc_model(**enc).logits
    if was_training:
        disc_model.train()
    return torch.softmax(logits, dim=1)[:, 1]   # P(real phishing)


def bert_detector_score(urls: list) -> torch.Tensor:

    if not urls:
        return torch.empty(0, device=device)
    was_training = disc_model.training
    disc_model.eval()
    enc = disc_tokenizer(urls, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = disc_model(**enc).logits
    if was_training:
        disc_model.train()
    probs = torch.softmax(logits, dim=1)
    return probs[:, 1] + probs[:, 2]   # P(real phishing) + P(GAN-generated)



# URLNet/CNN auxiliary scorer for Hybrid DRL
URL_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".-_~:/?#[]@!$&'()*+,;=%"
)
PAD_IDX = 0
UNK_IDX = 1
CHAR2IDX = {ch: idx + 2 for idx, ch in enumerate(URL_CHARS)}
CHAR_VOCAB_SIZE = len(CHAR2IDX) + 2
WORD_DELIMITERS = r"[.\-_~:/?#\[\]@!$&'()*+,;=%]"
WORD_VOCAB_SIZE = 10_000
MAX_CHAR_LEN = 200
MAX_WORD_LEN = 30
EMB_DIM_CHAR = 32
EMB_DIM_WORD = 64
CONV_FILTERS = 128
KERNEL_SIZES = [3, 5, 7]
FC_HIDDEN = 256
DROPOUT = 0.5

word2idx: dict = {}


def tokenise_chars(url: str) -> list[int]:
    indices = [CHAR2IDX.get(ch, UNK_IDX) for ch in str(url)[:MAX_CHAR_LEN]]
    indices += [PAD_IDX] * (MAX_CHAR_LEN - len(indices))
    return indices


def tokenise_words(url: str) -> list[int]:
    tokens = [t for t in re.split(WORD_DELIMITERS, str(url)) if t][:MAX_WORD_LEN]
    indices = [word2idx.get(t, UNK_IDX) for t in tokens]
    indices += [PAD_IDX] * (MAX_WORD_LEN - len(indices))
    return indices


class MultiKernelCNN(nn.Module):
    def __init__(self, emb_dim, n_filters, kernel_sizes, dropout):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(emb_dim, n_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return torch.cat(
            [F.adaptive_max_pool1d(b(x), 1).squeeze(-1) for b in self.branches],
            dim=1,
        )


class URLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.char_embedding = nn.Embedding(CHAR_VOCAB_SIZE, EMB_DIM_CHAR, padding_idx=PAD_IDX)
        self.char_cnn = MultiKernelCNN(EMB_DIM_CHAR, CONV_FILTERS, KERNEL_SIZES, DROPOUT)
        self.word_embedding = nn.Embedding(WORD_VOCAB_SIZE, EMB_DIM_WORD, padding_idx=PAD_IDX)
        self.word_cnn = MultiKernelCNN(EMB_DIM_WORD, CONV_FILTERS, KERNEL_SIZES, DROPOUT)
        combined_dim = 2 * len(KERNEL_SIZES) * CONV_FILTERS
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, FC_HIDDEN),
            nn.BatchNorm1d(FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FC_HIDDEN, 2),
        )

    def forward(self, x_char, x_word):
        char_out = self.char_cnn(self.char_embedding(x_char))
        word_out = self.word_cnn(self.word_embedding(x_word))
        return self.classifier(torch.cat([char_out, word_out], dim=1))


cnn_model = None
print("Loading URLNet/CNN query model for Multistep DRL...")
try:
    if Path(CFG.pretrained_cnn).exists() and Path(CFG.cnn_vocab_path).exists():
        with open(CFG.cnn_vocab_path, "r", encoding="utf-8") as f:
            word2idx = json.load(f)
        cnn_model = URLNet().to(device)
        cnn_model.load_state_dict(torch.load(CFG.pretrained_cnn, map_location=device))
        cnn_model.eval()
        print(f"  URLNet/CNN ready | vocab={len(word2idx)}")
    elif CFG.require_cnn_query:
        raise FileNotFoundError(
            f"Multistep checkpoint expects CNN evidence, but CNN files are missing: "
            f"model={CFG.pretrained_cnn}, vocab={CFG.cnn_vocab_path}"
        )
    else:
        print("  [WARN] URLNet/CNN files not found. Using configured CNN fallback.")
except Exception as e:
    if CFG.require_cnn_query:
        raise RuntimeError(
            f"Could not load required URLNet/CNN query model: {type(e).__name__}: {e}"
        ) from e
    print(f"  [WARN] Could not load URLNet/CNN ({type(e).__name__}: {e}). Using fallback.")
    cnn_model = None


@torch.no_grad()
def cnn_score(urls: list) -> torch.Tensor:
    if not urls:
        return torch.empty(0, device=device)
    if cnn_model is None:
        return torch.full((len(urls),), CFG.cnn_query_fallback, device=device)

    all_scores = []
    was_training = cnn_model.training
    cnn_model.eval()
    bs = 128
    for i in range(0, len(urls), bs):
        batch = urls[i:i + bs]
        x_char = torch.tensor([tokenise_chars(u) for u in batch], dtype=torch.long, device=device)
        x_word = torch.tensor([tokenise_words(u) for u in batch], dtype=torch.long, device=device)
        probs = torch.softmax(cnn_model(x_char, x_word), dim=1)[:, 1]
        all_scores.append(probs)
    if was_training:
        cnn_model.train()
    return torch.cat(all_scores, dim=0)


def query_structural_evidence(url: str) -> np.ndarray:
    cached = EVIDENCE_CACHE.get_struct(url)
    if cached is None:
        cached = multistep_structural_features(url)
        EVIDENCE_CACHE.put_struct(url, cached)
    return cached.copy()


def query_cnn_evidence(url: str) -> float:
    cached = EVIDENCE_CACHE.get_cnn(url)
    if cached is None:
        with torch.no_grad():
            cached = float(cnn_score([url]).item())
        EVIDENCE_CACHE.put_cnn(url, cached)
    return float(cached)


def query_bert_evidence(url: str) -> float:
    cached = EVIDENCE_CACHE.get_bert(url)
    if cached is None:
        with torch.no_grad():
            cached = float(bert_detector_score([url]).item())
        EVIDENCE_CACHE.put_bert(url, cached)
    return float(cached)


def train_discriminator(n_batches: int, smooth: float) -> float:
    if n_batches <= 0:
        return float("nan")
    disc_model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=smooth)
    losses = []
    n_each = max(2, CFG.gan_batch_size // 2)

    for _ in range(n_batches):
        real_phish   = random.sample(phishing_urls, n_each)
        benign_batch = random.sample(benign_urls, n_each)

        # Same veto/novelty filter as DRL GAN pool.
        gan_urls, _, _ = make_gan_pool(n_each, verbose=False)
        if not gan_urls:
            continue
        n_gan = len(gan_urls)

        texts  = real_phish + gan_urls + benign_batch
        labels = torch.cat([
            torch.full((n_each,), 1, dtype=torch.long),   # real phishing  → class 1
            torch.full((n_gan,),  2, dtype=torch.long),   # GAN-generated  → class 2
            torch.full((n_each,), 0, dtype=torch.long),   # benign         → class 0
        ]).to(device)

        enc  = disc_tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        loss = criterion(disc_model(**enc).logits, labels)
        opt_disc.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(disc_model.parameters(), 1.0)
        opt_disc.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")


# Replay buffer
replay_buffer = deque(maxlen=CFG.replay_buffer_size)


def update_replay_buffer(gan_urls: list, actions: np.ndarray) -> None:
    for url, action in zip(gan_urls, actions):
        if action == 0 and not is_benign_domain(url):   # fooled the detector AND genuinely adversarial
            replay_buffer.append(url)


def get_replay_batch(n: int) -> list:
    if not replay_buffer:
        return []
    return random.choices(list(replay_buffer), k=min(n, len(replay_buffer)))


# Multistep evidence-gathering environment
class PhishingEnv(gym.Env):
    metadata = {"render_modes": []}

    QUERY_STRUCT = 0
    QUERY_CNN = 1
    QUERY_BERT = 2
    CLASSIFY_BENIGN = 3
    CLASSIFY_PHISHING = 4
    DEFER = 5

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            "tokens": spaces.Box(
                low=0, high=VOCAB_SIZE - 1,
                shape=(CFG.max_obs_tokens,), dtype=np.int32,
            ),
            "aux": spaces.Box(
                low=0.0, high=1.0,
                shape=(CFG.aux_dim,), dtype=np.float32,
            ),
            "mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(CFG.mask_dim,), dtype=np.float32,
            ),
            "step": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,), dtype=np.float32,
            ),
        })

        self.gan_pool = []
        self.gan_pool_idx = 0
        self.replay_pool = []

        self.current_url = ""
        self.true_label = 0
        self.source = "benign"
        self.tokens = np.zeros((CFG.max_obs_tokens,), dtype=np.int32)
        self.aux = np.zeros((CFG.aux_dim,), dtype=np.float32)
        self.mask = np.zeros((CFG.mask_dim,), dtype=np.float32)
        self.current_step = 0
        self.query_cost_total = 0.0
        self.actions_taken = []

        self.benign_ratio = CFG.benign_ratio
        self.phishing_ratio = CFG.phishing_ratio
        self.gan_ratio = CFG.gan_ratio
        self.replay_ratio = CFG.replay_ratio
        self._recent_benign_acc = 1.0

        self._total_decisions = 0
        self._fn_count = 0


        self.cnn_available = True
        self.bert_available = True

    def reset_fn_rate(self):
        self._total_decisions = 0
        self._fn_count = 0

    def set_gan_pool(self, urls):
        self.gan_pool, self.gan_pool_idx = urls, 0

    def set_replay_pool(self, urls):
        self.replay_pool = urls

    def set_benign_acc(self, acc):
        self._recent_benign_acc = float(acc)

    def set_ratios(self, benign, phishing, gan, replay):
        total = benign + phishing + gan + replay
        self.benign_ratio = benign / total
        self.phishing_ratio = phishing / total
        self.gan_ratio = gan / total
        self.replay_ratio = replay / total

    def _next_gan_url(self) -> str:
        if self.gan_pool_idx >= len(self.gan_pool):
            self.gan_pool_idx = 0
        url = self.gan_pool[self.gan_pool_idx]
        self.gan_pool_idx += 1
        return url

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "tokens": self.tokens.astype(np.int32),
            "aux": np.nan_to_num(
                self.aux, nan=0.0, posinf=1.0, neginf=0.0
            ).clip(0.0, 1.0).astype(np.float32),
            "mask": self.mask.astype(np.float32),
            "step": np.asarray(
                [self.current_step / max(1, CFG.max_steps)], dtype=np.float32
            ),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        weights = [
            self.benign_ratio,
            self.phishing_ratio,
            self.gan_ratio,
            self.replay_ratio,
        ]
        source = random.choices(
            ["benign", "phishing", "gan", "replay"], weights=weights
        )[0]

        if source == "replay" and not self.replay_pool:
            source = "phishing"
        if source == "gan" and not self.gan_pool:
            source = "phishing"
        self.source = source

        if source == "benign":
            self.current_url, self.true_label = random.choice(benign_urls), 0
        elif source == "phishing":
            self.current_url, self.true_label = random.choice(phishing_urls), 1
        elif source == "gan":
            self.current_url = self._next_gan_url()
            label = gan_ground_truth_label(self.current_url)
            if label is None:
                self.current_url, self.true_label = random.choice(phishing_urls), 1
                self.source = "phishing"
            else:
                self.true_label = int(label)
        else:
            self.current_url, self.true_label = random.choice(self.replay_pool), 1
        forced_cnn = options.get("force_cnn_available") if options else None
        forced_bert = options.get("force_bert_available") if options else None
        if forced_cnn is None or forced_bert is None:
            outcome = random.choices(
                EVIDENCE_AVAILABILITY_OUTCOMES, weights=EVIDENCE_AVAILABILITY_WEIGHTS
            )[0]
            sampled_cnn = outcome in ("both", "bert_down")
            sampled_bert = outcome in ("both", "cnn_down")
            cnn_available = bool(forced_cnn) if forced_cnn is not None else sampled_cnn
            bert_available = bool(forced_bert) if forced_bert is not None else sampled_bert
        else:
            cnn_available, bert_available = bool(forced_cnn), bool(forced_bert)
        self.cnn_available = cnn_available
        self.bert_available = bert_available

        self.tokens = url_to_obs(self.current_url)
        base = query_structural_evidence(self.current_url)
        self.aux = np.concatenate(
            [base, [float(cnn_available), float(bert_available)]]
        ).astype(np.float32)
        self.mask = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.current_step = 0
        self.query_cost_total = 0.0
        self.actions_taken = []
        return self._obs(), {}

    def _query_structural(self) -> float:
        # Structural evidence (incl. Unicode features) is now always revealed at reset() -- mask[0] starts at 1.0, so this always falls through
        # to the repeat-query branch. Kept only so the action space size (Discrete(6)) doesn't have to change everywhere it's referenced.

        if self.mask[0] == 1.0:
            return CFG.repeat_query_penalty
        full = query_structural_evidence(self.current_url)
        self.aux[2:18] = full[2:]   # only the structural region, aux[18:20] are the v1-5 availability flags, untouched
        self.mask[0] = 1.0
        self.query_cost_total += abs(CFG.query_struct_cost)
        return CFG.query_struct_cost

    def _query_cnn(self) -> float:
        if not self.cnn_available:
            return CFG.unavailable_query_penalty
        if self.mask[1] == 1.0:
            return CFG.repeat_query_penalty
        score = query_cnn_evidence(self.current_url)
        confidence = entropy_confidence(score)
        self.aux[1] = score
        self.aux[15] = confidence
        self.mask[1] = 1.0
        self.query_cost_total += abs(CFG.query_cnn_cost)
        if self.mask[2] == 1.0:
            self.aux[17] = abs(self.aux[1] - self.aux[0])
        return CFG.query_cnn_cost + confidence * CFG.info_gain_weight

    def _query_bert(self) -> float:
        if not self.bert_available:
            return CFG.unavailable_query_penalty
        if self.mask[2] == 1.0:
            return CFG.repeat_query_penalty
        score = query_bert_evidence(self.current_url)
        confidence = entropy_confidence(score)
        self.aux[0] = score
        self.aux[16] = confidence
        self.mask[2] = 1.0
        self.query_cost_total += abs(CFG.query_bert_cost)
        if self.mask[1] == 1.0:
            self.aux[17] = abs(self.aux[1] - self.aux[0])
        return CFG.query_bert_cost + confidence * CFG.info_gain_weight

    def _disagreement_bonus(self) -> float:
        if self.mask[1] == 1.0 and self.mask[2] == 1.0 and self.aux[17] >= CFG.static_disagreement_aux17_threshold:
            return CFG.static_disagreement_resolution_bonus
        return 0.0

    def _final_reward(self, action: int) -> float:
        gt = int(self.true_label)

        if CFG.reward_mode == "static_match":
            self._total_decisions += 1

            if action == self.DEFER:
                return CFG.static_r_abstain
            if action == self.CLASSIFY_PHISHING and gt == 1:
                return CFG.static_r_tp + self._disagreement_bonus()
            if action == self.CLASSIFY_BENIGN and gt == 0:
                return CFG.static_r_tn + self._disagreement_bonus()
            if action == self.CLASSIFY_PHISHING and gt == 0:
                return CFG.static_r_fp

            # False negative. Use the previous FN count before incrementing.
            fn_rate = self._fn_count / max(1, self._total_decisions)
            reward = CFG.static_r_fn_base * (
                1.0 + CFG.static_fn_scale * fn_rate
            )
            self._fn_count += 1
            return reward

        is_gan = self.source in ("gan", "replay")
        conf = query_bert_evidence(self.current_url) if is_gan else float(gt)

        if is_gan:
            if action == self.DEFER:
                return (
                    CFG.defer_phish_gan * conf
                    if gt == 1
                    else CFG.defer_benign_gan * (1.0 - conf)
                )
            if action == self.CLASSIFY_PHISHING and gt == 1:
                return CFG.reward_tp * conf
            if action == self.CLASSIFY_BENIGN and gt == 0:
                return CFG.reward_tn * (1.0 - conf)
            if action == self.CLASSIFY_PHISHING and gt == 0:
                return CFG.reward_fp * (1.0 - conf)
            return CFG.reward_fn * conf

        if action == self.DEFER:
            return (
                CFG.defer_phish_dataset
                if gt == 1 else CFG.defer_benign_dataset
            )
        if action == self.CLASSIFY_PHISHING and gt == 1:
            return CFG.reward_tp
        if action == self.CLASSIFY_BENIGN and gt == 0:
            return CFG.reward_tn
        if action == self.CLASSIFY_PHISHING and gt == 0:
            mult = (
                CFG.fp_guard_multiplier
                if self._recent_benign_acc < CFG.fp_guard_threshold
                else 1.0
            )
            return CFG.reward_fp * mult
        return CFG.reward_fn

    def _timeout_reward(self) -> float:
        base = CFG.static_r_abstain if CFG.reward_mode == "static_match" else (
            CFG.defer_phish_dataset
            if self.true_label == 1
            else CFG.defer_benign_dataset
        )
        return base + CFG.timeout_defer_penalty_extra

    def step(self, action: int):
        action = int(action)
        self.actions_taken.append(action)
        self.current_step += 1

        terminated = False
        truncated = False
        final_action = None

        if action == self.QUERY_STRUCT:
            reward = self._query_structural()
        elif action == self.QUERY_CNN:
            reward = self._query_cnn()
        elif action == self.QUERY_BERT:
            reward = self._query_bert()
        elif action in (
            self.CLASSIFY_BENIGN,
            self.CLASSIFY_PHISHING,
            self.DEFER,
        ):
            reward = self._final_reward(action)
            terminated = True
            final_action = action
        else:
            reward = -0.25

        if not terminated and self.current_step >= CFG.max_steps:
            reward += self._timeout_reward()
            terminated = True
            truncated = True
            final_action = self.DEFER

        info = {
            "url": self.current_url,
            "label": self.true_label,
            "source": self.source,
            "final_action": final_action,
            "query_cost": self.query_cost_total,
            "queried_struct": int(self.mask[0]),
            "queried_cnn": int(self.mask[1]),
            "queried_bert": int(self.mask[2]),
            "query_count": int(self.mask.sum()),
            "cnn_available": int(self.cnn_available),
            "bert_available": int(self.bert_available),
            "actions_taken": list(self.actions_taken),
        }

        return self._obs(), float(reward), terminated, truncated, info


class MultistepHybridExtractor(BaseFeaturesExtractor):
    """Exact feature extractor used by TrainMultistepHybridDRL.py."""
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        emb_dim: int = 128,
        aux_hidden_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        token_space = observation_space.spaces["tokens"]
        vocab_size = int(token_space.high[0]) + 1

        aux_dim = int(observation_space.spaces["aux"].shape[0])
        mask_dim = int(observation_space.spaces["mask"].shape[0])
        step_dim = int(observation_space.spaces["step"].shape[0])
        side_dim = aux_dim + mask_dim + step_dim

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.token_cnn = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.side_mlp = nn.Sequential(
            nn.Linear(side_dim, aux_hidden_dim),
            nn.ReLU(),
            nn.Linear(aux_hidden_dim, aux_hidden_dim),
            nn.ReLU(),
        )
        self.final = nn.Sequential(
            nn.Linear(64 + aux_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        token_features = self.token_cnn(
            self.emb(obs["tokens"].long()).permute(0, 2, 1)
        )
        side_features = self.side_mlp(torch.cat([
            obs["aux"].float(),
            obs["mask"].float(),
            obs["step"].float(),
        ], dim=1))
        return self.final(torch.cat([token_features, side_features], dim=1))


class AdversarialCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = deque(maxlen=500)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
        return True

# Multistep DRL detector


print("Loading Multistep Hybrid DRL detector...")
_env = PhishingEnv()
env = DummyVecEnv([lambda: Monitor(_env)])

policy_kwargs = dict(
    features_extractor_class=MultistepHybridExtractor,
    features_extractor_kwargs=dict(
        features_dim=128,
        emb_dim=128,
        aux_hidden_dim=64,
    ),
    net_arch=[],
)

_multistep_zip = CFG.pretrained_multistep_drl
if _multistep_zip and not os.path.exists(_multistep_zip) and not _multistep_zip.endswith(".zip"):
    _multistep_zip += ".zip"

if _multistep_zip and os.path.exists(_multistep_zip):
    drl_model = PPO.load(_multistep_zip, env=env, device=device)

    drl_model.learning_rate = CFG.lr_drl
    drl_model.lr_schedule = lambda _progress_remaining: CFG.lr_drl
    for param_group in drl_model.policy.optimizer.param_groups:
        param_group["lr"] = CFG.lr_drl

    drl_model.n_epochs = CFG.drl_n_epochs
    drl_model.ent_coef = CFG.drl_ent_coef
    drl_model.clip_range = lambda _progress_remaining: CFG.drl_clip_range
    drl_model.target_kl = CFG.drl_target_kl

    print(f"  Loaded pretrained Multistep DRL: {_multistep_zip}")
    print(
        "  Adversarial fine-tune settings: "
        f"lr={CFG.lr_drl:g}, epochs={CFG.drl_n_epochs}, "
        f"clip={CFG.drl_clip_range}, target_kl={CFG.drl_target_kl}, "
        f"ent_coef={CFG.drl_ent_coef}"
    )
else:
    raise FileNotFoundError(
        "Pretrained Multistep DRL checkpoint was not found.\n"
        f"Expected: {_multistep_zip}\n"
    )

print("  Multistep DRL ready\n")

frozen_drl_sd = None


def update_frozen_policy() -> None:

    global frozen_drl_sd
    frozen_drl_sd = {
        key: value.detach().cpu().clone()
        for key, value in drl_model.policy.state_dict().items()
    }


def _policy_probs_from_obs(obs: dict[str, np.ndarray]) -> torch.Tensor:

    obs_t, _ = drl_model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        return drl_model.policy.get_distribution(
            obs_t
        ).distribution.probs[0]


def _initial_policy_obs(url: str) -> dict[str, np.ndarray]:

    base = query_structural_evidence(url)
    aux = np.concatenate([base, [1.0, 1.0]]).astype(np.float32)
    return {
        "tokens": url_to_obs(url).astype(np.int32),
        "aux": aux,
        "mask": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "step": np.zeros((1,), dtype=np.float32),
    }


def _run_multistep_urls(
    urls: list,
    use_frozen: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:

    urls = list(urls)
    if not urls:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            [],
        )

    restore_sd = None
    was_training = drl_model.policy.training

    if use_frozen:
        if frozen_drl_sd is None:
            raise RuntimeError(
                "Frozen DRL snapshot is not available. Call update_frozen_policy() first."
            )
        restore_sd = {
            key: value.detach().clone()
            for key, value in drl_model.policy.state_dict().items()
        }
        drl_model.policy.load_state_dict(
            {
                key: value.to(device)
                for key, value in frozen_drl_sd.items()
            }
        )

    policy = drl_model.policy
    policy.eval()

    final_actions = []
    detection_scores = []
    traces = []

    try:
        with torch.no_grad():
            for url in urls:
                obs = _initial_policy_obs(url)
                current_step = 0
                actions_taken = []
                query_cost = 0.0
                timed_out = False

                mapped_action = 2
                detection_score = 0.6

                for _ in range(CFG.max_steps):
                    obs["step"][0] = current_step / max(1, CFG.max_steps)

                    action_raw, _ = drl_model.predict(
                        obs,
                        deterministic=True,
                    )
                    action = int(np.asarray(action_raw).item())

                    probs = _policy_probs_from_obs(obs)

                    actions_taken.append(action)
                    current_step += 1

                    if action == PhishingEnv.QUERY_STRUCT:
                        if obs["mask"][0] == 0.0:
                            full = query_structural_evidence(url)
                            obs["aux"][2:18] = full[2:]   # aux[18:20] are the availability flags, untouched
                            obs["mask"][0] = 1.0
                            query_cost += abs(CFG.query_struct_cost)
                    elif action == PhishingEnv.QUERY_CNN:
                        if obs["mask"][1] == 0.0:
                            cnn_score = query_cnn_evidence(url)
                            obs["aux"][1] = cnn_score
                            obs["aux"][15] = entropy_confidence(cnn_score)
                            obs["mask"][1] = 1.0
                            query_cost += abs(CFG.query_cnn_cost)
                            if obs["mask"][2] == 1.0:
                                obs["aux"][17] = abs(obs["aux"][1] - obs["aux"][0])
                    elif action == PhishingEnv.QUERY_BERT:
                        if obs["mask"][2] == 0.0:
                            bert_score = query_bert_evidence(url)
                            obs["aux"][0] = bert_score
                            obs["aux"][16] = entropy_confidence(bert_score)
                            obs["mask"][2] = 1.0
                            query_cost += abs(CFG.query_bert_cost)
                            if obs["mask"][1] == 1.0:
                                obs["aux"][17] = abs(obs["aux"][1] - obs["aux"][0])
                    else:
                        terminal_mass = probs[3] + probs[4] + probs[5]
                        detection_score = float(
                            ((probs[4] + 0.6 * probs[5]) / (terminal_mass + 1e-8)).item()
                        )
                        mapped_action = {
                            PhishingEnv.CLASSIFY_BENIGN: 0,
                            PhishingEnv.CLASSIFY_PHISHING: 1,
                            PhishingEnv.DEFER: 2,
                        }[action]
                        break

                    if current_step >= CFG.max_steps:
                        timed_out = True
                        mapped_action = 2
                        detection_score = 0.6
                        break

                final_actions.append(mapped_action)
                detection_scores.append(detection_score)
                traces.append({
                    "actions_taken": actions_taken,
                    "query_count": int(obs["mask"].sum()),
                    "query_cost": float(query_cost),
                    "queried_struct": int(obs["mask"][0]),
                    "queried_cnn": int(obs["mask"][1]),
                    "queried_bert": int(obs["mask"][2]),
                    "episode_steps": len(actions_taken),
                    "immediate_decision": int(
                        len(actions_taken) == 1 and actions_taken[0] >= 3
                    ),
                    "timed_out": int(timed_out),
                })

    finally:
        if restore_sd is not None:
            drl_model.policy.load_state_dict(restore_sd)
        drl_model.policy.train(was_training)

    return (
        np.asarray(final_actions, dtype=np.int64),
        np.asarray(detection_scores, dtype=np.float32),
        traces,
    )


def _summarize_query_traces(traces: list[dict]) -> dict:
    if not traces:
        return {
            "avg_queries": 0.0,
            "avg_query_cost": 0.0,
            "struct_query_rate": 0.0,
            "cnn_query_rate": 0.0,
            "bert_query_rate": 0.0,
            "immediate_decision_rate": 0.0,
            "avg_episode_steps": 0.0,
            "timeout_rate": 0.0,
        }
    return {
        "avg_queries": float(np.mean([t["query_count"] for t in traces])),
        "avg_query_cost": float(np.mean([t["query_cost"] for t in traces])),
        "struct_query_rate": float(np.mean([t["queried_struct"] for t in traces])),
        "cnn_query_rate": float(np.mean([t["queried_cnn"] for t in traces])),
        "bert_query_rate": float(np.mean([t["queried_bert"] for t in traces])),
        "immediate_decision_rate": float(np.mean([t["immediate_decision"] for t in traces])),
        "avg_episode_steps": float(np.mean([t["episode_steps"] for t in traces])),
        "timeout_rate": float(np.mean([t["timed_out"] for t in traces])),
    }


def drl_detection_score(urls: list) -> torch.Tensor:
    if not urls:
        return torch.empty(0, device=device)
    if frozen_drl_sd is None:
        return torch.full((len(urls),), 0.5, device=device)
    _, scores, _ = _run_multistep_urls(urls, use_frozen=True)
    return torch.tensor(scores, dtype=torch.float32, device=device)


def predict_urls_with_defer(urls: list) -> np.ndarray:
    actions, _, _ = _run_multistep_urls(urls, use_frozen=False)
    return actions


def predict_with_defer(urls: list) -> np.ndarray:
    return predict_urls_with_defer(urls)

# Generator reward
def compute_generator_rewards(urls: list, iteration: int) -> torch.Tensor:
    rewards = torch.full((len(urls),), CFG.invalid_url_reward, dtype=torch.float32, device=device)
    valid_idx = [i for i, u in enumerate(urls) if is_valid_url(u)]
    if not valid_idx:
        return rewards

    valid_urls = [urls[i] for i in valid_idx]
    bert_p = bert_score(valid_urls)
    drl_p = drl_detection_score(valid_urls)             # frozen-detector detection prob
    domain_counts = Counter(get_registered_domain(u) for u in valid_urls)

    for j, i in enumerate(valid_idx):
        url = valid_urls[j]
        b = float(bert_p[j].item())
        if is_benign_domain(url):
            rewards[i] = CFG.benign_domain_reward
            continue


        sim = max_train_phish_similarity(url)
        nov = max(0.0, 1.0 - sim)
        if CFG.use_novelty_veto and sim >= CFG.novelty_veto_threshold:
            rewards[i] = CFG.novelty_penalty_reward
            continue

        if b < CFG.bert_gate:                            # doesn't look like phishing = not useful
            rewards[i] = b - CFG.bert_gate               # small negative, proportional
            continue


        evasion      = 1.0 - float(drl_p[j].item())      # (0=detected, 1=evaded)
        disc_realism = b                                  # P(class=1) — already gated ≥ bert_gate
        plausibility = phishing_plausibility_score(url)

        base = (0.60 * evasion
                + 0.25 * disc_realism
                + 0.10 * nov
                + 0.05 * plausibility)

        # Diversity multiplier: penalise repeated registrable domains
        rewards[i] = base * (1.0 / max(1, domain_counts[get_registered_domain(url)]))
    return rewards


def _clone_model_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _clone_optimizer_state(optimizer) -> Optional[dict]:
    if optimizer is None:
        return None
    try:
        return copy.deepcopy(optimizer.state_dict())
    except Exception:
        return None


def _restore_optimizer_state(optimizer, state: Optional[dict]) -> None:
    if optimizer is not None and state is not None:
        optimizer.load_state_dict(state)


def train_generator(n_batches: int, iteration: int, temperature: float):
    raw_rewards_log, accepted_rewards_log, kl_log = [], [], []
    skipped, accepted = 0, 0

    ppo_optimizer = getattr(ppo_trainer, "optimizer", None)
    saved_iter_model = _clone_model_state(gen_model.pretrained_model)
    saved_iter_optimizer = _clone_optimizer_state(ppo_optimizer)

    for _ in range(n_batches):
        try:
            queries = [prompt_ids[0].clone() for _ in range(CFG.gan_batch_size)]
            gen_model.eval()
            with torch.no_grad():
                responses = ppo_trainer.generate(
                    queries, max_new_tokens=CFG.gen_max_new_tokens, do_sample=True,
                    temperature=temperature, top_k=100, top_p=0.95,
                    pad_token_id=gen_tokenizer.eos_token_id, repetition_penalty=1.2,
                    return_prompt=True,
                )
            gen_model.train()

            base_urls = [
                clean_generated_text(u)
                for u in gen_tokenizer.batch_decode(
                    responses,
                    skip_special_tokens=True,
                )
            ]
            urls, _homoglyph_rows = augment_urls_with_homoglyphs(
                base_urls,
                probability=CFG.homoglyph_probability,
                rng=random,
                context="generator_ppo_reward",
                iteration=iteration,
            )
            with torch.no_grad():
                raw = compute_generator_rewards(urls, iteration)
            raw_rewards_log.append(float(raw.mean().item()))

            # Skip truly degenerate batches with no reward variation.
            raw_std = float(raw.std().item())
            if raw_std < 0.001:
                skipped += 1
                continue

            # Normalise (z-score) when there is enough spread, then clip for PPO stability.
            reward_t = raw.clone()
            if raw_std > 0.05:
                reward_t = (reward_t - reward_t.mean()) / (reward_t.std() + 1e-8)
            reward_t = torch.nan_to_num(reward_t, nan=CFG.invalid_url_reward,
                                        posinf=CFG.reward_clip, neginf=-CFG.reward_clip)
            reward_t = torch.clamp(reward_t, -CFG.reward_clip, CFG.reward_clip)

            reward_list = [reward_t[k].detach().float() for k in range(len(urls))]
            queries_for_step = [responses[k][:prompt_len] for k in range(len(responses))]
            responses_for_step = [responses[k][prompt_len:] for k in range(len(responses))]

            # Batch-level rollback snapshot. This is important: if a KL skip happens, the skipped PPO step must not remain in the generator weights.
            saved_batch_model = _clone_model_state(gen_model.pretrained_model)
            saved_batch_optimizer = _clone_optimizer_state(ppo_optimizer)

            stats = ppo_trainer.step(queries_for_step, responses_for_step, reward_list)

            kl = 0.0
            for key in ["objective/kl", "ppo/kl", "ppo/mean_kl", "train/kl_div_mean", "kl"]:
                if key in stats:
                    v = stats[key]
                    kl = abs(v.item() if isinstance(v, torch.Tensor) else float(v))
                    break

            if kl > CFG.kl_reset_limit:
                skipped += 1
                print(f"    [HARD RESET] KL={kl:.2f} - restoring iter-start generator weights")
                gen_model.pretrained_model.load_state_dict({k: v.to(device) for k, v in saved_iter_model.items()})
                _restore_optimizer_state(ppo_optimizer, saved_iter_optimizer)
                torch.cuda.empty_cache()
                break

            if kl > CFG.kl_skip_limit:
                skipped += 1
                print(f"    [SKIP-ROLLBACK] KL={kl:.2f} - restoring batch-start generator weights")
                gen_model.pretrained_model.load_state_dict({k: v.to(device) for k, v in saved_batch_model.items()})
                _restore_optimizer_state(ppo_optimizer, saved_batch_optimizer)
                torch.cuda.empty_cache()
                continue

            accepted += 1
            accepted_rewards_log.append(float(raw.mean().item()))
            kl_log.append(kl)
        except Exception as e:
            skipped += 1
            print(f"    [gen batch error] {type(e).__name__}: {e}")
            traceback.print_exc()

    avg_raw_r = float(np.mean(raw_rewards_log)) if raw_rewards_log else float("nan")
    avg_accepted_r = float(np.mean(accepted_rewards_log)) if accepted_rewards_log else float("nan")
    avg_kl = float(np.mean(kl_log)) if kl_log else 0.0
    if accepted == 0:
        print("    [WARN] Generator produced NO accepted PPO updates this iteration!")
    return avg_raw_r, avg_accepted_r, avg_kl, skipped, accepted


# Evaluation
def _subset_query_summary(
    traces: list[dict],
    keep_mask: list[bool],
) -> dict:
    subset = [
        trace
        for trace, keep in zip(traces, keep_mask)
        if keep
    ]
    return _summarize_query_traces(subset)


def evaluate(n: int, seed: int, novel_benchmark=None) -> dict:
    rng = random.Random(seed)

    gan_urls, n_genuine = generate_urls(n, allow_fallback=False)
    genuine_actions = np.array([], dtype=np.int64)
    gan_q = _summarize_query_traces([])
    if n_genuine == 0:
        fool_rate = det_rate = def_rate = 0.0
    else:
        genuine_actions, _, gan_traces = _run_multistep_urls(
            gan_urls[:n_genuine], use_frozen=False
        )
        gan_q = _summarize_query_traces(gan_traces)
        fool_rate = float((genuine_actions == 0).mean())
        det_rate = float((genuine_actions == 1).mean())
        def_rate = float((genuine_actions == 2).mean())

    gan_eval_urls = gan_urls[:n_genuine]
    homoglyph_mask = [is_homoglyph_url(u) for u in gan_eval_urls]
    normal_mask = [not flag for flag in homoglyph_mask]

    homoglyph_count = int(sum(homoglyph_mask))
    normal_gan_count = int(sum(normal_mask))

    if homoglyph_count:
        h_actions = genuine_actions[np.asarray(homoglyph_mask, dtype=bool)]
        homoglyph_fool_rate = float((h_actions == 0).mean())
        homoglyph_det_rate = float((h_actions == 1).mean())
        homoglyph_def_rate = float((h_actions == 2).mean())
        homoglyph_q = _subset_query_summary(gan_traces, homoglyph_mask)
    else:
        homoglyph_fool_rate = homoglyph_det_rate = homoglyph_def_rate = 0.0
        homoglyph_q = _summarize_query_traces([])

    if normal_gan_count:
        normal_actions = genuine_actions[np.asarray(normal_mask, dtype=bool)]
        normal_gan_fool_rate = float((normal_actions == 0).mean())
        normal_gan_det_rate = float((normal_actions == 1).mean())
        normal_gan_q = _subset_query_summary(gan_traces, normal_mask)
    else:
        normal_gan_fool_rate = normal_gan_det_rate = 0.0
        normal_gan_q = _summarize_query_traces([])

    phish_sample = rng.sample(eval_phishing, min(n, len(eval_phishing)))
    phish_acts, _, phish_traces = _run_multistep_urls(phish_sample, use_frozen=False)
    phish_q = _summarize_query_traces(phish_traces)
    real_phish_det = float((phish_acts == 1).mean())
    phish_defer = float((phish_acts == 2).mean())

    benign_sample = rng.sample(eval_benign, min(n, len(eval_benign)))
    benign_acts, _, benign_traces = _run_multistep_urls(benign_sample, use_frozen=False)
    benign_q = _summarize_query_traces(benign_traces)
    real_benign_acc = float((benign_acts == 0).mean())
    benign_defer = float((benign_acts == 2).mean())

    tp = int((phish_acts == 1).sum())
    fp = int((benign_acts == 1).sum())
    fn = int((phish_acts != 1).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    novel_det_rate = None
    if novel_benchmark:
        nov_acts = predict_urls_with_defer(novel_benchmark)
        novel_det_rate = float((nov_acts == 1).mean())

    return dict(
        fool_rate=fool_rate,
        det_rate=det_rate,
        def_rate=def_rate,
        n_genuine=n_genuine,
        real_phish_det=real_phish_det,
        phish_defer=phish_defer,
        real_benign_acc=real_benign_acc,
        benign_defer=benign_defer,
        precision=precision,
        recall=recall,
        f1=f1,
        novel_det_rate=novel_det_rate,
        gan_urls=gan_urls[:n_genuine],
        gan_actions=genuine_actions,
        gan_avg_queries=gan_q["avg_queries"],
        gan_avg_query_cost=gan_q["avg_query_cost"],
        gan_struct_query_rate=gan_q["struct_query_rate"],
        gan_cnn_query_rate=gan_q["cnn_query_rate"],
        gan_bert_query_rate=gan_q["bert_query_rate"],
        homoglyph_count=homoglyph_count,
        normal_gan_count=normal_gan_count,
        homoglyph_fool_rate=homoglyph_fool_rate,
        homoglyph_det_rate=homoglyph_det_rate,
        homoglyph_def_rate=homoglyph_def_rate,
        homoglyph_avg_queries=homoglyph_q["avg_queries"],
        homoglyph_avg_query_cost=homoglyph_q["avg_query_cost"],
        homoglyph_struct_query_rate=homoglyph_q["struct_query_rate"],
        homoglyph_cnn_query_rate=homoglyph_q["cnn_query_rate"],
        homoglyph_bert_query_rate=homoglyph_q["bert_query_rate"],
        normal_gan_fool_rate=normal_gan_fool_rate,
        normal_gan_det_rate=normal_gan_det_rate,
        normal_gan_avg_queries=normal_gan_q["avg_queries"],
        phish_avg_queries=phish_q["avg_queries"],
        phish_avg_query_cost=phish_q["avg_query_cost"],
        phish_struct_query_rate=phish_q["struct_query_rate"],
        phish_cnn_query_rate=phish_q["cnn_query_rate"],
        phish_bert_query_rate=phish_q["bert_query_rate"],
        benign_avg_queries=benign_q["avg_queries"],
        benign_avg_query_cost=benign_q["avg_query_cost"],
        benign_struct_query_rate=benign_q["struct_query_rate"],
        benign_cnn_query_rate=benign_q["cnn_query_rate"],
        benign_bert_query_rate=benign_q["bert_query_rate"],
        phish_immediate_rate=phish_q["immediate_decision_rate"],
        benign_immediate_rate=benign_q["immediate_decision_rate"],
    )

def print_samples(n: int = 8) -> None:
    urls, n_gen = generate_urls(n, allow_fallback=False)
    if not urls:
        print("    [WARN] no valid URLs to sample")
        return
    bp = bert_score(urls)
    acts = predict_urls_with_defer(urls)
    amap = {0: "BENIGN(fooled)", 1: "CAUGHT", 2: "DEFER"}
    for i, (u, b, a) in enumerate(zip(urls, bp, acts)):
        veto = " [VETO]" if is_benign_domain(u) else ""
        hflag = " [HOMOGLYPH]" if is_homoglyph_url(u) else ""
        print(
            f"    [{i+1:02d}] DRL={amap[int(a)]:14s} "
            f"BERT={float(b):.3f}{veto}{hflag}  {u[:80]}"
        )



# Stability, early-stop check

def check_stability(log: list) -> bool:
    if len(log) < CFG.nash_min_iter + CFG.nash_window:
        return False
    recent = log[-CFG.nash_window:]
    if any(r["n_genuine"] < 0.5 * CFG.eval_size for r in recent):
        return False
    fool = [r["fool_rate"] for r in recent]
    disc = [r["disc_loss"] for r in recent]
    grew = [r["gen_reward"] for r in recent if not np.isnan(r["gen_reward"])]
    if not grew:
        return False
    ok = ((max(fool) - min(fool) < CFG.nash_fool_tol) and
          (max(disc) - min(disc) < CFG.nash_disc_tol) and
          (max(grew) - min(grew) < 0.05) and
          (np.mean(fool) > CFG.nash_min_mean_fool))
    if ok:
        print("\n  [STABILITY] competitive equilibrium detected")
    return ok


# Warmup

def warmup_generator(n_steps: int) -> None:
    gen_model.pretrained_model.train()
    opt = Adam(gen_model.pretrained_model.parameters(), lr=CFG.lr_gen)
    urls = random.choices(phishing_urls, k=n_steps * CFG.gan_batch_size)
    for i in range(n_steps):
        batch = urls[i * CFG.gan_batch_size:(i + 1) * CFG.gan_batch_size]
        enc = gen_tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=CFG.max_obs_tokens).to(device)
        loss = gen_model.pretrained_model(**enc, labels=enc["input_ids"]).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_model.pretrained_model.parameters(), 1.0)
        opt.step()
        if i % 50 == 0:
            print(f"  warmup {i}/{n_steps} loss={loss.item():.4f}")



# Main training loop

def main() -> None:
    print("=" * 72)
    print("ADVERSARIAL DRL-GAN  (MULTISTEP evidence-gathering detector)")
    print("=" * 72)
    print(f"Device: {device} | seed: {CFG.seed} | iterations: {CFG.num_iterations} | eval_size: {CFG.eval_size}")
    print(f"Reward mode: {CFG.reward_mode}"
          + (f"  (fn dynamic: r_fn_base={CFG.static_r_fn_base}, fn_scale={CFG.static_fn_scale}, "
             f"reset_each_iter={CFG.reset_fn_rate_each_iter})" if CFG.reward_mode == "static_match"
             else "  (four-value defer, confidence-scaled)"))
    print()

    print("Warming up generator on phishing URLs...")
    warmup_generator(CFG.warmup_steps)
    print("Re-syncing PPO reference model after warmup...")
    ppo_trainer.ref_model.pretrained_model.load_state_dict(gen_model.pretrained_model.state_dict())
    print("  Reference model re-synced\n")
    print_samples(8)

    print("\nRunning baseline evaluation...")
    update_frozen_policy()
    baseline = evaluate(CFG.eval_size, seed=0)
    print(
        f"  Baseline F1={baseline['f1']:.4f} "
        f"phish_det={baseline['real_phish_det']:.1%} "
        f"benign_acc={baseline['real_benign_acc']:.1%}"
    )
    print(
        "  Baseline multistep policy | "
        f"phish Q={baseline['phish_avg_queries']:.3f} "
        f"struct={baseline['phish_struct_query_rate']:.1%} "
        f"CNN={baseline['phish_cnn_query_rate']:.1%} "
        f"BERT={baseline['phish_bert_query_rate']:.1%} | "
        f"benign Q={baseline['benign_avg_queries']:.3f} "
        f"struct={baseline['benign_struct_query_rate']:.1%} "
        f"CNN={baseline['benign_cnn_query_rate']:.1%} "
        f"BERT={baseline['benign_bert_query_rate']:.1%}"
    )

    if (
        baseline["f1"] < 0.90
        or baseline["real_phish_det"] < 0.90
        or baseline["real_benign_acc"] < 0.90
    ):
        raise RuntimeError(
            "MULTISTEP BASELINE PARITY CHECK FAILED. "
            "The pretrained checkpoint does not reproduce its standalone "
            "behaviour inside this loop. Adversarial training was stopped "
            "before any PPO detector update."
        )

    best_f1 = baseline["f1"]
    best_phish_det = baseline["real_phish_det"]
    best_benign_acc = baseline["real_benign_acc"]
    best_policy_sd = {k: v.clone() for k, v in drl_model.policy.state_dict().items()}
    best_disc_loss = float("inf")
    training_log = []
    novel_benchmark = []

    print("\n" + "=" * 72)
    print(f"ADVERSARIAL TRAINING ({CFG.num_iterations} iterations)")
    print("=" * 72)

    for iteration in range(1, CFG.num_iterations + 1):
        print(f"\n{'='*72}\nITERATION {iteration}/{CFG.num_iterations}\n{'='*72}")
        t0 = time.time()
        try:
            temperature = max(1.2, 1.5 - 0.01 * iteration)

            if CFG.reward_mode == "static_match" and CFG.reset_fn_rate_each_iter:
                _env.reset_fn_rate()

            print(f"[1] GAN pool ({CFG.gan_pool_size})...")
            pool_urls, pool_genuine, _ = make_gan_pool(CFG.gan_pool_size, temperature=temperature)
            if not pool_urls:
                print("    [WARN] empty GAN pool — falling back to real phishing for this iter")
                pool_urls = random.sample(phishing_urls, min(CFG.gan_pool_size, len(phishing_urls)))
            _env.set_gan_pool(pool_urls)
            _env.set_replay_pool(get_replay_batch(50))
            print(f"    pool genuine={pool_genuine} | replay buffer={len(replay_buffer)}")

            if iteration == 1:
                novel_benchmark = [u for u in pool_urls if not is_benign_domain(u)][:100]
                print(f"    novel benchmark locked: {len(novel_benchmark)}  "
                      f"(NOTE: saturates fast — replay-recatch is the real adaptation metric)")

            print(f"[2] Fine-tune DRL ({CFG.drl_timesteps} steps)...")
            pre_update_policy_sd = {
                k: v.detach().cpu().clone()
                for k, v in drl_model.policy.state_dict().items()
            }

            cb = AdversarialCallback()
            drl_model.learn(
                total_timesteps=CFG.drl_timesteps,
                callback=cb,
                reset_num_timesteps=False,
            )
            avg_drl_r = float(np.mean(cb.ep_rewards)) if cb.ep_rewards else float("nan")
            update_frozen_policy()

            print(f"[3] Evaluate ({CFG.eval_size})...")
            metrics = evaluate(CFG.eval_size, seed=iteration * 100, novel_benchmark=novel_benchmark)
            update_replay_buffer(metrics["gan_urls"], metrics["gan_actions"])

            if val_benign:
                va = predict_urls_with_defer(val_benign)
                _env.set_benign_acc(float((va == 0).mean()))

            print(f"    fool={metrics['fool_rate']:.1%}  phish_det={metrics['real_phish_det']:.1%}  "
                  f"benign_acc={metrics['real_benign_acc']:.1%}  F1={metrics['f1']:.4f}  "
                  f"(P={metrics['precision']:.3f} R={metrics['recall']:.3f})")
            print(f"    multistep queries | phish Q={metrics['phish_avg_queries']:.2f} "
                  f"BERT={metrics['phish_bert_query_rate']:.1%} | "
                  f"benign Q={metrics['benign_avg_queries']:.2f} "
                  f"BERT={metrics['benign_bert_query_rate']:.1%} | "
                  f"GAN Q={metrics['gan_avg_queries']:.2f} "
                  f"BERT={metrics['gan_bert_query_rate']:.1%}")
            print(
                "    homoglyph GAN | "
                f"n={metrics['homoglyph_count']} "
                f"fool={metrics['homoglyph_fool_rate']:.1%} "
                f"det={metrics['homoglyph_det_rate']:.1%} "
                f"Q={metrics['homoglyph_avg_queries']:.2f} "
                f"CNN={metrics['homoglyph_cnn_query_rate']:.1%} "
                f"BERT={metrics['homoglyph_bert_query_rate']:.1%}"
            )

            query_policy_collapsed = (
                metrics["phish_immediate_rate"] > 0.995
                and metrics["benign_immediate_rate"] > 0.995
                and metrics["phish_avg_queries"] < 0.01
                and metrics["benign_avg_queries"] < 0.01
            )
            if query_policy_collapsed:
                print(
                    "    [QUERY-POLICY COLLAPSE] detector stopped gathering evidence "
                    "and is making immediate terminal decisions."
                )

            # Dynamic ratios by GAN pressure
            fool = metrics["fool_rate"]
            if fool > 0.15:
                _env.set_ratios(0.50, 0.20, 0.20, 0.10)
            elif fool > 0.08:
                _env.set_ratios(0.50, 0.25, 0.15, 0.10)
            else:
                _env.set_ratios(0.50, 0.30, 0.10, 0.10)

            phish_floor = min(CFG.min_gate_phish_det, max(0.0, best_phish_det - CFG.max_gate_metric_drop))
            benign_floor = min(CFG.min_gate_benign_acc, max(0.0, best_benign_acc - CFG.max_gate_metric_drop))
            f1_drop = best_f1 - metrics["f1"]
            gate_failed = (
                metrics["real_phish_det"] < phish_floor
                or metrics["real_benign_acc"] < benign_floor
                or f1_drop > CFG.max_gate_f1_drop
                or query_policy_collapsed
            )

            if gate_failed:
                print(
                    f"    [REGRESSION] restoring best policy | "
                    f"phish_det={metrics['real_phish_det']:.1%} floor={phish_floor:.1%}, "
                    f"benign_acc={metrics['real_benign_acc']:.1%} floor={benign_floor:.1%}, "
                    f"F1_drop={max(0.0, f1_drop):.1%}"
                )

                drl_model.policy.load_state_dict(
                    {k: v.to(device) for k, v in pre_update_policy_sd.items()}
                )
                update_frozen_policy()
                print("    [DRL UPDATE REJECTED] pre-iteration multistep policy restored")
            elif metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_phish_det = metrics["real_phish_det"]
                best_benign_acc = metrics["real_benign_acc"]
                best_policy_sd = {k: v.clone() for k, v in drl_model.policy.state_dict().items()}
                drl_model.save(str(OUTPUT_DIR / "best_multistep_adversarial_drl_v1-6"))
                print(f"    [BEST] new best DRL F1={best_f1:.4f} | saved best_multistep_adversarial_drl_v1-6.zip")

            print("[3.5] Samples:")
            print_samples(8)

            print(f"[4] Train generator ({CFG.gan_batches} batches)...")
            avg_gen_raw_r, avg_gen_accepted_r, avg_kl, n_skip, n_accept = train_generator(
                CFG.gan_batches, iteration, temperature
            )
            print(
                f"    gen_raw_reward={avg_gen_raw_r:.4f}  "
                f"gen_accepted_reward={avg_gen_accepted_r:.4f}  "
                f"kl={avg_kl:.4f}  skipped={n_skip}  accepted={n_accept}"
            )

            smooth = max(0.05, 0.15 - 0.003 * iteration)
            print(f"[5] Train discriminator ({CFG.disc_batches} batches, smooth={smooth:.2f})...")
            disc_loss = train_discriminator(CFG.disc_batches, smooth)
            EVIDENCE_CACHE.clear_bert()
            print(f"    disc_loss={disc_loss:.4f} | evidence_cache={EVIDENCE_CACHE.stats()}")

            # BERT fool rate on the pool (smoothed) — diagnostic only
            pool_sample = [u for u in pool_urls if not is_benign_domain(u)][:20]
            recent_bert_fool = float((bert_score(pool_sample) < 0.5).float().mean().item()) if pool_sample else 0.0

            elapsed = time.time() - t0
            training_log.append(dict(
                iteration=iteration, fool_rate=metrics["fool_rate"], det_rate=metrics["det_rate"],
                def_rate=metrics["def_rate"], n_genuine=metrics["n_genuine"],
                real_phish_det=metrics["real_phish_det"], phish_defer=metrics["phish_defer"],
                real_benign_acc=metrics["real_benign_acc"], benign_defer=metrics["benign_defer"],
                precision=metrics["precision"], recall=metrics["recall"], f1=metrics["f1"],
                novel_det_rate=metrics["novel_det_rate"], gen_reward=avg_gen_raw_r,
                gen_raw_reward=avg_gen_raw_r, gen_accepted_reward=avg_gen_accepted_r,
                gen_kl=avg_kl, gen_skipped=n_skip, gen_ppo_updates=n_accept,
                drl_reward=avg_drl_r, disc_loss=disc_loss,
                bert_fool_rate_pool=recent_bert_fool, replay_buffer_size=len(replay_buffer),
                gan_avg_queries=metrics["gan_avg_queries"],
                gan_avg_query_cost=metrics["gan_avg_query_cost"],
                gan_struct_query_rate=metrics["gan_struct_query_rate"],
                gan_cnn_query_rate=metrics["gan_cnn_query_rate"],
                gan_bert_query_rate=metrics["gan_bert_query_rate"],
                homoglyph_count=metrics["homoglyph_count"],
                normal_gan_count=metrics["normal_gan_count"],
                homoglyph_fool_rate=metrics["homoglyph_fool_rate"],
                homoglyph_det_rate=metrics["homoglyph_det_rate"],
                homoglyph_def_rate=metrics["homoglyph_def_rate"],
                homoglyph_avg_queries=metrics["homoglyph_avg_queries"],
                homoglyph_avg_query_cost=metrics["homoglyph_avg_query_cost"],
                homoglyph_struct_query_rate=metrics["homoglyph_struct_query_rate"],
                homoglyph_cnn_query_rate=metrics["homoglyph_cnn_query_rate"],
                homoglyph_bert_query_rate=metrics["homoglyph_bert_query_rate"],
                normal_gan_fool_rate=metrics["normal_gan_fool_rate"],
                normal_gan_det_rate=metrics["normal_gan_det_rate"],
                normal_gan_avg_queries=metrics["normal_gan_avg_queries"],
                phish_avg_queries=metrics["phish_avg_queries"],
                phish_avg_query_cost=metrics["phish_avg_query_cost"],
                phish_struct_query_rate=metrics["phish_struct_query_rate"],
                phish_cnn_query_rate=metrics["phish_cnn_query_rate"],
                phish_bert_query_rate=metrics["phish_bert_query_rate"],
                benign_avg_queries=metrics["benign_avg_queries"],
                benign_avg_query_cost=metrics["benign_avg_query_cost"],
                benign_struct_query_rate=metrics["benign_struct_query_rate"],
                benign_cnn_query_rate=metrics["benign_cnn_query_rate"],
                benign_bert_query_rate=metrics["benign_bert_query_rate"],
                phish_immediate_rate=metrics["phish_immediate_rate"],
                benign_immediate_rate=metrics["benign_immediate_rate"],
                elapsed_s=elapsed,
            ))
            pd.DataFrame(training_log).to_csv(
                OUTPUT_DIR / "training_log_v1-6.csv",
                index=False,
            )

            if CFG.save_homoglyph_examples and HOMOGLYPH_EXAMPLE_ROWS:
                pd.DataFrame(HOMOGLYPH_EXAMPLE_ROWS).drop_duplicates(
                    subset=[
                        "source_url",
                        "final_url",
                        "context",
                        "iteration",
                    ],
                    keep="last",
                ).to_csv(
                    OUTPUT_DIR / "homoglyph_generated_urls_v1-6.csv",
                    index=False,
                )

            iter_dir = OUTPUT_DIR / f"iter_{iteration:02d}"
            iter_dir.mkdir(exist_ok=True)
            drl_model.save(str(iter_dir / "drl"))
            if disc_loss < best_disc_loss:
                best_disc_loss = disc_loss
                gen_model.save_pretrained(str(iter_dir / "generator"))
                disc_model.save_pretrained(str(iter_dir / "discriminator"))
                disc_tokenizer.save_pretrained(str(iter_dir / "discriminator"))
                print(f"    [BEST] new best disc checkpoint (loss={best_disc_loss:.4f})")

            print(f"Iteration {iteration} done [{elapsed:.1f}s]")

            if check_stability(training_log):
                print(f"\nSTABILITY REACHED at iteration {iteration} — stopping early")
                drl_model.save(str(iter_dir / "drl_stable"))
                gen_model.save_pretrained(str(iter_dir / "generator_stable"))
                break

        except Exception as exc:
            print(f"\nITERATION {iteration} FAILED: {exc}")
            traceback.print_exc()
            break

    print("\n" + "=" * 72 + "\nTRAINING COMPLETE\n" + "=" * 72)
    if training_log:
        df = pd.DataFrame(training_log)
        df.to_csv(OUTPUT_DIR / "training_log_v1-6.csv", index=False)
        print(df[["iteration", "fool_rate", "real_phish_det", "real_benign_acc",
                  "f1", "phish_avg_queries", "benign_avg_queries", "gan_avg_queries",
                  "phish_bert_query_rate", "benign_bert_query_rate", "gan_bert_query_rate",
                  "gen_raw_reward", "gen_accepted_reward", "gen_ppo_updates",
                  "disc_loss"]].to_string(index=False))
        print(f"\nBest F1={best_f1:.4f}  best phish_det={best_phish_det:.1%}")
        print(f"Log -> {OUTPUT_DIR / 'training_log_v1-6.csv'}")
        if CFG.save_homoglyph_examples:
            print(
                "Homoglyph examples -> "
                f"{OUTPUT_DIR / 'homoglyph_generated_urls_v1-6.csv'}"
            )


if __name__ == "__main__":
    main()