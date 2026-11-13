"""Archival: earlier multistep DRL detector (char+word URLNet feature extractor).

Kept for reference only. This script is NOT part of the runnable pipeline in the
repository root: it expects externally trained models (a fine-tuned BERT, a URLNet
checkpoint with a single word_vocab.json) and the earlier 70/30 data split, none of
which are included here. Update the paths in Config to point at your own artifacts
before running.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from transformers import BertForSequenceClassification, BertTokenizer, GPT2Tokenizer


@dataclass
class Config:
    seed: int = 42

    # Edit these paths to point at your own data and model artifacts.
    benign_data: str = "data/legit_train70.csv"
    phishing_data: str = "data/phish_train70.csv"
    bert_model_path: str = "models/finetunedBERT/checkpoint"
    cnn_model_path: str = "models/URLNet/best_model.pt"
    cnn_vocab_path: str = "models/URLNet/word_vocab.json"
    output_dir: str = "output/DRL_Unicode_Fed"
    resume_checkpoint: str = ""
    holdout: int = 500
    eval_size: int = 500
    max_obs_tokens: int = 80
    aux_dim: int = 20
    mask_dim: int = 3

    # Multistep MDP
    max_steps: int = 4
    query_struct_cost: float = -0.01
    query_cnn_cost: float = -0.03
    query_bert_cost: float = -0.03
    repeat_query_penalty: float = -0.10
    unavailable_query_penalty: float = -0.10
    info_gain_weight: float = 0.0
    evidence_dropout_both_available: float = 0.60
    evidence_dropout_cnn_unavailable: float = 0.25
    evidence_dropout_bert_unavailable: float = 0.10
    evidence_dropout_both_unavailable: float = 0.05
    r_tp: float = 1.0
    r_tn: float = 1.0
    r_fp: float = -1.0
    r_fn: float = -2.0
    r_abstain: float = -0.10
    timeout_defer_penalty_extra: float = -0.05

    # Sampling
    benign_ratio: float = 0.50
    phishing_ratio: float = 0.50
    disagreement_oversample_ratio: float = 0.20   # fraction of episodes drawn from the mined disagreement pools instead of normal random sampling
    disagreement_resolution_bonus: float = 0.30   # added on top of the normal TP/TN reward when the decision was correct AND aux[17] shows real CNN/BERT disagreement
    disagreement_aux17_threshold: float = 0.30    # |cnn_score - bert_score| above this counts as "real" disagreement for the bonus, not just both oracles being near 0.5
    expert_failure_oversample_ratio: float = 0.05 # separate, smaller ratio for the both-wrong pool -- rarer and harder, no tie-break lesson applies (aux[17] near 0)

    deferred_buffer_size: int = 2000     # bounded FIFO -- oldest deferred cases age out as training progresses
    deferred_replay_ratio: float = 0.10  # fraction of training episodes drawn from this buffer instead of normal/expert-failure/disagreement sampling

    # PPO
    total_timesteps: int = 200_000
    n_steps: int = 1024
    batch_size: int = 128
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Feature extractor
    token_emb_dim: int = 128
    features_dim: int = 128
    side_hidden_dim: int = 64

    # Eval callback
    eval_every_steps: int = 10_000
    save_best_by: str = "strict_f1"
    benign_accuracy_gate: float = 0.97
    robustness_eval_size: int = 200
    dependency_eval_size: int = 400

    # Cache
    score_cache_size: int = 100_000


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


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(CFG.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TWO_LEVEL_SUFFIXES = {
    "co.uk", "ac.uk", "gov.uk", "org.uk", "com.au", "net.au", "org.au",
    "co.jp", "com.br", "com.cn", "com.np", "co.in", "co.nz", "co.za",
}

PHISH_BRANDS = {
    "paypal", "apple", "google", "microsoft", "amazon", "netflix", "facebook",
    "instagram", "linkedin", "twitter", "dropbox", "wellsfargo", "bankofamerica",
    "chase", "hsbc", "dhl", "fedex", "ups", "usps", "irs", "coinbase", "binance",
    "office", "outlook", "icloud", "verify", "secure", "account", "login",
}

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online", "live",
    "click", "link", "support", "ru", "cn", "pw", "zip", "mov", "icu", "shop",
    "work", "cfd", "quest", "monster", "lol",
}

INVALID_CHARS = set("*'\"\\|{}[]^`~<>")

HOMOGLYPH_MAP = {
    "a": ("а",),
    "c": ("с",),
    "e": ("е",),
    "i": ("і",),
    "j": ("ј",),
    "o": ("о",),
    "p": ("р",),
    "s": ("ѕ",),
    "x": ("х",),
    "y": ("у",),
}

HOMOGLYPH_REVERSE = {
    confusable: ascii_character
    for ascii_character, values in HOMOGLYPH_MAP.items()
    for confusable in values
}


def get_host(url: str) -> str:
    try:
        u = str(url).strip()
        parsed = urlparse(u if "://" in u else "http://" + u)
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


def hostname_scripts(hostname: str) -> set[str]:
    scripts: set[str] = set()

    for character in str(hostname):
        if not character.isalpha():
            continue

        name = unicodedata.name(character, "")
        if "LATIN" in name:
            scripts.add("LATIN")
        elif "CYRILLIC" in name:
            scripts.add("CYRILLIC")
        elif "GREEK" in name:
            scripts.add("GREEK")
        elif name:
            scripts.add("OTHER")

    return scripts


def unicode_structural_features(hostname: str) -> list[float]:
    host = str(hostname or "")
    host_lower = host.casefold()

    contains_non_ascii = (
        1.0 if any(ord(character) > 127 for character in host)
        else 0.0
    )

    mixed_script = 1.0 if len(hostname_scripts(host)) > 1 else 0.0

    contains_punycode = (
        1.0
        if any(label.startswith("xn--") for label in host_lower.split("."))
        else 0.0
    )

    confusable_count = sum(
        1
        for character in host
        if character in HOMOGLYPH_REVERSE
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


def structural_features(url: str) -> np.ndarray:

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
        for b in PHISH_BRANDS
    )
    brand_is_registered = any(
        b in registered_base
        for b in PHISH_BRANDS
    )
    brand_misuse = 1.0 if (
        brand_in_sub_or_path and not brand_is_registered
    ) else 0.0

    suspicious_tld = 1.0 if tld in SUSPICIOUS_TLDS else 0.0
    subdomain_depth_norm = min(len(subdomains) / 4.0, 1.0)
    hyphen_in_domain = 1.0 if "-" in registered_base else 0.0
    has_at = 1.0 if "@" in low else 0.0
    long_path = 1.0 if len(path) > 40 else 0.0
    digit_in_domain = 1.0 if any(c.isdigit() for c in registered_base) else 0.0
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
    structural_score = float(np.clip(structural_score, 0.0, 1.0))

    unicode_features = unicode_structural_features(host)
    struct_confidence = entropy_confidence(structural_score)

    return np.asarray(
        [
            0.0,
            0.0,
            structural_score,
            length_norm,
            ip_host,
            suspicious_tld,
            subdomain_depth_norm,
            has_at,
            digit_in_domain,
            long_path,
            *unicode_features,
            struct_confidence,  # aux[14] -- always visible, paired with structural_score
            0.0,                # aux[15] -- cnn_confidence, hidden until query_cnn
            0.0,                # aux[16] -- bert_confidence, hidden until query_bert
            0.0,                # aux[17] -- cnn/bert disagreement, 0.0 until both queried
        ],
        dtype=np.float32,
    )

# DATA

def read_url_csv(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p)
    col = next((c for c in ["url", "URL", "urls", "link", "uri", "domain"] if c in df.columns), df.columns[0])
    urls = df[col].dropna().astype(str).str.strip().tolist()
    return [u for u in urls if u]


print("=" * 80)
print("MULTISTEP HYBRID DRL TRAINING (v1.5 -- oracle-disagreement oversampling + resolution bonus)")
print("=" * 80)
print(f"Device: {device}")
print(f"Auxiliary dimension: {CFG.aux_dim} "
      f"(v1.2's 18 + 2 evidence-availability flags)")
print(f"Info-gain query reward weight: {CFG.info_gain_weight}")
print(f"Query costs: struct={CFG.query_struct_cost} cnn={CFG.query_cnn_cost} bert={CFG.query_bert_cost} (equalized)")
print(f"Evidence dropout: both={CFG.evidence_dropout_both_available:.0%} "
      f"cnn_down={CFG.evidence_dropout_cnn_unavailable:.0%} "
      f"bert_down={CFG.evidence_dropout_bert_unavailable:.0%} "
      f"both_down={CFG.evidence_dropout_both_unavailable:.0%}")
print(f"Checkpoint gate: benign_accuracy >= {CFG.benign_accuracy_gate:.0%}, then best {CFG.save_best_by}")
print(f"Disagreement oversample ratio: {CFG.disagreement_oversample_ratio:.0%}  "
      f"resolution bonus: {CFG.disagreement_resolution_bonus}  "
      f"aux[17] threshold: {CFG.disagreement_aux17_threshold}")
print("Training a new v1.5 checkpoint from scratch.")
print("Loading data...")

benign_all = read_url_csv(CFG.benign_data)
phishing_all = read_url_csv(CFG.phishing_data)

if len(benign_all) <= CFG.holdout or len(phishing_all) <= CFG.holdout:
    raise ValueError("Not enough rows for holdout split.")

eval_benign = benign_all[-CFG.holdout:]
eval_phishing = phishing_all[-CFG.holdout:]
train_benign = benign_all[:-CFG.holdout]
train_phishing = phishing_all[:-CFG.holdout]

assert not (set(train_benign) & set(eval_benign)), "Benign leakage: train/eval overlap."
assert not (set(train_phishing) & set(eval_phishing)), "Phishing leakage: train/eval overlap."

print(f"  Train benign   : {len(train_benign)}")
print(f"  Train phishing : {len(train_phishing)}")
print(f"  Eval benign    : {len(eval_benign)}")
print(f"  Eval phishing  : {len(eval_phishing)}\n")



# TOKENIZER

print("Loading GPT-2 tokenizer...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
VOCAB_SIZE = gpt2_tokenizer.vocab_size


def url_to_tokens(url: str) -> np.ndarray:
    toks = gpt2_tokenizer.encode(
        str(url),
        max_length=CFG.max_obs_tokens,
        padding="max_length",
        truncation=True,
    )
    return np.asarray(toks, dtype=np.int32)


# BERT SCORER  (2-class, detection-only)


print("Loading BERT scorer...")
bert_tokenizer = BertTokenizer.from_pretrained(CFG.bert_model_path)
bert_model = BertForSequenceClassification.from_pretrained(CFG.bert_model_path, num_labels=2).to(device)
bert_model.eval()
print("  BERT ready")


@torch.no_grad()
def bert_score_batch(urls: List[str], batch_size: int = 64) -> np.ndarray:
    if not urls:
        return np.empty((0,), dtype=np.float32)
    out = []
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        enc = bert_tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        logits = bert_model(**enc).logits
        probs = torch.softmax(logits, dim=1)[:, 1]
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


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
word2idx: Dict[str, int] = {}


def tokenise_chars(url: str) -> List[int]:
    indices = [CHAR2IDX.get(ch, UNK_IDX) for ch in str(url)[:MAX_CHAR_LEN]]
    indices += [PAD_IDX] * (MAX_CHAR_LEN - len(indices))
    return indices


def tokenise_words(url: str) -> List[int]:
    tokens = [t for t in re.split(WORD_DELIMITERS, str(url)) if t][:MAX_WORD_LEN]
    indices = [word2idx.get(t, UNK_IDX) for t in tokens]
    indices += [PAD_IDX] * (MAX_WORD_LEN - len(indices))
    return indices


class MultiKernelCNN(nn.Module):
    def __init__(self, emb_dim, n_filters, kernel_sizes, dropout):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(emb_dim, n_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for k in kernel_sizes
        ])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return torch.cat([F.adaptive_max_pool1d(b(x), 1).squeeze(-1) for b in self.branches], dim=1)


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


print("Loading URLNet/CNN scorer...")
with open(CFG.cnn_vocab_path, "r", encoding="utf-8") as f:
    word2idx = json.load(f)
cnn_model = URLNet().to(device)
cnn_model.load_state_dict(torch.load(CFG.cnn_model_path, map_location=device))
cnn_model.eval()
print("  URLNet/CNN ready\n")


@torch.no_grad()
def cnn_score_batch(urls: List[str], batch_size: int = 128) -> np.ndarray:
    if not urls:
        return np.empty((0,), dtype=np.float32)
    out = []
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        x_char = torch.tensor([tokenise_chars(u) for u in batch], dtype=torch.long, device=device)
        x_word = torch.tensor([tokenise_words(u) for u in batch], dtype=torch.long, device=device)
        probs = torch.softmax(cnn_model(x_char, x_word), dim=1)[:, 1]
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


class ScoreCache:
    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self.bert_cache: OrderedDict[str, float] = OrderedDict()
        self.cnn_cache: OrderedDict[str, float] = OrderedDict()
        self.struct_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def _put(self, cache: OrderedDict, key: str, value):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self.max_size:
            cache.popitem(last=False)

    def bert(self, url: str) -> float:
        if url not in self.bert_cache:
            self._put(self.bert_cache, url, float(bert_score_batch([url])[0]))
        return float(self.bert_cache[url])

    def cnn(self, url: str) -> float:
        if url not in self.cnn_cache:
            self._put(self.cnn_cache, url, float(cnn_score_batch([url])[0]))
        return float(self.cnn_cache[url])

    def struct(self, url: str) -> np.ndarray:
        if url not in self.struct_cache:
            self._put(self.struct_cache, url, structural_features(url))
        return self.struct_cache[url].copy()

    def stats(self) -> dict:
        return {
            "bert_cached": len(self.bert_cache),
            "cnn_cached": len(self.cnn_cache),
            "struct_cached": len(self.struct_cache),
        }


score_cache = ScoreCache(CFG.score_cache_size)



def mine_disagreement_pool(urls: List[str], labels: List[int], threshold: float = 0.5) -> tuple[list, list, list]:

    if not urls:
        return [], [], []
    cnn_scores = cnn_score_batch(urls)
    bert_scores = bert_score_batch(urls)
    cnn_pred = (cnn_scores >= threshold).astype(int)
    bert_pred = (bert_scores >= threshold).astype(int)
    labels_arr = np.asarray(labels, dtype=int)
    disagree = cnn_pred != bert_pred
    cnn_right = (cnn_pred == labels_arr)
    bert_right = (bert_pred == labels_arr)
    cnn_right_bert_wrong = [
        (u, int(l)) for u, l, d, cr, br in zip(urls, labels_arr, disagree, cnn_right, bert_right)
        if d and cr and not br
    ]
    bert_right_cnn_wrong = [
        (u, int(l)) for u, l, d, cr, br in zip(urls, labels_arr, disagree, cnn_right, bert_right)
        if d and br and not cr
    ]
    both_wrong = [
        (u, int(l)) for u, l, d, cr, br in zip(urls, labels_arr, disagree, cnn_right, bert_right)
        if not d and not cr and not br
    ]
    return cnn_right_bert_wrong, bert_right_cnn_wrong, both_wrong


print("Mining CNN/BERT failure cases (ExpertFailureBuffer -- one-time pass, used for oversampling + eval)...")
_train_urls = train_benign + train_phishing
_train_labels = [0] * len(train_benign) + [1] * len(train_phishing)
train_cnn_right_bert_wrong, train_bert_right_cnn_wrong, train_both_wrong = mine_disagreement_pool(_train_urls, _train_labels)
print(f"  Train pools: cnn_right_bert_wrong={len(train_cnn_right_bert_wrong)}  "
      f"bert_right_cnn_wrong={len(train_bert_right_cnn_wrong)}  "
      f"both_wrong={len(train_both_wrong)}  (of {len(_train_urls)} training URLs)")

_eval_urls = eval_benign + eval_phishing
_eval_labels = [0] * len(eval_benign) + [1] * len(eval_phishing)
eval_cnn_right_bert_wrong, eval_bert_right_cnn_wrong, eval_both_wrong = mine_disagreement_pool(_eval_urls, _eval_labels)
print(f"  Eval pools:  cnn_right_bert_wrong={len(eval_cnn_right_bert_wrong)}  "
      f"bert_right_cnn_wrong={len(eval_bert_right_cnn_wrong)}  "
      f"both_wrong={len(eval_both_wrong)}  (of {len(_eval_urls)} eval URLs)\n")


deferred_replay_buffer: deque = deque(maxlen=CFG.deferred_buffer_size)
deferred_defer_counts: dict = {}



# MULTISTEP ENVIRONMENT
class MultistepPhishingEnv(gym.Env):
    metadata = {"render_modes": []}

    QUERY_STRUCT = 0
    QUERY_CNN = 1
    QUERY_BERT = 2
    CLASSIFY_BENIGN = 3
    CLASSIFY_PHISHING = 4
    DEFER = 5

    def __init__(self, benign_urls: List[str], phishing_urls: List[str], cache: ScoreCache):
        super().__init__()
        self.benign_urls = benign_urls
        self.phishing_urls = phishing_urls
        self.cache = cache

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            "tokens": spaces.Box(low=0, high=VOCAB_SIZE - 1, shape=(CFG.max_obs_tokens,), dtype=np.int32),
            "aux": spaces.Box(low=0.0, high=1.0, shape=(CFG.aux_dim,), dtype=np.float32),
            "mask": spaces.Box(low=0.0, high=1.0, shape=(CFG.mask_dim,), dtype=np.float32),
            "step": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        self.current_url = ""
        self.true_label = 0
        self.tokens = np.zeros((CFG.max_obs_tokens,), dtype=np.int32)
        self.aux = np.zeros((CFG.aux_dim,), dtype=np.float32)
        self.mask = np.zeros((CFG.mask_dim,), dtype=np.float32)
        self.current_step = 0
        self.query_cost_total = 0.0
        self.actions_taken: List[int] = []
        self.cnn_available = True
        self.bert_available = True

    def _sample_url(self) -> tuple[str, int]:
        if random.random() < CFG.expert_failure_oversample_ratio and train_both_wrong:
            return random.choice(train_both_wrong)
        if random.random() < CFG.deferred_replay_ratio and deferred_replay_buffer:
            return random.choice(deferred_replay_buffer)
        if random.random() < CFG.disagreement_oversample_ratio and (train_cnn_right_bert_wrong or train_bert_right_cnn_wrong):
            if train_cnn_right_bert_wrong and train_bert_right_cnn_wrong:
                pool = train_cnn_right_bert_wrong if random.random() < 0.5 else train_bert_right_cnn_wrong
            else:
                pool = train_cnn_right_bert_wrong or train_bert_right_cnn_wrong
            return random.choice(pool)
        if random.random() < CFG.benign_ratio:
            return random.choice(self.benign_urls), 0
        return random.choice(self.phishing_urls), 1

    def _sample_availability(self) -> tuple[bool, bool]:
        outcome = random.choices(EVIDENCE_AVAILABILITY_OUTCOMES, weights=EVIDENCE_AVAILABILITY_WEIGHTS)[0]
        cnn_available = outcome in ("both", "bert_down")
        bert_available = outcome in ("both", "cnn_down")
        return cnn_available, bert_available

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "url" in options:
            self.current_url = str(options["url"])
            self.true_label = int(options["label"])
        else:
            self.current_url, self.true_label = self._sample_url()

        forced_cnn = options.get("force_cnn_available") if options else None
        forced_bert = options.get("force_bert_available") if options else None
        if forced_cnn is None or forced_bert is None:
            sampled_cnn, sampled_bert = self._sample_availability()
            cnn_available = bool(forced_cnn) if forced_cnn is not None else sampled_cnn
            bert_available = bool(forced_bert) if forced_bert is not None else sampled_bert
        else:
            cnn_available, bert_available = bool(forced_cnn), bool(forced_bert)
        self.cnn_available = cnn_available
        self.bert_available = bert_available

        self.tokens = url_to_tokens(self.current_url)
        base = self.cache.struct(self.current_url)
        self.aux = np.concatenate(
            [base, [float(cnn_available), float(bert_available)]]
        ).astype(np.float32)
        self.mask = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.current_step = 0
        self.query_cost_total = 0.0
        self.actions_taken = []
        return self._obs(), {}

    def _obs(self) -> dict:
        return {
            "tokens": self.tokens.astype(np.int32),
            "aux": np.nan_to_num(self.aux, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0).astype(np.float32),
            "mask": self.mask.astype(np.float32),
            "step": np.asarray([self.current_step / max(1, CFG.max_steps)], dtype=np.float32),
        }

    def _query_struct(self) -> float:
        if self.mask[0] == 1.0:
            return CFG.repeat_query_penalty
        full = self.cache.struct(self.current_url)
        self.aux[2:18] = full[2:]   # only the structural region -- aux[18:20] are the availability flags, untouched
        self.mask[0] = 1.0
        self.query_cost_total += abs(CFG.query_struct_cost)
        return CFG.query_struct_cost

    def _query_cnn(self) -> float:
        if not self.cnn_available:
            return CFG.unavailable_query_penalty
        if self.mask[1] == 1.0:
            return CFG.repeat_query_penalty
        score = self.cache.cnn(self.current_url)
        confidence = entropy_confidence(score)
        self.aux[1] = score
        self.aux[15] = confidence
        self.mask[1] = 1.0
        self.query_cost_total += abs(CFG.query_cnn_cost)
        if self.mask[2] == 1.0:
            self.aux[17] = abs(self.aux[1] - self.aux[0])
        # Info-gain shaping: a confident (informative) CNN score offsets more
        # of the flat query cost than one that comes back near 0.5.
        return CFG.query_cnn_cost + confidence * CFG.info_gain_weight

    def _query_bert(self) -> float:
        if not self.bert_available:
            return CFG.unavailable_query_penalty
        if self.mask[2] == 1.0:
            return CFG.repeat_query_penalty
        score = self.cache.bert(self.current_url)
        confidence = entropy_confidence(score)
        self.aux[0] = score
        self.aux[16] = confidence
        self.mask[2] = 1.0
        self.query_cost_total += abs(CFG.query_bert_cost)
        if self.mask[1] == 1.0:
            self.aux[17] = abs(self.aux[1] - self.aux[0])
        return CFG.query_bert_cost + confidence * CFG.info_gain_weight

    def _disagreement_bonus(self) -> float:
        if self.mask[1] == 1.0 and self.mask[2] == 1.0 and self.aux[17] >= CFG.disagreement_aux17_threshold:
            return CFG.disagreement_resolution_bonus
        return 0.0

    def _decision_reward(self, action: int) -> float:
        gt = self.true_label

        if action == self.DEFER:
            return CFG.r_abstain
        if action == self.CLASSIFY_PHISHING and gt == 1:
            return CFG.r_tp + self._disagreement_bonus()
        if action == self.CLASSIFY_BENIGN and gt == 0:
            return CFG.r_tn + self._disagreement_bonus()
        if action == self.CLASSIFY_PHISHING and gt == 0:
            return CFG.r_fp
        return CFG.r_fn

    def _timeout_reward(self) -> float:
        return CFG.r_abstain + CFG.timeout_defer_penalty_extra

    def step(self, action: int):
        action = int(action)
        self.actions_taken.append(action)
        self.current_step += 1
        done = False
        truncated = False
        final_action = None

        if action == self.QUERY_STRUCT:
            reward = self._query_struct()
        elif action == self.QUERY_CNN:
            reward = self._query_cnn()
        elif action == self.QUERY_BERT:
            reward = self._query_bert()
        elif action in (self.CLASSIFY_BENIGN, self.CLASSIFY_PHISHING, self.DEFER):
            reward = self._decision_reward(action)
            done = True
            final_action = action
        else:
            reward = -0.25

        if not done and self.current_step >= CFG.max_steps:
            reward += self._timeout_reward()
            done = True
            truncated = True
            final_action = self.DEFER

        info = {
            "url": self.current_url,
            "label": self.true_label,
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
        return self._obs(), float(reward), done, truncated, info



# PPO FEATURE EXTRACTOR
class MultistepHybridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = CFG.features_dim):
        super().__init__(observation_space, features_dim)
        vocab = int(observation_space.spaces["tokens"].high[0]) + 1
        side_dim = (
            int(observation_space.spaces["aux"].shape[0])
            + int(observation_space.spaces["mask"].shape[0])
            + int(observation_space.spaces["step"].shape[0])
        )

        self.emb = nn.Embedding(vocab, CFG.token_emb_dim)
        self.token_cnn = nn.Sequential(
            nn.Conv1d(CFG.token_emb_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.side_mlp = nn.Sequential(
            nn.Linear(side_dim, CFG.side_hidden_dim),
            nn.ReLU(),
            nn.Linear(CFG.side_hidden_dim, CFG.side_hidden_dim),
            nn.ReLU(),
        )
        self.final = nn.Sequential(
            nn.Linear(64 + CFG.side_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        token_features = self.token_cnn(self.emb(obs["tokens"].long()).permute(0, 2, 1))
        side = torch.cat([obs["aux"].float(), obs["mask"].float(), obs["step"].float()], dim=1)
        side_features = self.side_mlp(side)
        return self.final(torch.cat([token_features, side_features], dim=1))



# EVALUATION

ACTION_NAMES = {
    0: "query_struct",
    1: "query_cnn",
    2: "query_bert",
    3: "benign",
    4: "phishing",
    5: "defer",
}


def run_one_episode(
    model: PPO,
    env: MultistepPhishingEnv,
    url: str,
    label: int,
    force_cnn_available: Optional[bool] = None,
    force_bert_available: Optional[bool] = None,
):
    options = {"url": url, "label": label}
    if force_cnn_available is not None:
        options["force_cnn_available"] = force_cnn_available
    if force_bert_available is not None:
        options["force_bert_available"] = force_bert_available

    obs, _ = env.reset(options=options)
    done = False
    total_reward = 0.0
    last_info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        done = terminated or truncated
        last_info = info
    final_action = last_info.get("final_action")
    if final_action is None:
        final_action = MultistepPhishingEnv.DEFER
    return int(final_action), total_reward, last_info


def compute_metrics(counts: Counter) -> dict:
    tp = counts.get("tp", 0)
    fp = counts.get("fp", 0)
    fn = counts.get("fn", 0)
    tn = counts.get("tn", 0)
    defer_phish = counts.get("defer_phish", 0)
    defer_benign = counts.get("defer_benign", 0)
    total = tp + fp + fn + tn + defer_phish + defer_benign
    total_phish = tp + fn + defer_phish
    total_benign = tn + fp + defer_benign

    precision = tp / (tp + fp + 1e-8)
    strict_recall = tp / (total_phish + 1e-8)
    recall_non_deferred = tp / (tp + fn + 1e-8)
    strict_f1 = 2 * precision * strict_recall / (precision + strict_recall + 1e-8)
    f1_non_deferred = 2 * precision * recall_non_deferred / (precision + recall_non_deferred + 1e-8)
    strict_accuracy = (tp + tn) / (total + 1e-8)
    detection_rate = tp / (total_phish + 1e-8) if total_phish else 0.0
    benign_accuracy = tn / (total_benign + 1e-8) if total_benign else 0.0
    fpr = fp / (total_benign + 1e-8) if total_benign else 0.0
    defer_rate = (defer_phish + defer_benign) / (total + 1e-8)

    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        defer_phish=defer_phish,
        defer_benign=defer_benign,
        total=total,
        precision=precision,
        strict_recall=strict_recall,
        recall_non_deferred=recall_non_deferred,
        strict_f1=strict_f1,
        f1_non_deferred=f1_non_deferred,
        strict_accuracy=strict_accuracy,
        detection_rate=detection_rate,
        benign_accuracy=benign_accuracy,
        false_positive_rate=fpr,
        defer_rate=defer_rate,
    )


def _evaluate_population(
    model: PPO,
    env: MultistepPhishingEnv,
    urls: List[str],
    labels: List[int],
    force_cnn_available: Optional[bool],
    force_bert_available: Optional[bool],
) -> dict:
    counts = Counter()
    query_counts, query_costs, cnn_used, bert_used, rewards = [], [], [], [], []
    paths = Counter()

    for u, y in zip(urls, labels):
        final_action, ep_reward, info = run_one_episode(
            model, env, u, y,
            force_cnn_available=force_cnn_available,
            force_bert_available=force_bert_available,
        )
        if final_action == MultistepPhishingEnv.DEFER:
            counts["defer_phish" if y == 1 else "defer_benign"] += 1
        elif final_action == MultistepPhishingEnv.CLASSIFY_PHISHING and y == 1:
            counts["tp"] += 1
        elif final_action == MultistepPhishingEnv.CLASSIFY_PHISHING and y == 0:
            counts["fp"] += 1
        elif final_action == MultistepPhishingEnv.CLASSIFY_BENIGN and y == 1:
            counts["fn"] += 1
        elif final_action == MultistepPhishingEnv.CLASSIFY_BENIGN and y == 0:
            counts["tn"] += 1

        query_counts.append(info.get("query_count", 0))
        query_costs.append(info.get("query_cost", 0.0))
        cnn_used.append(info.get("queried_cnn", 0))
        bert_used.append(info.get("queried_bert", 0))
        rewards.append(ep_reward)
        paths[tuple(ACTION_NAMES.get(a, str(a)) for a in info.get("actions_taken", []))] += 1

    metrics = compute_metrics(counts)
    avg_cost = float(np.mean(query_costs)) if query_costs else 0.0
    metrics.update({
        "avg_query_count": float(np.mean(query_counts)) if query_counts else 0.0,
        "avg_query_cost": avg_cost,
        "cnn_query_rate": float(np.mean(cnn_used)) if cnn_used else 0.0,
        "bert_query_rate": float(np.mean(bert_used)) if bert_used else 0.0,
        "avg_episode_reward": float(np.mean(rewards)) if rewards else 0.0,
        "cost_adjusted_score": metrics["strict_f1"] - 0.25 * avg_cost,
        "top_action_paths": str(paths.most_common(10)),
    })
    return metrics


def evaluate_model(
    model: PPO,
    env: MultistepPhishingEnv,
    n: int,
    seed: int,
    name: str,
    force_cnn_available: Optional[bool] = None,
    force_bert_available: Optional[bool] = None,
) -> dict:
    rng = random.Random(seed)
    benign_sample = rng.sample(eval_benign, min(n, len(eval_benign)))
    phish_sample = rng.sample(eval_phishing, min(n, len(eval_phishing)))
    urls = phish_sample + benign_sample
    labels = [1] * len(phish_sample) + [0] * len(benign_sample)
    metrics = _evaluate_population(model, env, urls, labels, force_cnn_available, force_bert_available)
    metrics["name"] = name
    return metrics


def evaluate_dependency_metrics(model: PPO, env: MultistepPhishingEnv, n: int, seed: int, cnn_threshold: float = 0.5) -> dict:

    rng = random.Random(seed)
    benign_sample = rng.sample(eval_benign, min(n, len(eval_benign)))
    phish_sample = rng.sample(eval_phishing, min(n, len(eval_phishing)))
    cnn_scores_benign = cnn_score_batch(benign_sample)
    cnn_scores_phish = cnn_score_batch(phish_sample)
    cnn_fn_urls = [u for u, s in zip(phish_sample, cnn_scores_phish) if s < cnn_threshold]
    cnn_fp_urls = [u for u, s in zip(benign_sample, cnn_scores_benign) if s >= cnn_threshold]

    def _override_rate(urls: List[str], true_label: int) -> tuple[float, int]:
        if not urls:
            return 0.0, 0
        wanted = MultistepPhishingEnv.CLASSIFY_PHISHING if true_label == 1 else MultistepPhishingEnv.CLASSIFY_BENIGN
        correct = sum(
            1 for u in urls
            if run_one_episode(
                model, env, u, true_label,
                force_cnn_available=False, force_bert_available=True,
            )[0] == wanted
        )
        return correct / len(urls), len(urls)

    fn_override_rate, fn_pool = _override_rate(cnn_fn_urls, 1)
    fp_override_rate, fp_pool = _override_rate(cnn_fp_urls, 0)
    return {
        "cnn_fn_pool_size": fn_pool,
        "cnn_fp_pool_size": fp_pool,
        "cnn_fn_override_rate": fn_override_rate,
        "cnn_fp_override_rate": fp_override_rate,
    }


def evaluate_disagreement_resolution(model: PPO, env: MultistepPhishingEnv) -> dict:

    def _accuracy(pool: list) -> tuple[float, int]:
        if not pool:
            return 0.0, 0
        correct = 0
        for u, label in pool:
            wanted = MultistepPhishingEnv.CLASSIFY_PHISHING if label == 1 else MultistepPhishingEnv.CLASSIFY_BENIGN
            final_action, _, _ = run_one_episode(
                model, env, u, label, force_cnn_available=True, force_bert_available=True,
            )
            if final_action == wanted:
                correct += 1
        return correct / len(pool), len(pool)

    cnn_right_acc, cnn_right_n = _accuracy(eval_cnn_right_bert_wrong)
    bert_right_acc, bert_right_n = _accuracy(eval_bert_right_cnn_wrong)
    total_n = cnn_right_n + bert_right_n
    overall = (
        (cnn_right_acc * cnn_right_n + bert_right_acc * bert_right_n) / total_n
        if total_n else 0.0
    )

    joint_failure_acc, joint_failure_n = _accuracy(eval_both_wrong)
    return {
        "disagreement_cnn_right_pool": cnn_right_n,
        "disagreement_cnn_right_resolved": cnn_right_acc,
        "disagreement_bert_right_pool": bert_right_n,
        "disagreement_bert_right_resolved": bert_right_acc,
        "disagreement_overall_resolved": overall,
        "joint_failure_pool": joint_failure_n,
        "joint_failure_recovery_rate": joint_failure_acc,
    }


def run_eval_suite(model: PPO, env: MultistepPhishingEnv, seed: int, name: str) -> dict:

    full = evaluate_model(model, env, CFG.eval_size, seed, "full_system",
                           force_cnn_available=True, force_bert_available=True)
    cnn_masked = evaluate_model(model, env, CFG.robustness_eval_size, seed + 1, "cnn_masked",
                                 force_cnn_available=False, force_bert_available=True)
    bert_masked = evaluate_model(model, env, CFG.robustness_eval_size, seed + 2, "bert_masked",
                                  force_cnn_available=True, force_bert_available=False)
    dep = evaluate_dependency_metrics(model, env, CFG.dependency_eval_size, seed + 3)
    disagreement = evaluate_disagreement_resolution(model, env)

    row = {"name": name}
    for prefix, m in (("full", full), ("cnnmasked", cnn_masked), ("bertmasked", bert_masked)):
        for k, v in m.items():
            if k in ("name", "top_action_paths"):
                continue
            row[f"{prefix}_{k}"] = v
    for k, v in dep.items():
        row[f"dep_{k}"] = v
    for k, v in disagreement.items():
        row[k] = v
    return row


MIN_OVERRIDE_POOL_SIZE = 15


def _fmt_override(rate: float, pool: int) -> str:
    if pool < MIN_OVERRIDE_POOL_SIZE:
        return f"n/a, pool too small (n={pool})"
    return f"{rate:.1%} (n={pool})"


def print_eval_suite(row: dict):
    print(
        f"[{row['name']}] "
        f"FULL: F1={row['full_strict_f1']:.4f} BenignAcc={row['full_benign_accuracy']:.4f} "
        f"Det={row['full_detection_rate']:.4f} CNN%={row['full_cnn_query_rate']:.1%} BERT%={row['full_bert_query_rate']:.1%} | "
        f"CNN-MASKED: F1={row['cnnmasked_strict_f1']:.4f} BenignAcc={row['cnnmasked_benign_accuracy']:.4f} | "
        f"BERT-MASKED: F1={row['bertmasked_strict_f1']:.4f} BenignAcc={row['bertmasked_benign_accuracy']:.4f} | "
        f"CNN-wrong override: FN={_fmt_override(row['dep_cnn_fn_override_rate'], row['dep_cnn_fn_pool_size'])} "
        f"FP={_fmt_override(row['dep_cnn_fp_override_rate'], row['dep_cnn_fp_pool_size'])} | "
        f"Disagreement resolved: CNN-was-right={_fmt_override(row['disagreement_cnn_right_resolved'], row['disagreement_cnn_right_pool'])} "
        f"BERT-was-right={_fmt_override(row['disagreement_bert_right_resolved'], row['disagreement_bert_right_pool'])} "
        f"overall={row['disagreement_overall_resolved']:.1%}"
    )


# CALLBACKS


class DeferredCaptureCallback(BaseCallback):

    def __init__(self):
        super().__init__()
        self.repeat_defer_events = 0
        self.total_defer_events = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("final_action") != MultistepPhishingEnv.DEFER:
                continue
            url = info.get("url")
            label = info.get("label")
            if url is None or label is None:
                continue
            self.total_defer_events += 1
            if url in deferred_defer_counts:
                self.repeat_defer_events += 1
            deferred_defer_counts[url] = deferred_defer_counts.get(url, 0) + 1
            deferred_replay_buffer.append((url, int(label)))
        return True

    def repeat_defer_rate(self) -> float:
        return self.repeat_defer_events / max(1, self.total_defer_events)


class EvalCallback(BaseCallback):
    def __init__(self, eval_env: MultistepPhishingEnv, defer_callback: "DeferredCaptureCallback" = None):
        super().__init__()
        self.eval_env = eval_env
        self.defer_callback = defer_callback
        self.last_eval = 0
        self.best_qualified_score = -float("inf")
        self.best_unconstrained_score = -float("inf")
        self.any_qualified = False
        self.rows = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval < CFG.eval_every_steps:
            return True
        self.last_eval = self.num_timesteps

        row = run_eval_suite(self.model, self.eval_env, seed=CFG.seed + self.num_timesteps, name=f"step_{self.num_timesteps}")
        row["timesteps"] = self.num_timesteps
        if self.defer_callback is not None:
            row["deferred_buffer_size"] = len(deferred_replay_buffer)
            row["deferred_unique_urls"] = len(deferred_defer_counts)
            row["repeat_defer_rate"] = self.defer_callback.repeat_defer_rate()
            row["total_defer_events"] = self.defer_callback.total_defer_events
            print(f"  [DEFER] buffer={row['deferred_buffer_size']} unique={row['deferred_unique_urls']} "
                  f"repeat_defer_rate={row['repeat_defer_rate']:.1%} (of {row['total_defer_events']} defer events this run)")
        self.rows.append(row)
        print_eval_suite(row)
        pd.DataFrame(self.rows).to_csv(OUTPUT_DIR / "multistep_training_eval_log_v1-5.csv", index=False)

        score = float(row.get(f"full_{CFG.save_best_by}", row["full_strict_f1"]))
        benign_acc = float(row["full_benign_accuracy"])
        qualifies = benign_acc >= CFG.benign_accuracy_gate

        if qualifies:
            self.any_qualified = True
            if score > self.best_qualified_score:
                self.best_qualified_score = score
                self.model.save(str(OUTPUT_DIR / "best_multistep_hybrid_drl_unicode_v1-5"))
                print(f"  [BEST-QUALIFIED] benign_acc={benign_acc:.4f} {CFG.save_best_by}={score:.4f} saved")

        if score > self.best_unconstrained_score:
            self.best_unconstrained_score = score
            self.model.save(str(OUTPUT_DIR / "best_multistep_hybrid_drl_unicode_v1-5_unconstrained"))

        return True



# TRAINING

def build_model(env: MultistepPhishingEnv) -> PPO:
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    policy_kwargs = dict(
        features_extractor_class=MultistepHybridExtractor,
        features_extractor_kwargs=dict(features_dim=CFG.features_dim),
        net_arch=[],
    )

    if CFG.resume_checkpoint:
        ckpt = CFG.resume_checkpoint if CFG.resume_checkpoint.endswith(".zip") else CFG.resume_checkpoint + ".zip"
        if Path(ckpt).exists():
            print(f"Loading checkpoint: {ckpt}")
            return PPO.load(ckpt, env=vec_env, device=device)

    return PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=CFG.learning_rate,
        n_steps=CFG.n_steps,
        batch_size=CFG.batch_size,
        n_epochs=CFG.n_epochs,
        gamma=CFG.gamma,
        gae_lambda=CFG.gae_lambda,
        clip_range=CFG.clip_range,
        ent_coef=CFG.ent_coef,
        vf_coef=CFG.vf_coef,
        max_grad_norm=CFG.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
    )


def debug_episode(model: PPO, env: MultistepPhishingEnv, url: str, label: int):
    final_action, ep_reward, info = run_one_episode(model, env, url, label)
    path = [ACTION_NAMES.get(a, str(a)) for a in info.get("actions_taken", [])]
    print("\nDebug episode")
    print("-" * 80)
    print(f"URL   : {url[:120]}")
    print(f"Label : {'phishing' if label == 1 else 'benign'}")
    print(f"Path  : {' -> '.join(path)}")
    print(f"Final : {ACTION_NAMES.get(final_action, final_action)}")
    print(f"Reward: {ep_reward:.4f}")
    print(f"Info  : queries={info.get('query_count')} cost={info.get('query_cost'):.3f} "
          f"struct={info.get('queried_struct')} cnn={info.get('queried_cnn')} bert={info.get('queried_bert')} "
          f"cnn_available={info.get('cnn_available')} bert_available={info.get('bert_available')}")


def main():
    print("\nBuilding environments...")
    train_env = MultistepPhishingEnv(train_benign, train_phishing, score_cache)
    eval_env = MultistepPhishingEnv(eval_benign, eval_phishing, score_cache)

    print("Building PPO model...")
    model = build_model(train_env)

    print("\nBaseline evaluation before training...")
    base = run_eval_suite(model, eval_env, seed=CFG.seed, name="before_training")
    print_eval_suite(base)

    print("\nStarting PPO training...")
    print(f"Total timesteps: {CFG.total_timesteps}")
    print("Action space: 0=query_struct, 1=query_cnn, 2=query_bert, 3=benign, 4=phishing, 5=defer\n")

    defer_callback = DeferredCaptureCallback()
    callback = EvalCallback(eval_env, defer_callback=defer_callback)
    model.learn(total_timesteps=CFG.total_timesteps, callback=[defer_callback, callback], reset_num_timesteps=False)

    final_path = OUTPUT_DIR / "final_multistep_hybrid_drl_unicode_v1-5"
    model.save(str(final_path))
    print(f"\nFinal model saved -> {final_path}.zip")

    print("\nFinal evaluation...")
    final = run_eval_suite(model, eval_env, seed=CFG.seed + 999, name="final")
    print_eval_suite(final)
    pd.DataFrame([base, final]).to_csv(OUTPUT_DIR / "multistep_final_summary_v1-5.csv", index=False)

    if not callback.any_qualified:
        print(
            f"\n[WARNING] No checkpoint during this run ever reached "
            f"benign_accuracy >= {CFG.benign_accuracy_gate:.0%}. "
            f"'best_multistep_hybrid_drl_unicode_v1-5.zip' was never written -- "
            f"use 'best_multistep_hybrid_drl_unicode_v1-5_unconstrained.zip' "
            f"(best {CFG.save_best_by} with no gate) instead, and treat the "
            f"benign-accuracy target as not yet met."
        )
    else:
        print(f"\nBest qualified checkpoint: best_multistep_hybrid_drl_unicode_v1-5.zip "
              f"({CFG.save_best_by}={callback.best_qualified_score:.4f}, benign_accuracy >= {CFG.benign_accuracy_gate:.0%})")

    print(f"\nDeferred-replay buffer: {len(deferred_replay_buffer)} cases held "
          f"({len(deferred_defer_counts)} unique URLs ever deferred, "
          f"{defer_callback.total_defer_events} total defer events, "
          f"repeat-defer rate={defer_callback.repeat_defer_rate():.1%}). "
          f"before_training vs final disagreement_overall_resolved: "
          f"{base['disagreement_overall_resolved']:.1%} -> {final['disagreement_overall_resolved']:.1%}, "
          f"joint_failure_recovery_rate: {base['joint_failure_recovery_rate']:.1%} -> {final['joint_failure_recovery_rate']:.1%}")

    print("\nInspecting sample decision paths...")
    for u in random.sample(eval_phishing, min(3, len(eval_phishing))):
        debug_episode(model, eval_env, u, 1)
    for u in random.sample(eval_benign, min(3, len(eval_benign))):
        debug_episode(model, eval_env, u, 0)

    print("\nCache stats:")
    print(score_cache.stats())
    print(f"\nLogs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
