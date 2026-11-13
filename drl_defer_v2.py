import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces
from transformers import GPT2Tokenizer
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

# Hyperparameters (unchanged)
MAX_URL_LENGTH = 80
EMBEDDING_DIM = 128
FEATURES_DIM = 64

TOTAL_TIMESTEPS = 200_000
LR = 3e-4
N_STEPS = 4096
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
SEED = 44

#Clean, leakage-free data
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "CleanedDATA"
PHISH_TRAIN_PATH  = DATA_DIR / "Train_2025.csv"    # label 1
LEGIT_TRAIN_PATH  = DATA_DIR / "Train_Benign.csv"  # label 0
TEST_PHISH_2025   = DATA_DIR / "Test_2025.csv"     # label 1
TEST_PHISH_2026   = DATA_DIR / "Test_2026.csv"     # label 1
TEST_BENIGN_PATH  = DATA_DIR / "Test_Benign.csv"   # label 0

TOKENIZER_NAME = "gpt2"
SAVE_DIR = SCRIPT_DIR / "DRL_defer_v2_seed44"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# reward is stationary.
REWARD_CONFIG = {
    "r_tp": 1.0,
    "r_tn": 1.0,
    "r_fp": -1.0,
    "r_fn_base": -2.0,
    "fn_scale": 3.0,
    "r_defer_phish": 0.3,
    "r_defer_legit": -0.1,
}

ACTION_LEGIT, ACTION_PHISH, ACTION_DEFER = 0, 1, 2


def normalize_url(u: str) -> str:
    u = str(u).strip().replace("\\", "/")
    try:
        u = u.encode("idna").decode("ascii", errors="ignore")
    except Exception:
        pass
    return u.lower()


def load_urls(path, label_value: int) -> list[str]:
    df = pd.read_csv(path, low_memory=False)
    col = "url" if "url" in df.columns else df.columns[0]
    urls = [normalize_url(u) for u in df[col].dropna()]
    urls = [u for u in urls if u]
    return list(dict.fromkeys(urls))          # de-dup, preserve order


def encode_url(u: str, tokenizer: GPT2Tokenizer, max_len: int) -> np.ndarray:
    toks = tokenizer.encode(u, max_length=max_len, padding="max_length", truncation=True)
    return np.array(toks, dtype=np.int32)


def encode_batch(urls: list[str], tokenizer, max_len: int) -> np.ndarray:
    return np.stack([encode_url(u, tokenizer, max_len) for u in urls]) if urls \
        else np.empty((0, max_len), np.int32)



# ENVIRONMENT 3 actions)


class PhishingEnv(gym.Env):
    def __init__(self, tokenizer, legit_urls, phish_urls, max_len=MAX_URL_LENGTH,
                 mix_probs=(0.5, 0.5), seed=SEED, reward_config=None):
        super().__init__()
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer
        self.legit_urls = legit_urls
        self.phish_urls = phish_urls
        self.max_len = max_len
        self.mix_probs = mix_probs

        self.action_space = spaces.Discrete(3)  # 0=legit, 1=phish, 2=defer
        self.observation_space = spaces.Box(
            low=0, high=self.tokenizer.vocab_size - 1,
            shape=(self.max_len,), dtype=np.int32,
        )
        self.reward_config = reward_config or REWARD_CONFIG.copy()
        self.total_steps = 0
        self.fn_count = 0

    def _sample_url(self):
        r = self.rng.random()
        legit_p, _ = self.mix_probs
        if r < legit_p:
            return self.rng.choice(self.legit_urls), 0
        return self.rng.choice(self.phish_urls), 1

    def _obs_from_url(self, url: str):
        return encode_url(url, self.tokenizer, self.max_len)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_url, self.current_label = self._sample_url()
        return self._obs_from_url(self.current_url), {}

    def step(self, action):
        gt = self.current_label
        cfg = self.reward_config
        self.total_steps += 1
        reward = 0

        if action == ACTION_DEFER:
            reward = cfg["r_defer_phish"] if gt == 1 else cfg["r_defer_legit"]
        else:
            if action == ACTION_PHISH and gt == 1:
                reward = cfg["r_tp"]
            elif action == ACTION_LEGIT and gt == 0:
                reward = cfg["r_tn"]
            elif action == ACTION_PHISH and gt == 0:
                reward = cfg["r_fp"]
            elif action == ACTION_LEGIT and gt == 1:
                self.fn_count += 1
                fn_rate = self.fn_count / self.total_steps
                reward = cfg["r_fn_base"] * (1.0 + cfg.get("fn_scale", 0.0) * fn_rate)

        self.current_url, self.current_label = self._sample_url()
        return self._obs_from_url(self.current_url), reward, True, False, {}


class URLCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim=FEATURES_DIM, emb_dim=EMBEDDING_DIM):
        super().__init__(observation_space, features_dim)
        vocab_size = int(observation_space.high[0]) + 1
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, features_dim, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.embedding(obs.long()).permute(0, 2, 1)
        return self.cnn(x)



# HELD-OUT EVALUATION
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def pct(k, n):
    p, lo, hi = wilson(k, n)
    return f"{100 * p:5.1f}% [{100 * lo:.1f}-{100 * hi:.1f}]"


def _predict_actions(model, urls, tokenizer, device, batch=4096):
    out = []
    for i in range(0, len(urls), batch):
        obs = encode_batch(urls[i:i + batch], tokenizer, MAX_URL_LENGTH)
        actions, _ = model.predict(obs, deterministic=True)
        out.append(np.asarray(actions).reshape(-1))
    return np.concatenate(out) if out else np.empty(0, int)


def evaluate_full(model, tokenizer, device) -> list[dict]:
    benign = load_urls(TEST_BENIGN_PATH, 0)
    ben_act = _predict_actions(model, benign, tokenizer, device)

    results = []
    for name, phish_path in [("Test_2025", TEST_PHISH_2025), ("Test_2026", TEST_PHISH_2026)]:
        if not phish_path.exists():
            continue
        phish = load_urls(phish_path, 1)
        ph_act = _predict_actions(model, phish, tokenizer, device)

        # phishing (gt=1)
        tp = int((ph_act == ACTION_PHISH).sum())
        fn = int((ph_act == ACTION_LEGIT).sum())
        defer_p = int((ph_act == ACTION_DEFER).sum())
        # benign (gt=0)
        tn = int((ben_act == ACTION_LEGIT).sum())
        fp = int((ben_act == ACTION_PHISH).sum())
        defer_b = int((ben_act == ACTION_DEFER).sum())

        n = len(phish) + len(benign)
        committed = tp + fn + tn + fp
        coverage = committed / n
        defer_rate = (defer_p + defer_b) / n

        recall_committed = tp / (tp + fn) if (tp + fn) else float("nan")      # on committed phishing
        benign_committed = tn / (tn + fp) if (tn + fp) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = 2 * precision * recall_committed / (precision + recall_committed) \
            if precision and recall_committed else float("nan")
        acc_committed = (tp + tn) / committed if committed else float("nan")
        recall_strict = tp / len(phish)   # defer counts as a miss (conservative)

        print("\n" + "=" * 66)
        print(f"DRL defer-v1  —  {name} (phishing) + Test_Benign")
        print("=" * 66)
        print(f"  n = {n:,}  (phishing {len(phish):,}  benign {len(benign):,})")
        print(f"  Coverage (committed)             : {100 * coverage:5.1f}%   "
              f"(defer rate {100 * defer_rate:.1f}%)")
        print(f"    defer on phishing {defer_p:,}  |  defer on benign {defer_b:,}")
        print(f"  -- metrics on COMMITTED decisions --")
        print(f"  Phishing recall (committed)      : {pct(tp, tp + fn)}")
        print(f"  Benign accuracy (committed)      : {pct(tn, tn + fp)}")
        print(f"  Precision {precision:.4f}  Recall {recall_committed:.4f}  F1 {f1:.4f}")
        print(f"  Accuracy (committed)             : {acc_committed:.4f}")
        print(f"  -- conservative (defer = missed phishing) --")
        print(f"  Phishing recall (strict)         : {pct(tp, len(phish))}")
        print(f"  Confusion (committed) [rows=actual, cols=pred legit/phish]:")
        print(f"    act BENIGN  {tn:>8}  {fp:>8}")
        print(f"    act PHISH   {fn:>8}  {tp:>8}")

        results.append({
            "name": name, "n": n, "coverage": coverage, "defer_rate": defer_rate,
            "defer_phish": defer_p, "defer_benign": defer_b,
            "recall_committed": recall_committed, "benign_committed": benign_committed,
            "precision": precision, "f1": f1, "accuracy_committed": acc_committed,
            "recall_strict_defer_as_miss": recall_strict,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    out = SAVE_DIR / "test_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")
    return results


# MAIN
def make_tokenizer():
    tok = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)
    tok.pad_token = tok.eos_token
    return tok


def main():
    parser = argparse.ArgumentParser(description="DRL defer-v1 — leakage-free train + full eval")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--ckpt", type=str, default=str(SAVE_DIR / "ppo_phish_detector"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(SEED)
    tokenizer = make_tokenizer()

    print(f"Device: {device}  |  mode: {args.mode}")

    if args.mode == "train":
        train_legit = load_urls(LEGIT_TRAIN_PATH, 0)
        train_phish = load_urls(PHISH_TRAIN_PATH, 1)
        print(f"Train — benign: {len(train_legit):,}   phishing: {len(train_phish):,}")

        train_env = DummyVecEnv([lambda: Monitor(
            PhishingEnv(tokenizer, train_legit, train_phish))])

        policy_kwargs = dict(features_extractor_class=URLCnnExtractor,
                             features_extractor_kwargs=dict(features_dim=FEATURES_DIM,
                                                            emb_dim=EMBEDDING_DIM),
                             net_arch=[])
        model = PPO("MlpPolicy", train_env, learning_rate=LR, n_steps=N_STEPS,
                    batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, gamma=GAMMA,
                    gae_lambda=GAE_LAMBDA, clip_range=CLIP_RANGE, ent_coef=ENTROPY_COEF,
                    vf_coef=VF_COEF, max_grad_norm=MAX_GRAD_NORM, verbose=1,
                    policy_kwargs=policy_kwargs, device=device, seed=SEED,
                    tensorboard_log=str(SAVE_DIR / "tb"))

        print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps ...")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
        model.save(args.ckpt)
        print(f"Saved model -> {args.ckpt}")

        evaluate_full(model, tokenizer, device)

    elif args.mode == "eval":
        model = PPO.load(args.ckpt, device=device)
        evaluate_full(model, tokenizer, device)


if __name__ == "__main__":
    main()
