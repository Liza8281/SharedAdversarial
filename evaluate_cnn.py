import os
os.environ.setdefault("USE_TF", "0")
import sys
import json
import random
from pathlib import Path
import numpy as np
import torch
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import CLeanURLNet as U


BENIGN_TEST_CSV = SCRIPT_DIR / "CleanedDATA" / "Test_Benign.csv"
PHISH_TEST_CSV  = SCRIPT_DIR / "CleanedDATA" / "homoglyph_from_tranco.csv"

# Trained URLNet checkpoint to evaluate (produced by CLeanURLNet.py).
CHECKPOINT_DIR  = SCRIPT_DIR / "CleanURLNET"

VOCAB_DIR = SCRIPT_DIR / "CleanURLNET"

THRESHOLD = 0.9

# Class ratio used for the reported metrics, expressed as  phishing : benign.
#   None: use every URL in both files
#   1.0 : equal numbers of phishing and benign
#   0.5 : half as many phishing as benign
#   2.0 :twice as many phishing as benign

PHISH_PER_BENIGN = None
BATCH_SIZE = 512
SEED = 42


def load_cnn(checkpoint_dir, vocab_dir):
    checkpoint_dir = Path(checkpoint_dir)

    # pick the weights file
    if (checkpoint_dir / "best_model.pt").exists():
        weights = checkpoint_dir / "best_model.pt"
    elif (checkpoint_dir / "urlnet.pt").exists():
        weights = checkpoint_dir / "urlnet.pt"
    else:
        sys.exit(f"ERROR: no best_model.pt or urlnet.pt in {checkpoint_dir}")
    vocab_src = checkpoint_dir if (checkpoint_dir / "char_dict.json").exists() else Path(vocab_dir)
    for f in ("char_dict.json", "word_dict.json", "ngram_dict.json", "config.json"):
        if not (vocab_src / f).exists():
            sys.exit(f"ERROR: vocab/config file missing: {vocab_src / f}")

    U.dicts_from_dir(str(vocab_src))                       # sets U.char_dict / word_dict / ngram_dict
    with open(vocab_src / "config.json") as fh:
        cfg = json.load(fh)
    model = U.URLNet(cfg["char_vocab"], cfg["word_vocab"], cfg["ngram_vocab"]).to(U.DEVICE)
    model.load_state_dict(torch.load(weights, map_location=U.DEVICE))
    model.eval()
    print(f"  weights    : {weights}")
    print(f"  vocab/cfg  : {vocab_src}")
    return model


def phish_prob(model, urls, batch=BATCH_SIZE):
    """P(class == phishing) for each URL, using URLNet's own encoder/vocabs."""
    if not urls:
        return np.zeros(0, dtype=np.float32)
    model.eval()
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(urls), batch):
            enc = [U._encode(u) for u in urls[i:i + batch]]
            x_char = torch.as_tensor(np.stack([e[0] for e in enc]), device=dev)
            x_word = torch.as_tensor(np.stack([e[1] for e in enc]), device=dev)
            x_wc   = torch.as_tensor(np.stack([e[2] for e in enc]), device=dev)
            probs = torch.softmax(model(x_char, x_word, x_wc), dim=1)[:, 1]
            out.append(probs.detach().cpu().numpy())
    return np.concatenate(out)


def apply_ratio(n_phish, n_benign, phish_per_benign, rng):
    if phish_per_benign is None:
        return n_phish, n_benign
    r = float(phish_per_benign)
    if r <= 0:
        raise ValueError("PHISH_PER_BENIGN must be > 0 or None")
    # b = min(n_benign, n_phish / r); p = r * b  -> guarantees p<=n_phish, b<=n_benign
    keep_benign = int(min(n_benign, n_phish / r))
    keep_phish = int(round(r * keep_benign))
    keep_phish = min(keep_phish, n_phish)
    keep_benign = max(keep_benign, 1)
    keep_phish = max(keep_phish, 1)
    return keep_phish, keep_benign


def subsample(urls, k, rng):
    if k >= len(urls):
        return urls
    return rng.sample(urls, k)


def metrics_at(pp, pb, thr):
    """pp = phishing-set P_phish, pb = benign-set P_phish. Phishing is the positive class."""
    tp = int((pp >= thr).sum()); fn = int((pp < thr).sum())
    fp = int((pb >= thr).sum()); tn = int((pb < thr).sum())
    recall    = tp / (tp + fn) if (tp + fn) else 0.0   # phishing caught
    benignacc = tn / (tn + fp) if (tn + fp) else 0.0   # benign passed (specificity)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, recall=recall, benign_acc=benignacc,
                precision=precision, f1=f1, accuracy=accuracy)


def resolve_threshold(ckpt_dir, thr):
    if thr is not None:
        return float(thr)
    tf = Path(ckpt_dir) / "optimal_threshold.txt"
    if tf.exists():
        try:
            return float(tf.read_text().strip())
        except Exception:
            pass
    return 0.5


def main():
    rng = random.Random(SEED)
    np.random.seed(SEED); torch.manual_seed(SEED)

    thr = resolve_threshold(CHECKPOINT_DIR, THRESHOLD)
    print("=" * 70)
    print("URLNet CNN evaluation")
    print("=" * 70)
    print(f"checkpoint : {CHECKPOINT_DIR}")
    print(f"benign CSV : {BENIGN_TEST_CSV}")
    print(f"phish  CSV : {PHISH_TEST_CSV}")
    print(f"threshold  : {thr}")
    print(f"device     : {U.DEVICE}")

    for p in (BENIGN_TEST_CSV, PHISH_TEST_CSV, CHECKPOINT_DIR):
        if not Path(p).exists():
            sys.exit(f"ERROR: path does not exist: {p}")

    model = load_cnn(CHECKPOINT_DIR, VOCAB_DIR)

    benign_urls, _ = U._load_urls(BENIGN_TEST_CSV, 0)
    phish_urls,  _ = U._load_urls(PHISH_TEST_CSV, 1)
    print(f"\nloaded (de-duped): benign={len(benign_urls)}  phishing={len(phish_urls)}")

    # apply the manual class ratio
    keep_p, keep_b = apply_ratio(len(phish_urls), len(benign_urls), PHISH_PER_BENIGN, rng)
    phish_urls  = subsample(phish_urls, keep_p, rng)
    benign_urls = subsample(benign_urls, keep_b, rng)
    if PHISH_PER_BENIGN is None:
        print("ratio      : natural (all URLs used)")
    else:
        got = keep_p / keep_b if keep_b else float("inf")
        print(f"ratio      : target phish:benign = {PHISH_PER_BENIGN} : 1  "
              f"-> using phishing={keep_p}  benign={keep_b}  (actual {got:.3f} : 1)")

    # score
    pp = phish_prob(model, phish_urls)
    pb = phish_prob(model, benign_urls)

    m = metrics_at(pp, pb, thr)

    print("\n" + "-" * 70)
    print(f"confusion @ thr={thr}   (positive = phishing)")
    print(f"  TP={m['tp']:6d}   FN={m['fn']:6d}   (phishing set, n={m['tp']+m['fn']})")
    print(f"  FP={m['fp']:6d}   TN={m['tn']:6d}   (benign   set, n={m['fp']+m['tn']})")
    print("-" * 70)
    print(f"  recall    (phishing caught) : {m['recall']:.4f}")
    print(f"  benign_acc(benign passed)   : {m['benign_acc']:.4f}")
    print(f"  precision                   : {m['precision']:.4f}")
    print(f"  f1                          : {m['f1']:.4f}")
    print(f"  accuracy                    : {m['accuracy']:.4f}")

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        y = np.concatenate([np.ones_like(pp), np.zeros_like(pb)])
        s = np.concatenate([pp, pb])
        print("-" * 70)
        print(f"  ROC-AUC  (ratio-independent) : {roc_auc_score(y, s):.4f}")
        print(f"  PR-AUC   (ratio-dependent)   : {average_precision_score(y, s):.4f}")
    except Exception as e:
        print(f"  (AUC skipped: {e})")
    print("=" * 70)


if __name__ == "__main__":
    main()
