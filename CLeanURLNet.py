import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Clean data
SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "CleanedDATA"
TRAIN_PHISH_CSV  = DATA_DIR / "Train_2025.csv"    # label 1
TRAIN_BENIGN_CSV = DATA_DIR / "Train_Benign.csv"  # label 0
OUTPUT_DIR  = SCRIPT_DIR / "CleanURLNET"
EMB_DIM          = 32
FILTER_SIZES     = (3, 4, 5, 6)
NUM_FILTERS      = 256
FC_LAYERS        = (512, 256, 128)
MAX_LEN_CHARS    = 200     # whole-URL character sequence
MAX_LEN_WORDS    = 200     # words per URL
MAX_LEN_SUBWORDS = 20      # characters per word (incl. the < > markers)
MIN_WORD_FREQ    = 1       # keep words appearing >= this many times in train
PROJ_DIM         = 512     # per-branch projection before concat
BATCH_SIZE   = 128
EPOCHS       = 5
LR           = 1e-3
DROPOUT      = 0.5         # = 1 - dropout_keep_prob(0.5)
L2_REG       = 0.0
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORD_RE = re.compile(r"[a-zA-Z0-9]+|[^a-zA-Z0-9]")

char_dict:  dict[str, int] = {}   # whole-URL characters
word_dict:  dict[str, int] = {}   # words
ngram_dict: dict[str, int] = {}   # characters inside a word


def url_words(url: str) -> list[str]:
    return WORD_RE.findall(url)[:MAX_LEN_WORDS]


def _word_is_unknown(tok: str) -> bool:
    return (len(tok) + 2) > MAX_LEN_SUBWORDS


def build_vocabs(urls: list[str]):
    global char_dict, word_dict, ngram_dict
    chars, ngrams, wcount = set(), {"<", ">"}, Counter()
    for u in urls:
        chars.update(u[:MAX_LEN_CHARS])
        for tok in url_words(u):
            wcount[tok] += 1
            ngrams.update(tok)
    char_dict  = {c: i + 1 for i, c in enumerate(sorted(chars))}            # 0 = pad/unk
    ngram_dict = {c: i + 1 for i, c in enumerate(sorted(ngrams))}           # 0 = pad
    kept = sorted(w for w, c in wcount.items() if c >= MIN_WORD_FREQ)
    word_dict  = {w: i + 2 for i, w in enumerate(kept)}                     # 0 = pad, 1 = UNK
    print(f"  Vocab sizes -> char: {len(char_dict)+1}  word: {len(word_dict)+2}  "
          f"word-char: {len(ngram_dict)+1}")


def _encode(url: str):
    cs = np.zeros(MAX_LEN_CHARS, np.int64)
    ids = [char_dict.get(c, 0) for c in url[:MAX_LEN_CHARS]]
    cs[:len(ids)] = ids

    wd = np.zeros(MAX_LEN_WORDS, np.int64)
    wc = np.zeros((MAX_LEN_WORDS, MAX_LEN_SUBWORDS), np.int64)
    for j, tok in enumerate(url_words(url)):
        wd[j] = 1 if _word_is_unknown(tok) else word_dict.get(tok, 1)
        chars = (["<"] + list(tok) + [">"])[:MAX_LEN_SUBWORDS]   # char_ngram_len = 1
        cids = [ngram_dict.get(c, 0) for c in chars]
        wc[j, :len(cids)] = cids
    return cs, wd, wc


def dicts_to_dir(output_dir):
    for name, d in [("char_dict.json", char_dict),
                    ("word_dict.json", word_dict),
                    ("ngram_dict.json", ngram_dict)]:
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            json.dump(d, f)


def dicts_from_dir(output_dir):
    global char_dict, word_dict, ngram_dict
    with open(os.path.join(output_dir, "char_dict.json"), encoding="utf-8") as f:
        char_dict = json.load(f)
    with open(os.path.join(output_dir, "word_dict.json"), encoding="utf-8") as f:
        word_dict = json.load(f)
    with open(os.path.join(output_dir, "ngram_dict.json"), encoding="utf-8") as f:
        ngram_dict = json.load(f)
    print(f"  Loaded vocabs -> char: {len(char_dict)+1}  word: {len(word_dict)+2}  "
          f"word-char: {len(ngram_dict)+1}")


class URLNetDataset(Dataset):
    def __init__(self, urls: list[str], labels: list[int]):
        n = len(urls)
        self.char_seq  = np.zeros((n, MAX_LEN_CHARS), np.int64)
        self.word      = np.zeros((n, MAX_LEN_WORDS), np.int64)
        self.word_char = np.zeros((n, MAX_LEN_WORDS, MAX_LEN_SUBWORDS), np.int16)
        self.labels    = np.asarray(labels, np.int64)
        for i, u in enumerate(urls):
            cs, wd, wc = _encode(u)
            self.char_seq[i] = cs
            self.word[i]     = wd
            self.word_char[i] = wc.astype(np.int16)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (torch.from_numpy(self.char_seq[i]),
                torch.from_numpy(self.word[i]),
                torch.from_numpy(self.word_char[i].astype(np.int64)),
                int(self.labels[i]))


def _load_urls(path: Path, label: int) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    col = "url" if "url" in df.columns else df.columns[0]
    urls = [str(u).strip() for u in df[col].dropna() if str(u).strip()]
    urls = list(dict.fromkeys(urls))          # de-dup
    return urls, [label] * len(urls)


def build_train_dataset() -> URLNetDataset:
    print("Loading clean training data (CleanedDATA)...")
    phish_urls,  phish_lbls  = _load_urls(TRAIN_PHISH_CSV,  1)
    benign_urls, benign_lbls = _load_urls(TRAIN_BENIGN_CSV, 0)
    print(f"  benign: {len(benign_urls)}   phishing: {len(phish_urls)}")

    urls   = benign_urls + phish_urls
    labels = benign_lbls + phish_lbls

    print("Building vocabularies (train only)...")
    build_vocabs(urls)
    print("Encoding URLs...")
    return URLNetDataset(urls, labels)

class URLNet(nn.Module):
    def __init__(self, char_vocab: int, word_vocab: int, ngram_vocab: int):
        super().__init__()
        # Embeddings (padding_idx=0 so padded positions contribute zeros)
        self.char_seq_emb = nn.Embedding(char_vocab,  EMB_DIM, padding_idx=0)  # whole-URL chars
        self.word_emb     = nn.Embedding(word_vocab,  EMB_DIM, padding_idx=0)  # words
        self.ngram_emb    = nn.Embedding(ngram_vocab, EMB_DIM, padding_idx=0)  # chars-in-word

        # Two independent multi-kernel conv stacks (VALID padding = padding 0)
        self.char_convs = nn.ModuleList(
            [nn.Conv1d(EMB_DIM, NUM_FILTERS, k) for k in FILTER_SIZES])
        self.word_convs = nn.ModuleList(
            [nn.Conv1d(EMB_DIM, NUM_FILTERS, k) for k in FILTER_SIZES])

        total = NUM_FILTERS * len(FILTER_SIZES)          #1024
        self.proj_char = nn.Linear(total, PROJ_DIM)      #wc
        self.proj_word = nn.Linear(total, PROJ_DIM)      #ww
        self.dropout = nn.Dropout(DROPOUT)

        dims = [2 * PROJ_DIM, *FC_LAYERS]                #1024 -> 512 -> 256 -> 128
        head = []
        for a, b in zip(dims[:-1], dims[1:]):
            head += [nn.Linear(a, b), nn.ReLU()]
        head += [nn.Linear(dims[-1], 2)]                 #2 logits
        self.head = nn.Sequential(*head)

        # Xavier init on dense layers (TextCNN uses xavier for the FC/projection)
        for m in [self.proj_char, self.proj_word, *[l for l in self.head if isinstance(l, nn.Linear)]]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _conv_branch(emb: torch.Tensor, convs: nn.ModuleList) -> torch.Tensor:
        x = emb.permute(0, 2, 1)                          # [B, emb, L]
        outs = []
        for conv in convs:
            c = F.relu(conv(x))                           # [B, nf, L-k+1]
            outs.append(F.adaptive_max_pool1d(c, 1).squeeze(-1))  # [B, nf]
        return torch.cat(outs, dim=1)                     # [B, nf*len(sizes)]

    def forward(self, x_char_seq, x_word, x_word_char):
        # Character-sequence branch
        char_feat = self.dropout(self._conv_branch(self.char_seq_emb(x_char_seq),
                                                    self.char_convs))       # [B,1024]
        # Word branch
        we = self.word_emb(x_word)                        # [B, W, emb]
        cw = self.ngram_emb(x_word_char).sum(dim=2)       # [B, W, emb] (sum over chars-in-word)
        word_feat = self.dropout(self._conv_branch(we + cw, self.word_convs))  # [B,1024]
        # Project each branch to 512, concatenate, classify
        combined = torch.cat([self.proj_word(word_feat), self.proj_char(char_feat)], dim=1)
        return self.head(combined)                        # [B,2]

def train(model: nn.Module, train_ds: URLNetDataset, output_dir, epochs: int) -> nn.Module:
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2_REG)
    loss_fn = nn.CrossEntropyLoss()
    history = []

    print(f"\nTraining URLNet (emb_mode 5) on {DEVICE}  ({epochs} epochs, no validation)")
    print("=" * 65)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss, correct, seen = 0.0, 0, 0
        for x_char, x_word, x_wc, y in loader:
            x_char, x_word, x_wc, y = (x_char.to(DEVICE), x_word.to(DEVICE),
                                       x_wc.to(DEVICE), y.to(DEVICE))
            optimizer.zero_grad()
            logits = model(x_char, x_word, x_wc)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * len(y)
            correct  += (logits.argmax(1) == y).sum().item()
            seen     += len(y)
        tr_loss, tr_acc = tot_loss / seen, correct / seen
        print(f"Epoch {epoch:02d}/{epochs}  train_loss={tr_loss:.4f}  "
              f"train_acc={tr_acc:.3f}  [{time.time() - t0:.1f}s]")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc})

    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    dicts_to_dir(output_dir)
    config = {
        "emb_mode": 5, "emb_dim": EMB_DIM, "filter_sizes": list(FILTER_SIZES),
        "num_filters": NUM_FILTERS, "fc_layers": list(FC_LAYERS),
        "max_len_chars": MAX_LEN_CHARS, "max_len_words": MAX_LEN_WORDS,
        "max_len_subwords": MAX_LEN_SUBWORDS, "min_word_freq": MIN_WORD_FREQ,
        "char_vocab": len(char_dict) + 1, "word_vocab": len(word_dict) + 2,
        "ngram_vocab": len(ngram_dict) + 1, "epochs": epochs,
        "lr": LR, "batch_size": BATCH_SIZE, "dropout": DROPOUT,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nModel + vocabularies saved -> {output_dir}")
    print("Evaluate separately on CleanedDATA/Test_* with load_model() + predict_batch().")
    return model

def load_model(checkpoint_dir) -> "URLNet":
    dicts_from_dir(checkpoint_dir)
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        cfg = json.load(f)
    model = URLNet(cfg["char_vocab"], cfg["word_vocab"], cfg["ngram_vocab"]).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pt"),
                                     map_location=DEVICE))
    model.eval()
    return model


def predict_batch(urls: list[str], model: "URLNet",
                  threshold: float = 0.5, batch_size: int = 256) -> list[dict]:
    model.eval()
    results = []
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        enc = [_encode(u) for u in batch]
        x_char = torch.as_tensor(np.stack([e[0] for e in enc]), device=DEVICE)
        x_word = torch.as_tensor(np.stack([e[1] for e in enc]), device=DEVICE)
        x_wc   = torch.as_tensor(np.stack([e[2] for e in enc]), device=DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(x_char, x_word, x_wc), dim=1)
        for url, p in zip(batch, probs):
            phish_p = p[1].item()
            results.append({
                "url": url, "phishing_prob": round(phish_p, 4),
                "benign_prob": round(p[0].item(), 4),
                "prediction": "PHISHING" if phish_p >= threshold else "BENIGN",
            })
    return results


def predict_url(url: str, model: "URLNet", threshold: float = 0.5) -> dict:
    return predict_batch([url], model, threshold)[0]


# MAIN

def main():
    parser = argparse.ArgumentParser(description="CleanURLNet — faithful URLNet (emb_mode 5), train-only")
    parser.add_argument("--mode",   choices=["train", "predict"], default="train")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--url",    type=str, help="URL to classify (--mode predict)")
    parser.add_argument("--ckpt",   type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--thresh", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 65)
    print("CleanURLNet  |  URLNet (Le et al. 2018)  |  full model, emb_mode 5")
    print("=" * 65)
    print(f"Device : {DEVICE}\nMode   : {args.mode}\n")

    if args.mode == "train":
        train_ds = build_train_dataset()
        model = URLNet(len(char_dict) + 1, len(word_dict) + 2, len(ngram_dict) + 1).to(DEVICE)
        print(f"Model parameters : {sum(p.numel() for p in model.parameters()):,}")
        train(model, train_ds, args.ckpt, args.epochs)

    elif args.mode == "predict":
        if not args.url:
            parser.error("--url is required for --mode predict")
        model = load_model(args.ckpt)
        r = predict_url(args.url, model, args.thresh)
        print(f"\nURL        : {r['url']}")
        print(f"Prediction : {r['prediction']}")
        print(f"Phish prob : {r['phishing_prob']:.4f}")
        print(f"Benign prob: {r['benign_prob']:.4f}")


if __name__ == "__main__":
    main()
