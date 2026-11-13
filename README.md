# Adversarial GAN/DRL training for phishing-URL detection

A controlled study of whether adversarial training improves a phishing-URL detector.
A URLNet character/word CNN detector is compared across three training regimes:

1. **Static** - trained on real URLs only (the baseline).
2. **Augmented** - trained on real URLs plus synthetic phishing URLs from a frozen
   fine-tuned GPT-2 generator.
3. **Adversarial** - trained in a co-evolutionary loop against a GPT-2 generator that
   adapts via PPO, with a BERT discriminator supplying a realism signal.

The repository also contains a parallel detector track based on a PPO reinforcement-learning
agent (`drl_defer_*`, `adversarial_loop_drl.py`).

## Summary of findings

Under a leakage-free evaluation (train/validation/test split by registrable domain), the
three regimes are effectively indistinguishable on threshold-free metrics: all reach about
0.99 ROC-AUC on real phishing and all perform poorly on homoglyph attacks (about 0.62).
Two diagnostics explain earlier, more optimistic-looking results:

- The apparent "benign-accuracy collapse" was **host-level data leakage** in random
  URL-level splits (validation and training shared most of their domains), which distorted
  threshold calibration rather than the model's ranking ability.
- The generator's apparent evasion success was **reward hacking and mode collapse**: it
  produced benign-looking, low-diversity strings rather than realistic evasive phishing, so
  a high "fool rate" did not indicate a strong attacker.

## Repository layout

```
.
├── CleanedDATA/                 CSV datasets (url,label), label 1 = phishing, 0 = benign
├── CLeanURLNet.py               URLNet CNN detector: train / predict
├── evaluate_cnn.py              Evaluate a trained URLNet on the held-out sets
├── finetune_gpt2generator.py    Fine-tune GPT-2 on phishing URLs; emit a synthetic pool
├── warmup_discriminator.py      Warm-start the BERT discriminator (real vs generated)
├── gan_gpt2_bert_train.py       Shared helpers: generation, BERT scoring, URL validation
├── gan_gpt2_bert_ppo.py         Standalone GAN: GPT-2 generator vs BERT discriminator (PPO)
├── adversarial_loop.py          Co-evolutionary loop: generator vs DRL detector
├── adversarial_loop_drl.py      DRL-detector variant of the loop
├── adversarial_loop_urlnet.py   URLNet-detector variant of the loop (the main experiment)
├── drl_defer_v1.py              PPO detector with a defer action (seed 44)
├── drl_defer_v2.py              PPO detector variant (seed 45)
└── archive/                     Earlier architecture, kept for reference only (see note below)
```

## Datasets (`CleanedDATA/`)

Each file is a CSV with `url` and `label` columns.

| File | Label | Contents |
|---|---|---|
| `Train_2025.csv`   | 1 | Real phishing URLs, training split |
| `Train_2026.csv`   | 1 | Real phishing URLs, later-period training split |
| `Train_Benign.csv` | 0 | Benign URLs, training split |
| `Test_2025.csv`    | 1 | Real phishing URLs, held-out test (in-period) |
| `Test_2026.csv`    | 1 | Real phishing URLs, held-out test (later period) |
| `Test_Benign.csv`  | 0 | Benign URLs, held-out test |
| `homoglyph_from_tranco.csv` | 1 | Homoglyph/lookalike URLs derived from Tranco domains |

Notes:
- URLs are used only as text strings. Nothing in this code resolves, visits, or registers
  any URL. The test sets are used for evaluation only and never for training or model
  selection.
- The corpus is drawn from public phishing feeds and the Tranco top-sites list and includes
  URLs across all content categories as they were collected.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended for the generator and loop steps. See `requirements.txt`
for the PyTorch/CUDA note. The scripts auto-detect the device.

## Running the pipeline

All scripts read `CleanedDATA/` by default and write their outputs next to themselves.
Trained models and run outputs are git-ignored (see `.gitignore`); regenerate them with the
steps below.

**1. Train the static URLNet baseline**
```bash
python CLeanURLNet.py --mode train        # writes ./CleanURLNET/
```

**2. Evaluate the detector on the held-out sets**
```bash
python evaluate_cnn.py                     # evaluates ./CleanURLNET/ by default
```

**3. Fine-tune the GPT-2 generator** (for the augmented and adversarial regimes)
```bash
python finetune_gpt2generator.py           # writes ./PreTrainedGenerator/ (+ generated_urls.csv)
```

**4. Warm-start the BERT discriminator**
```bash
python warmup_discriminator.py             # writes ./WarmedDiscriminator/
```

**5. Adversarial hardening loop (URLNet detector)** - the main experiment
```bash
python adversarial_loop_urlnet.py          # needs PreTrainedGenerator, WarmedDiscriminator, CleanURLNET
```

**Optional - standalone GAN (generator vs discriminator)**
```bash
python gan_gpt2_bert_ppo.py --steps 1000
```

**Optional - DRL detector track**
```bash
python drl_defer_v1.py --mode train        # writes ./DRL_defer_v1_seed44/
python adversarial_loop.py                 # co-evolutionary loop vs the DRL detector
```
The DRL loop expects a trained detector checkpoint; adjust the path in the `Config`
dataclass at the top of `adversarial_loop.py` if your checkpoint is stored elsewhere.

Most scripts accept `--help` for the full set of options (seeds, epochs, batch sizes, etc.).

## `archive/`

The two scripts in `archive/` are an earlier architecture (a multistep DRL detector with a
char+word URLNet feature extractor and a TRL-based generator loop). They are kept for
reference and are **not** part of the runnable pipeline above: they expect externally trained
models and the earlier 70/30 data split, which are not included here. The paths in their
`Config` blocks are placeholders to be edited before use.
