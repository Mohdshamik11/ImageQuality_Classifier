# Photo Quality Classifier

A convolutional neural network, **trained from scratch**, that inspects a photograph and flags
five kinds of quality defect at once — **blur, underexposure, overexposure, sensor noise, and
low contrast**, then runs a full restoration on the whole image.

**Live demo:** https://imagequalityclassifier.streamlit.app
**Full build log:** [`docs/writeup.html`](docs/writeup.html) — data pipeline, the classifier's
four training iterations, the learned restoration model, metrics, and the reasoning behind every decision.

---

## What it does

- **Multi-label classification.** A photo can be blurry *and* underexposed *and* noisy at once,
  so the model has five independent yes/no outputs, not one "pick a class."
- **Tiled inference.** Uploads are scanned by sliding a 256-pixel window across the whole frame,
  so a defect anywhere in the image is caught (blur is often local; exposure/noise are global).
- **Learned restoration (phase 2b).** The "Enhance" button runs a blind ~4.3M-parameter residual
  U-Net trained on a realistic on-the-fly degradation pipeline. It denoises, corrects exposure and
  contrast, and improves mild-to-moderate blur. Strong motion blur is the hard case. The classical
  per-flag fixes (phase 2a) remain as a fallback.

### Results (held-out test set, threshold 0.5)

| | macro-F1 | blur | underexposed | overexposed | noise | contrast |
|---|---|---|---|---|---|---|
| single-defect | **0.911** | 0.92 | 0.93 | 0.89 | 0.84 | 0.98 |
| multi-defect | **0.883** | 0.83 | 0.97 | 0.82 | 0.86 | 0.93 |

Started at macro-F1 0.85 (single) / 0.53 (multi-defect); four diagnosis-driven iterations —
grain-preserving crop, threshold tuning, stacked-defect training data — closed the gap.

---

## How it works

There is no public dataset of photos labelled "blurry" or "underexposed," so the data is
**synthesised**: 750 clean COCO photos, each degraded five ways with controlled image maths,
giving 4,500 single-defect images plus a matched clean copy of every scene. Validation and test
also get 100 genuinely *multi-defect* images as a held-out probe.

```
CLASSIFIER (phase 1)
  download_raw_images.py → data/raw/       BaselineCNN (src/model.py), 4 conv stages
  generate_synthetic.py → data/synthetic/  trained in notebooks/ 01→04
  split_dataset.py / generate_combos.py    → models/traincombo_best.pt

RESTORATION (phase 2b)
  degrade.py          realistic degradation, applied on the fly (no files saved)
  restore_dataset.py  (clean crop → degrade → pair) DataLoaders
  restore_model.py    ~4.3M-param residual U-Net
  train_restore.py    training loop (run on Kaggle T4) → models/restore_best_lpips.pt

APP
  predict.py       tiled classifier inference
  restore_infer.py tiled restoration inference
  enhance.py       learned restoration, classical fallback
  app.py           Streamlit UI
```

---

## Running it

**Requires:** [conda](https://docs.conda.io/en/latest/miniconda.html). See [`SETUP.md`](SETUP.md)
for the full first-time setup.

```bash
git clone https://github.com/Mohdshamik11/ImageQuality_Classifier.git
cd ImageQuality_Classifier

conda create -n imageQuality_Classifier python=3.11
conda activate imageQuality_Classifier
pip install -r requirements-dev.txt        # full env; requirements.txt alone is app-only
```

### The app

Both trained models (`models/traincombo_best.pt`, `models/restore_best_lpips.pt`) ship with the
repo, so the app runs immediately:

```bash
streamlit run app.py
```

Upload up to 15 photos → each is classified and shown as a card (click to see the five scores) →
"Enhance" restores the flagged ones and shows the before/after.

### Reproduce — classifier

```bash
python src/download_raw_images.py --num-images 750    # ~1 GB download from COCO
python src/generate_synthetic.py                      # 4,500 images, ~2 GB
python src/split_dataset.py                           # scene-level 70/15/15 split
python src/generate_combos.py                         # 50 val + 50 test multi-defect probes
python src/generate_combos.py --n-train 525           # add training combos (iteration 4)
```

Then run `notebooks/01_baseline.ipynb` → `04_train_combos.ipynb` in order. Seeded (`42`).

### Reproduce — restoration model

```bash
python src/download_raw_images.py --num-images 4000 --output-dir data/clean_pool/coco
python src/download_div2k.py                          # +800 hi-res images
python src/shrink_pool.py                             # → data/clean_pool_small/ (≤800 px)
python src/train_restore.py                           # local; or notebooks/kaggle_train.ipynb on a Kaggle GPU
```

`train_restore.py` generates the degraded inputs on the fly (`src/degrade.py`) — no paired
dataset is stored. Config (epochs, loss weights, model size) is the block at the top of the file.
Evaluate on real photos with `python src/eval_restore.py` (drop photos in `data/real_test/`).

---

## Project structure

```
app.py                     Streamlit UI (deployment entry point)
src/
  download_raw_images.py    COCO subset → data/raw/ (also the restoration clean pool)
  download_div2k.py         DIV2K hi-res images for the clean pool
  generate_synthetic.py     the five degradations → data/synthetic/ + labels.csv
  split_dataset.py          scene-level train/val/test split
  generate_combos.py        multi-defect images (val/test probes; --n-train for training)
  dataset.py                classifier Dataset + DataLoaders (augment=True = crop pipeline)
  model.py                  BaselineCNN — 4 conv stages + a small head
  metrics.py                per-class precision/recall/F1, macro-F1, PR-AUC, threshold sweep
  predict.py                tiled classifier inference: PIL image → per-defect probabilities
  degrade.py                realistic on-the-fly degradation pipeline
  shrink_pool.py            resize the clean pool to ≤800 px
  restore_dataset.py        (clean crop → degrade → pair) DataLoaders + sealed test split
  restore_model.py          the ~4.3M-param residual U-Net
  train_restore.py          restoration training loop (config block at top)
  restore_infer.py          tiled restoration inference
  eval_restore.py           real-photo eval: before/after + no-ref metrics + classifier re-check
  enhance.py                learned restoration, with the classical per-flag fixes as fallback
notebooks/
  01_baseline … 04_train_combos   the classifier's four iterations
  kaggle_train.ipynb              thin wrapper to run train_restore.py on a Kaggle GPU
docs/writeup.html                 full build log with charts
models/traincombo_best.pt         the frozen classifier
models/restore_best_lpips.pt      the restoration model
```

---

## Results — restoration model

On a 29-photo real-world test set (`src/eval_restore.py`):

- **BRISQUE 15.9 → 10.6** (−33%, lower is better) · **MUSIQ 63.2 → 65.6** (up on 27 of 29)
- Denoising and exposure/contrast correction are genuinely useful; overexposure flags drop.
- **Mild-to-moderate blur** improves noticeably.
- **Strong motion blur barely moves** — see below.

## Known limits & what's next

- **Strong motion blur.** L1 / SSIM / perceptual are regression losses: when a blurry patch could
  have come from many sharp patches, the loss-minimising output is their (soft) average. More
  epochs or channels don't change that ceiling. The fix is an **adversarial (GAN) loss** — a
  discriminator that penalises "looks blurry" directly — plus rebalancing `degrade.py` toward the
  recoverable blur range. That's the next training iteration.
- **Blown-out highlights** are unrecoverable (detail was clipped at capture).
- **Portrait bokeh.** The model can't tell intentional shallow depth-of-field from a blur defect —
  training pairs are always uniformly blurred → uniformly sharp, and tiled inference sees each
  256×256 window alone with no view of a sharp subject elsewhere. It may over-sharpen tasteful
  bokeh backgrounds. A fix needs a whole-frame-aware model; accepted, not addressed.
- **Output is capped at 768 px** long side for CPU speed on the free host.

---

## Notes

- Built as a learning project — the emphasis was on understanding each step (why a scene-level
  split, why PR-AUC as a diagnostic, why a crop instead of a resize), not just the final number.
  `docs/writeup.html` walks through the reasoning.
- Deployed on Streamlit Community Cloud's free tier (CPU, ~1 GB RAM); the 15-image cap and model
  sizes are chosen for it.
