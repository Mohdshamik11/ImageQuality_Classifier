# Photo Quality Classifier

A convolutional neural network, **trained from scratch**, that inspects a photograph and flags
five kinds of quality defect at once — **blur, underexposure, overexposure, sensor noise, and
low contrast** — then runs a flag-driven repair on the ones it flags.

**Live demo:** https://imagequalityclassifier.streamlit.app
**Full build log:** [`docs/writeup.html`](docs/writeup.html) — data pipeline, four training
iterations, metrics, and the reasoning behind every decision.

---

## What it does

- **Multi-label classification.** A photo can be blurry *and* underexposed *and* noisy at once,
  so the model has five independent yes/no outputs, not one "pick a class."
- **Tiled inference.** Uploads are scanned by sliding a 256-pixel window across the whole frame,
  so a defect anywhere in the image is caught (blur is often local; exposure/noise are global).
- **Flag-driven enhancement.** An "Enhance" button applies only the fixes the classifier flagged
  — gamma correction for exposure, histogram stretch for contrast, non-local means for noise,
  unsharp mask for blur. The fix strength scales with the model's confidence.

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
data pipeline            model                        app
─────────────            ─────                        ───
download_raw_images.py   BaselineCNN (src/model.py)   predict.py  — tiled inference
  → data/raw/            4 conv stages, 8.8M params    enhance.py  — 5 classical fixes
generate_synthetic.py    trained in notebooks/        app.py      — Streamlit UI
  → data/synthetic/        01_baseline → 04_train_combos
split_dataset.py         → models/traincombo_best.pt
generate_combos.py
  → labels.csv
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

The trained model (`models/traincombo_best.pt`) ships with the repo, so the app runs immediately:

```bash
streamlit run app.py
```

Upload up to 15 photos → each is classified and shown as a card (click to see the five scores) →
"Enhance" repairs the flagged ones and shows the before/after.

### Reproduce the pipeline and training

```bash
python src/download_raw_images.py --num-images 750    # ~1 GB download from COCO
python src/generate_synthetic.py                      # 4,500 images, ~2 GB
python src/split_dataset.py                           # scene-level 70/15/15 split
python src/generate_combos.py                         # 50 val + 50 test multi-defect probes
python src/generate_combos.py --n-train 525           # add training combos (iteration 4)
```

Then run the notebooks in order — `notebooks/01_baseline.ipynb` through
`04_train_combos.ipynb`. Each writes a checkpoint and a per-epoch history to `models/`.
Everything is seeded (`42`).

---

## Project structure

```
app.py                     Streamlit UI (deployment entry point)
src/
  download_raw_images.py    COCO subset → data/raw/
  generate_synthetic.py     the five degradations → data/synthetic/ + labels.csv
  split_dataset.py          scene-level train/val/test split
  generate_combos.py        multi-defect images (val/test probes; --n-train for training)
  dataset.py                PyTorch Dataset + DataLoaders (augment=True = crop pipeline)
  model.py                  BaselineCNN — 4 conv stages + a small head
  metrics.py                per-class precision/recall/F1, macro-F1, PR-AUC, threshold sweep
  predict.py                tiled inference: PIL image → per-defect probabilities
  enhance.py                the five classical fixes + a flag-driven orchestrator
notebooks/                  01–04, one per training iteration
docs/writeup.html           full build log with charts
models/traincombo_best.pt   the frozen model the app uses
```

---

## Future plans — reliable enhancement

Phase 2a (the current "Enhance" button) is **classical**, and classical repair of noise and blur
has a hard ceiling: it cannot invent detail. Exposure and contrast come out genuinely fixed;
noise and blur are only nudged. The staged plan to fix that:

- **Stage 1 — a small restoration network.** A U-Net or DnCNN-style residual net trained on the
  *(degraded → clean)* pairs `generate_synthetic.py` already produces. L1 (+ optional SSIM /
  perceptual) loss, PSNR / SSIM as metrics, still flag-driven. Runs on CPU — same deployment.
  Clearly better than unsharp mask; not commercial-grade.
- **Stage 2 — realistic degradations.** Replace the clean Gaussian synthetics with a randomised
  pipeline (motion blur, mixed noise models, JPEG re-compression) plus a larger dataset. This is
  what makes it generalise to real photos. Needs a GPU.
- **Stage 3 — bigger models.** A modern architecture (Restormer, NAFNet) from scratch, or load
  pretrained weights and run inference only.

**Hard limits no method fixes:** blown-out highlights (the data was clipped at capture), and
perfect deblurring (it's an ill-posed problem).

---

## Notes

- Built as a learning project — the emphasis was on understanding each step (why a scene-level
  split, why PR-AUC as a diagnostic, why a crop instead of a resize), not just the final number.
  `docs/writeup.html` walks through the reasoning.
- Deployed on Streamlit Community Cloud's free tier (CPU, ~1 GB RAM); the 15-image cap and model
  sizes are chosen for it.
