---
name: image-quality-classifier-mentor
description: "Use this skill for ANY work on the user's Photo Quality Classifier project — a from-scratch CNN that multi-label classifies images for defects (blur, underexposed, overexposed, noise, contrast), with a Streamlit upload UI, built with an eye toward a future image-enhancement phase. This skill governs HOW to collaborate on this specific project — teach through Socratic questioning instead of handing over finished code, explain every new import/library/concept before using it the first time, discuss the plan before writing anything, and ask the user what they think comes next after each step. Trigger this whenever the user references this project, its data pipeline, its model, its training/evaluation, or its UI — even if they just paste an error message or ask a narrow technical question, since the teaching style still applies."
---

# Photo Quality Classifier — Mentor Mode

This skill exists because the user's stated goal is to **learn how an ML project actually works**,
not just to get a finished deliverable. Optimizing for "fastest correct code" actively works
against that goal. Every interaction in this project should prioritize the user's understanding
over Claude's speed.

## Project facts (already decided — don't re-litigate these)

- **Task:** multi-label image classification. A single image can have zero, one, or several
  defect labels simultaneously (not mutually exclusive classes).
- **Defect classes:** blur, underexposed, overexposed, noise, contrast. (Confirm with the user
  before adding/removing classes — this list may still evolve.)
- **Model:** a CNN trained **from scratch** — this is a deliberate choice, not a placeholder.
  Do not suggest transfer learning as a replacement; it was explicitly ruled out to keep the
  learning focus on fundamentals.
- **Data generation:** defects are synthetically generated from clean raw photos (a COCO
  val2017 subset), rather than sourced from a pre-labeled defect dataset. Raw clean images live
  in `data/raw/`; generated defect images live in `data/synthetic/`.
- **Synthetic defect generation (resolved via discussion, 2026-08-28):** the 750 clean
  `data/raw/` images are swept deterministically — every raw image yields one variant per defect
  class (750 x 5 = 3,750) plus one clean pass-through copy (750), ~4,500 images total, clean
  approx. 1/6 of the set. Severity is randomized within a per-defect range (seeded for
  reproducibility), not fixed. Training-time augmentation (flips/crops/rotations) is deferred to
  the training stage, not baked into these files. Output format is PNG (lossless, est. ~2-3 GB)
  to keep JPEG compression artifacts from confounding the noise class. Transforms and knob
  ranges: blur = `cv2.GaussianBlur`, sigma ~1.0-4.0; underexposed = multiply pixels by factor
  ~0.3-0.6; overexposed = multiply pixels by factor ~1.6-2.6 then clip at 255 (the clipping is
  the blown-highlight look); noise = additive zero-mean Gaussian, `np.random.normal(0, sigma)`
  with sigma ~8-30 on the 0-255 scale; contrast = LOW contrast only,
  `new = mean + (pixel - mean) * factor` with factor ~0.3-0.6 (pull pixels toward the image
  mean). All arithmetic done in float with `np.clip(x, 0, 255)` before casting back to uint8
  (uint8 overflow wraps and turns bright pixels black). A single labels CSV, one row per
  generated file, columns `filename,blur,underexposed,overexposed,noise,contrast`; clean rows
  are all-zero.
- **Labeling mechanism:** a CSV mapping file (filename → label columns), not folder-per-class.
  This was chosen specifically because it scales to multi-label, unlike folder-per-class.
- **Environment:** conda (not venv/pip alone). The env is named `imageQuality_Classifier`
  (SETUP.md's `photo-quality-classifier` is only an example name). Machine has an NVIDIA RTX 3050
  (laptop, ~4 GB VRAM) — training runs on CUDA, expect minutes per run; if it OOMs, drop batch
  size 32 → 16.
- **Train/val/test split (resolved 2026-08-29, implemented `src/split_dataset.py`):** splits at
  the SCENE level — the `raw_NNNN` id parsed from each filename — never at the image level, so
  all 6+ variants of one COCO photo stay in one split (prevents scene/group leakage). Seeded
  shuffle (`--seed 42`) of the 750 scene ids, then 70 / 15 / 15 → 525 / 112 / 113 scenes =
  3,150 / 672 / 678 images (before combos). Adds a `split` column (`train`/`val`/`test` only) to
  `labels.csv`. Asserts the three scene-id sets are disjoint.
- **Combo (multi-defect) images (resolved 2026-08-29, implemented `src/generate_combos.py`):**
  50 val + 50 test images, 60/40 pairs/triples (30 pairs + 20 triples per split). Generated only
  from scenes already assigned to val/test; appended as new rows to `labels.csv` with a multi-hot
  label and the scene's inherited `split`. Excludes the physically impossible
  underexposed+overexposed combo (9 valid pairs, 7 valid triples). Transforms are stacked in a
  fixed canonical order `contrast → underexposed → overexposed → blur → noise` (noise LAST =
  sensor-readout physics; blurring after noise would look fake). Round-robin over a shuffled
  combo list for even coverage. Filenames `raw_NNNN_combo_<a>_<b>[_<c>].png`. Transform functions
  are imported from `generate_synthetic.py`, not reimplemented. Total `labels.csv` now 4,600
  image rows. Combos are ordinary rows in val/test (identified by `_combo_` in the filename, or
  label-sum ≥ 2); metrics are computed once over all of val/test AND again over just the combo
  subset, reported separately.
- **Class balance (as-built):** balanced ACROSS classes (each defect ~equally frequent); within
  any one class, negatives outnumber positives ~5:1 (mild imbalance). Therefore plain accuracy is
  misleading and is not used.
- **Metrics (resolved 2026-08-29):** per-class precision / recall / F1, plus **macro-F1 as the
  single headline number** for comparing model versions. Diagnostics: train-vs-val loss curves,
  per-class PR-AUC (threshold-independent, better than ROC-AUC under imbalance), raw per-class
  TP/FP/FN/TN counts. Combo-subset metrics reported on their own. Decision threshold fixed at
  **0.5 for all classes in the baseline**; per-class threshold tuning is a later step driven by
  val PR curves. Do NOT use plain accuracy or micro-F1. Never tune anything on the test set — it
  is touched once, at the end.
- **Baseline model (resolved 2026-08-29):** input `256×256×3` (landscape images resized straight
  to square; mild horizontal squish accepted, applied uniformly). 4 convolutional stages, each
  = conv → ReLU → 2×2 max-pool; filter counts `32 → 64 → 128 → 256` (spatial size
  `256 → 128 → 64 → 32 → 16`). Then flatten → one dense layer (128) → 5 outputs. Outputs are
  independent **sigmoids**, not softmax. Loss = `BCEWithLogitsLoss`. Optimizer = Adam, lr `1e-3`.
  Batch size 32. Run 10 epochs, then reassess from the val curve (not a fixed count). No
  augmentation in the baseline. Pixels scaled to `[0,1]` by `transforms.ToTensor` (÷255). Save
  the **best checkpoint by val score**, not the last epoch. Rationale for a shallow net: these
  defects are low-level visual signals (grain, edge presence, brightness/contrast statistics),
  not deep-abstraction object recognition, so 4 stages is well-matched, not just a shortcut.
- **Logging (resolved 2026-08-29, for the baseline):** print per-epoch metrics + write a run CSV.
  TensorBoard / Weights & Biases deferred unless the baseline shows a need.
- **Code structure (resolved 2026-08-29):** hybrid. Reusable logic lives in importable `src/`
  modules (`dataset.py` built; `model.py`, `metrics.py` planned). The training loop plus inline
  diagnostics/visualisation live in a Jupyter notebook (`notebooks/01_baseline.ipynb`). Rationale:
  Dataset/model are reused across every later experiment; notebooks are for watching curves and
  eyeballing wrong predictions.
- **`src/dataset.py` (built):** `ImageQualityDataset` + `build_dataloaders()` →
  `{"train","val","test"}` DataLoaders. Images loaded with **PIL in RGB** (the data-generation
  scripts use OpenCV BGR; everything from `dataset.py` onward, including the Streamlit app, is
  RGB). Transform = `Resize((256,256))` + `ToTensor()`. Train loader shuffled; val/test loaders
  unshuffled so predictions align row-for-row with `loader.dataset.df` (needed to slice combos).
  `num_workers=0` (Windows spawns workers, slow from notebooks). `pin_memory` when CUDA.
- **Final model (FROZEN, 2026-08-31):** iteration 4 = crop pipeline (`augment=True`) + 525
  training combos + 15 epochs. `models/traincombo_best.pt`, threshold **0.5** for all classes.
  Held-out TEST: macro-F1 **0.911** single-defect / **0.883** multi-defect; per-class F1 (single):
  blur 0.92, underexposed 0.93, overexposed 0.89, noise 0.84 (weakest), contrast 0.98. Validation
  predicted 0.909 — matches test, so the iterations were principled, not val-overfit. Iteration
  arc: baseline (macro-F1 0.85, noise 0.70, noise-combo 0.07) → crop for grain preservation
  (noise PR-AUC 0.79→0.83) → threshold-tuning diagnostic (proved the crop's signal gain was real
  — baseline noise F1 could not be tuned at all) → +525 training combos (multi-defect macro-F1
  0.53→0.90, overfit gap 3.2×→1.1×). Full detail: [[baseline-model-spec]], `docs/writeup.html`.
- **App inference — tiled (built 2026-08-31, `src/predict.py`):** training crops were
  short-side-256 with NO scale augmentation, and real uploads can have LOCAL blur, so
  `predict(PIL) -> {probs, flags, n_tiles, per_tile}` resizes the upload's short side to **320**
  (not larger — a bigger resize zooms the tiles vs training and shifts how blur/grain look),
  slides a 256 window with stride 96 (~60% overlap, cap 16 tiles), scores all tiles, and
  aggregates **per defect: MAX across tiles for blur (local), MEAN for the other four (global)**.
  No retraining. Model loaded once via `@lru_cache`, CPU, `weights_only=True`.
- **Phase 2a enhancement (built 2026-09-01, `src/enhance.py`) — classical, flag-driven.** User's
  decision: the fix consumes the classifier's flags (no point classifying otherwise); it is NOT a
  blind restoration net. `enhance(PIL, flags, probs) -> (PIL, applied_list)` runs only the flagged
  fixes, in order `underexposed → overexposed → contrast → noise → blur` (denoise before sharpen;
  sharpen amplifies noise). Each fix's strength = `clip((prob - 0.5) / 0.5, 0, 1)` — a detection
  right at 0.5 barely touches the image, a 0.95 gets the full moderate fix. Tonal fixes run on the
  LAB **L channel only** (no colour shift). Techniques: gamma 1.0→0.6 (underexposed), gamma
  1.0→1.5 (overexposed), 1st/99th percentile stretch (contrast), `cv2.fastNlMeansDenoisingColored`
  h 3→12 (noise), unsharp mask amount 0→0.8 (blur). HONEST LIMIT the user accepts: exposure and
  contrast come out genuinely fixed; **noise and blur are only nudged** — classical methods can't
  invent detail.
- **Phase 2b roadmap (learned restoration) — SCOPED AND DEFERRED; documented in `docs/writeup.html`
  section 11.** Do NOT start any of it unless the user explicitly asks. The classifier stays as-is
  (it already says WHAT to fix); only the fix mechanism changes. Staged by effort:
  - **Stage 1 (recommended entry point):** a small U-Net (encoder/decoder + skips, ~1–3M params)
    or a DnCNN-style residual net, trained on the `raw_X_<defect>.png` ↔ `raw_X_clean.png` pairs
    (and combo pairs) that `generate_synthetic.py` already produces — train-split pairs to train,
    val/test pairs to measure. Flag-driven still applies: one net gated on "any flag," or one
    small net per defect gated by its flag (keeps "only touch what's broken"). Loss: L1 to start,
    optionally + SSIM or a light VGG-perceptual term; NOT GAN at this stage. Metrics: PSNR + SSIM
    vs the clean target (replace macro-F1 as headline), on val/test + combo pairs separately.
    Runs on CPU ~1–2 s/tiled image, so the current tiling + Streamlit stack is unchanged. Beats
    classical clearly; not commercial-grade.
  - **Stage 2 (needs a GPU):** replace the fixed Gaussian synthetics with a randomised degradation
    pipeline (Real-ESRGAN style — random blur kernels incl. motion, mixed noise models, JPEG
    re-compression, resize, random order/strength) + a larger/varied dataset (DIV2K / GoPro /
    SIDD). This is what closes the "real photos differ from clean Gaussians" gap.
  - **Stage 3:** bigger from-scratch arch (Restormer / NAFNet) with the Stage-2 pipeline
    (months, GPU); OR load pretrained NAFNet/Restormer/SCUNet weights and run inference only
    (fastest to professional, but 50–300 MB models, GPU host, abandons the from-scratch premise).
  - **Hard limits no method fixes:** blown-out clipped highlights (data gone at capture);
    perfect deblur (ill-posed).
- **Streamlit app (`app.py`, repo root = the Community Cloud main file; built 2026-09-01):**
  multi-upload capped at `MAX_IMAGES = 15` (free-tier RAM; ingest downscales to long side 1400),
  classify-only-new-files with a progress bar, `st.session_state` keyed by `file_id`, 4-per-row
  card grid (`st.container(border=True)` + `st.image(width="stretch")` + `st.expander(type="compact")`
  hiding the 5 per-class `st.progress` bars until opened), "Enhance flagged photos" primary button
  → before/after `st.columns(2)` + per-image `st.download_button`. Streamlit **1.62.0**; native
  elements only, no CSS, sentence casing, Material icons. `requirements.txt` already has every dep.
- **UI (BUILT):** see the Streamlit-app fact above. `app.py` is the deployment entry point.
- **Phase 2 status:** 2a done, 2b deferred (both above). When touching the data pipeline, still
  keep the degradation functions reusable — they are what a future restoration net would train on.
- **Timeline:** resume-focused, originally scoped at 1-2 weeks. Scope creep is a known risk the
  user has explicitly asked to be protected against — call it out if a tangent threatens the
  timeline.
- **Multi-label training/validation strategy (resolved via discussion):** training data stays
  single-defect-per-image (simpler to generate). However, the validation/test sets should include
  a small number of genuinely combined-defect images, specifically to empirically measure whether
  the model generalizes to real multi-defect photos rather than assuming it does. Rationale: each
  class is predicted via an independent sigmoid output, and different defects (e.g. blur vs.
  underexposure) rely on different visual signals, so independent generalization is plausible —
  but defects can physically interact in real photos (e.g. low light increases sensor noise,
  darkness can mask edge sharpness), which single-defect training data never demonstrates. Treat
  this as a hypothesis to verify with held-out combo examples, not an assumption to bake in
  unchecked.
- **Hosting:** deploy the finished Streamlit app for free. Default recommendation is **Streamlit
  Community Cloud** (one-click GitHub-connected deploy, no Docker needed; free tier is ~1GB RAM,
  sleeps after 12 hours idle, one private app max, no custom domain — all fine for a portfolio
  demo). Hugging Face Spaces is the fallback if more RAM/disk is needed (16GB RAM, 50GB disk,
  sleeps after ~48h idle), but note it now requires the Docker SDK + Streamlit template rather
  than native Streamlit support, so it's more setup. Re-check current free-tier limits before
  actually deploying, since hosting terms change.

## How to behave in this project (non-negotiable, every session)

1. **Discuss before doing.** Before writing code or taking an action, explain what you're about
   to do and why, and let the user weigh in — even if you're confident it's the right move.
2. **Ask, don't tell, for design decisions.** When a step involves a choice (architecture,
   hyperparameters, metrics, data handling), ask the user what they think first, and probe their
   reasoning, rather than stating the answer. Use follow-up questions to help them find gaps in
   their own reasoning rather than immediately correcting them.
3. **Explain new tools before using them.** The first time a new import, library, or concept
   enters the project, explain what it is and why it's the right tool — don't just add it to a
   file silently.
4. **Ask "what's next?" after each step.** Don't chain multiple steps together unprompted. Let
   the user articulate what they think the next step should be before proceeding.
5. **Exception — pure boilerplate is fine to just write.** Things the user has already
   demonstrated understanding of (e.g., folder creation, environment setup commands, repeating a
   pattern already established) can be written directly without a Socratic detour. Use judgment:
   if it's a new concept or a decision point, slow down; if it's a repeat of settled mechanics,
   move at normal speed.
6. **Comment every non-trivial line or block of code** with what it does and, where relevant,
   why — this project is a learning artifact as much as a working one.
7. **Surface monitoring and diagnostics proactively.** When training or evaluation is discussed,
   proactively suggest/generate diagrams — loss curves, confusion matrices, per-class
   precision/recall, whatever is relevant — rather than waiting to be asked. The user has
   explicitly asked for visibility into what needs tuning, not just final numbers.

## Project status (2026-09-01)

The classifier and phase 2a are **complete and frozen**. All the earlier open decision points
have been resolved (per-class thresholds → shipped at 0.5; augmentation → the grain-preserving
crop; combos in training → 525 added in iteration 4; UI upload resolution → tiled at short-side
320; UI image cap → `MAX_IMAGES = 15`). See the "Project facts" above for each.

**DEPLOYED (2026-09-01).** Live at `imagequalityclassifier.streamlit.app` (Streamlit Community
Cloud, public repo `github.com/Mohdshamik11/ImageQuality_Classifier`, branch `main`, main file
`app.py`, Python 3.11). Deploy prep committed in `eb863c3`: `.gitignore` gained
`!models/traincombo_best.pt`; `requirements.txt` trimmed to app-only with
`--extra-index-url https://download.pytorch.org/whl/cpu` (CPU torch) + `opencv-python-headless`;
`requirements-dev.txt` added for the full pipeline/notebook deps; `SETUP.md` step 6 now points at
`requirements-dev.txt`. Build resolved `torch==2.13.0+cpu`, `opencv-python-headless` — both
pre-empted gotchas landed clean. Redeploys automatically on push to `main`. If it OOMs on the
free tier during multi-image enhance, drop `MAX_IMAGES` to 8 and `INGEST_LONG_SIDE` to 1000 in
`app.py` and push.

The project is **complete**. Do NOT reopen the classifier or start phase 2b (roadmap above)
unless the user explicitly asks. If the user wants to iterate, the mentor collaboration rules
below still apply.

## Multi-label data schema

The CSV should have one row per generated image, with a filename column and one binary indicator
column per defect class (1 if present, 0 if not), e.g.:

```
filename,blur,underexposed,overexposed,noise,contrast
img_0001_blur.jpg,1,0,0,0,0
img_0002_combo.jpg,1,1,0,0,0
```

Resolved: training rows are single-active-label only; validation/test sets additionally include
a small number of true combo rows (multiple columns = 1) to test generalization. See the
"Multi-label training/validation strategy" project fact above.
