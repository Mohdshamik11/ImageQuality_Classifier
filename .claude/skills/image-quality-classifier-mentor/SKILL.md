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
- **UI:** Streamlit app where a user uploads an image and gets defect predictions back.
- **Forward compatibility:** the project has a planned phase 2 (image enhancement/restoration
  per detected defect). Whenever making architecture or pipeline decisions now, consider whether
  the choice would make it harder to bolt on an enhancement stage later — flag it if so, but
  don't build phase 2 features now.
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

## Open decision points — work these out WITH the user, don't pre-answer

These are questions the user has raised but explicitly wants to reason through rather than be
told the answer to. Do not resolve them unilaterally in code or in explanations — treat each as
a live discussion to have when the project reaches that stage:

- **Per-class decision thresholds** — after the baseline runs, use the validation PR curves to
  decide whether any class wants a threshold other than 0.5. Discuss, don't auto-pick.
- **Training-time augmentation** — deferred from the data stage. Revisit after seeing the
  baseline's train-vs-val gap; the diagnostics decide whether it's needed and which augmentations.
- **Whether to add combo images to the *training* set** — only if the baseline's combo-subset
  metrics show single-defect training doesn't generalize. Evidence first, then decide together.
- **What resolution to standardize *uploaded* images to in the UI** — training is fixed at
  256×256, so the model needs 256×256 input, but the UI's downscaling/latency tradeoff for
  large user uploads is still an open discussion.
- **Whether/how to cap the number of images the UI processes at once.**

Already resolved (see "Project facts" above, do NOT reopen without the user asking): number of
conv layers (4), baseline architecture, multi-label metrics, train/val/test split ratios &
methodology, logging approach for the baseline, code structure.

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
