---
name: image-quality-classifier-mentor
description: "Use this skill for ANY work on the user's Photo Quality Classifier project — a from-scratch CNN that multi-label classifies images for defects (blur, underexposed, overexposed, noise, contrast), a learned restoration U-Net that repairs flagged photos, and a Streamlit upload UI. This skill governs HOW to collaborate on this specific project — teach through Socratic questioning instead of handing over finished code, explain every new import/library/concept before using it the first time, discuss the plan before writing anything, and ask the user what they think comes next after each step. Trigger this whenever the user references this project, its data pipeline, its models, its training/evaluation, or its UI — even if they just paste an error message or ask a narrow technical question, since the teaching style still applies."
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
- **Enhancement — current design (2026-09-12), one unified path, no mode toggle.**
  `enhance(image, flags, probs, strength=1.0)` in `src/enhance.py`. Every fix's strength scales
  DIRECTLY off the classifier's raw per-defect probability (`strength_from_prob(p) = clip(p,0,1)`)
  — NOT gated by the 50% "flagged" threshold, and NOT a mode you toggle. A photo just under
  threshold still gets a proportionally mild fix; a genuinely clean photo (probabilities near 0)
  comes back untouched because there's nothing to scale up. Fixes below `MIN_STRENGTH` (0.02) are
  skipped outright. Order: tonal fixes (underexposed → overexposed → contrast, gamma-curve based,
  data-driven — see below) → blur (the GAN, see next fact) → noise (`cv2.fastNlMeansDenoisingColored`,
  LAST, since it must not run before the blur-GAN — denoising first would smooth away texture the
  GAN needs to sharpen; the classical-unsharp-mask fallback wants the OPPOSITE order, denoise
  before sharpen, since sharpening amplifies noise).
  - `fix_overexposed`/`fix_underexposed` measure severity from the image's own 75th/25th
    percentile (NOT 95th/5th — that was tried first and was a real bug: gamma applies to the
    WHOLE luminance channel at once, so using an extreme percentile as the severity signal let a
    handful of already-blown highlight pixels demand a huge gamma that crushed every midtone in
    the photo; verified with real numbers — a "mild" ×1.05 brightening collapsed median brightness
    176→95 before the fix). 75th/25th percentile tracks overall brightness, which is what
    "overexposed"/"underexposed" actually mean. Gamma bound clamped to [0.5, 1.8]. Hard limit:
    gamma leaves 0 and 255 exactly unchanged for any exponent (`(255/255)^gamma = 1` always) —
    truly clipped pixels are unrecoverable by any gamma curve, classical or learned.
  - Real-ESRGAN and the from-scratch phase-2b U-Net are BOTH fully removed from this path — see
    the next fact for why, and for what replaced them.
- **Blur-specialist GAN (`src/restore_blur_gan.py`, shipped 2026-09-12) — what replaced Real-ESRGAN
  and the from-scratch U-Net for blur.** Same `RestoreUNet` architecture
  (`base_channels=48, n_blocks=3`, ~4.3M params) as the original phase-2b U-Net below, but
  continued training with an ADVERSARIAL (GAN) loss instead of stopping at pure regression loss —
  this is exactly "the one real open item" the earlier phase-2b section below used to call out as
  not-yet-started; it's now built, trained, and shipped.
  - **Discriminator (`src/discriminator.py`):** `UNetDiscriminatorSN`, Real-ESRGAN's own design —
    a small U-Net with spectral norm on every conv except first/last, outputs a per-pixel
    real/fake logit map. Training-only scaffolding; NOT part of the deployed app, thrown away
    after training (same as Real-ESRGAN's own release ships only its generator).
  - **Stage 1 (`src/train_restore_gan.py`):** warm-started from the phase-2b checkpoint
    (`restore_best_lpips.pt`), fine-tuned on COCO images with `degrade.py`'s new
    `degrade_blur_only()` (blur-only synthetic degradation — exposure/contrast/noise are already
    handled elsewhere in the pipeline, so this stage's whole job is stopping the blurry-average
    output specifically for blur). Loss `1.0·L1 + 0.05·perceptual + 0.05·adversarial`, no SSIM
    (fights sharpness). Run on a rented RunPod RTX 4090 (Secure Cloud — Community Cloud hit a
    real, repeatable CUDA-runtime-broken-host issue, not fixable from inside the container). 25
    epochs, ~42 min, best val PSNR 24.98 / LPIPS 0.253. Checkpoint: `restore_blur_gan_lpips.pt`.
  - **Stage 2 (`src/realblur_dataset.py` + `src/train_realblur_gan.py`) — real-data fine-tune,
    the part that actually closed the gap.** RealBlur-J (CC BY 4.0, `rimchang/RealBlur` on
    GitHub): the same scene shot through a beam-splitter rig simultaneously at long exposure
    (real camera-shake blur) and short exposure (sharp), pre-aligned via ECC + intensity
    correction, so a same-coordinates crop from both sides stays aligned. Warm-started from the
    stage-1 checkpoint (not from scratch — same architecture/loss/optimizer settings as stage 1,
    only the data source changed, deliberately, to avoid confounding what caused any improvement).
    Run on a rented RunPod RTX PRO 6000 (Secure Cloud): first 15 epochs (PSNR 27.90/LPIPS 0.155),
    then extended 15 MORE epochs from that checkpoint (PSNR 28.11/LPIPS 0.148 — diminishing
    returns by the end, PSNR/SSIM had plateaued by epoch ~5 both times, LPIPS kept improving
    longer). Shipped checkpoint: `restore_blur_gan_real_ext_lpips.pt` (whitelisted in
    `.gitignore`). Confirmed genuinely better by direct before/after comparison on the user's own
    real test photos, not just the metrics.
  - **Resolution caveat (real bug, fixed 2026-09-12):** the model only processes at ≤768px long
    side internally (matches its training crop scale). It USED to just return the image at that
    shrunk size — any upload over 768px silently came back smaller, which briefly looked like "the
    model just doesn't work well" before being root-caused. Now scales its output back up to the
    input's exact original size before returning — but the actual deblurring computation still
    only ever saw the downscaled proxy, so large photos still end up slightly softer than the
    model's true capability, just no longer smaller.
- **From-scratch phase-2b U-Net (BUILT, evaluated, NOT shipped, kept as reference) —
  `src/restore_model.py` / `restore_infer.py`, weights `models/restore_best_lpips.pt`.** This is
  the base model the blur-GAN above warm-started from. On real photos it corrects exposure but
  **softens detail + flattens contrast** (regression-loss artifact — L1/SSIM/perceptual pick the
  average of all plausible sharp patches, and the average of many plausible sharp images is a
  blurry image). No-ref eval metrics (BRISQUE/MUSIQ) missed it because they reward smoothness; the
  user (a photographer) caught it by eye. This is WHY the adversarial-loss stage above was built.
  - **Training (`src/train_restore.py`, run on a Kaggle T4 via `notebooks/kaggle_train.ipynb`):**
    target = real clean image; input generated ON THE FLY by `src/degrade.py` (Real-ESRGAN-style:
    Gaussian/motion/defocus/anisotropic blur, Poisson-Gaussian noise, JPEG + resize artifacts,
    tone shifts; 1–3 per image, random order/strength). Loss `L1 + 0.1·(1−SSIM) + 0.05·VGG-perceptual`.
    70 epochs, Adam 1e-4 + LinearLR warmup + grad-clip 1.0 (three earlier runs diverged before
    those stabilisers; AMP was removed — SSIM unstable under fp16). Clean pool = COCO 4000 +
    DIV2K 800, resized ≤800 px (`src/shrink_pool.py` → `data/clean_pool_small/`).
  - **Result (29 real photos, `src/eval_restore.py`):** BRISQUE 15.9→10.6 (−33%), MUSIQ 63.2→65.6
    (up on 27/29), exposure flags down, mild/moderate blur improved. **Strong motion blur barely
    moves** — regression loss produces soft output on ill-posed deblur. Some light over-smoothing
    (NIQE occasionally worsens).
  - **Hard limits (still apply to the shipped blur-GAN too):** blown clipped highlights (gone at
    capture, gamma can't move a pixel already at 0 or 255); perfect deblur is ill-posed.
  - **Known blind spot (documented, NOT fixed — user's call 2026-09-05):** can't tell intentional
    portrait bokeh from a blur defect (uniform-blur→uniform-sharp training pairs; tiling sees each
    256 window alone). May over-sharpen tasteful bokeh. In README's Known Limits section,
    `docs/writeup.html`, [[phase2b-restoration-plan]].
- **Real-ESRGAN (`src/restore_sota.py`, `models/realesr-general-x4v3.pth`) — TRIED, then DROPPED,
  now orphaned.** Shipped briefly (2026-09-11) as an "AI restoration" toggle, then removed
  2026-09-12: made real photos look artificially smooth/"plasticky" on close inspection, same
  general over-smoothing problem as the from-scratch U-Net just via a different mechanism. The
  file, its weights, and the `spandrel` dependency are no longer imported by anything reachable
  from `app.py` — full audit confirmed zero references anywhere in the live code path. Candidate
  for deletion if the user wants a cleanup pass (along with the unused `seaborn` dev dependency).
- **Ongoing practice (started 2026-09-12, standing instruction for the rest of this project):**
  the user wants technical explanations (concepts, design tradeoffs, root-caused bugs) logged to
  `docs/review_notes.md` as they come up, not just left in chat — they intend to self-test on this
  material after the project wraps up. See [[feedback_review_notes]]. Keep doing this without
  being asked again each time.
- **Streamlit app (`app.py`, repo root = the Community Cloud main file; built 2026-09-01):**
  multi-upload capped at `MAX_IMAGES = 15` (free-tier RAM; ingest downscales to long side 1400),
  classify-only-new-files with a progress bar, `st.session_state` keyed by `file_id`, 4-per-row
  card grid (`st.container(border=True)` + `st.image(width="stretch")` + `st.expander(type="compact")`
  hiding the 5 per-class `st.progress` bars until opened), an "Enhance" primary button that runs
  every uploaded photo (no "flagged only" gate anymore — `enhance()`'s own proportional-strength
  design already leaves clean photos untouched, so the gate would've been redundant) →
  before/after `st.columns(2)` + per-image `st.download_button`. No mode checkbox of any kind —
  removed along with Real-ESRGAN, see the enhancement facts above. Streamlit **1.62.0**; native
  elements only, no CSS, sentence casing, Material icons. `requirements.txt` already has every dep.
- **UI (BUILT):** see the Streamlit-app fact above. `app.py` is the deployment entry point.
- **Phase 2 status:** 2a (classical exposure/contrast/noise fixes) and 2b (blur-specialist GAN,
  two-stage trained, shipped) both done — see the enhancement facts above.
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

## Project status (updated 2026-09-13)

**All phases done and DEPLOYED, user considers the project complete** — live at
`imagequalityclassifier.streamlit.app` (Streamlit Community Cloud, public repo
`github.com/Mohdshamik11/ImageQuality_Classifier`, branch `main`, main file `app.py`,
Python 3.11; redeploys on push to `main`). `restore-gan` feature branch merged to `main`
(fast-forward, no conflicts) 2026-09-13.

- **Phase 1 — classifier:** frozen, `models/traincombo_best.pt`. See fact above + [[baseline-model-spec]].
- **Enhancement:** ONE unified `enhance()` path, proportional to each defect's raw probability —
  classical fixes for exposure/contrast/noise, a two-stage-trained blur-specialist GAN
  (`restore_blur_gan_real_ext_lpips.pt`) for blur. Real-ESRGAN and the from-scratch U-Net were
  BOTH tried and dropped (over-smoothing); the from-scratch U-Net is kept as the base the GAN
  warm-started from, Real-ESRGAN's code/weights are now orphaned (candidate for a cleanup pass —
  see the fact above). See the enhancement facts above + [[phase2b-restoration-plan]].
- **README.md, docs/writeup.html, SKILL.md (this file)** brought back in sync with the code
  2026-09-13 — they had drifted to describe the dropped Real-ESRGAN toggle after the GAN work
  shipped. If a future session finds another mismatch between docs and `enhance.py`/`app.py`,
  trust the code and fix the docs, not the other way around.

Project is user-declared DONE. Do not reopen or extend any phase unless the user explicitly asks.
Known not-started ideas, lowest priority now that the user is happy with the result: a repo
cleanup pass (delete `restore_sota.py`, `realesr-general-x4v3.pth`, the `spandrel`/`seaborn`
dependencies — see the orphaned-Real-ESRGAN fact above); moving to a GPU host for a bigger
model or a dedicated face-restoration sub-pipeline (GFPGAN/CodeFormer) — floated early on, never
pursued, no indication the user still wants it.

`requirements.txt` (app-only, CPU torch) needs no changes for the current blur-GAN —
`restore_blur_gan.py` uses only torch/numpy/PIL, same as `restore_infer.py` before it. The
eval/training-only deps (`pyiqa`, `lpips`, `pytorch-msssim`, `matplotlib`) are in
`requirements-dev.txt`. OOM fallback for the app: `MAX_IMAGES` 8 / `INGEST_LONG_SIDE` 1000.

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
