# Review notes — concepts to be tested on

Running log of technical concepts and decisions covered while building the blur-specialist
GAN and refining the enhancement pipeline. Meant for self-testing later: read a heading,
try to explain it out loud before re-reading the note.

---

## Why GAN loss beats pure regression loss for deblurring

A model trained on L1/perceptual loss alone learns to minimize *average* pixel error across
many plausible sharp outputs for the same blurry input — and the mathematical average of many
plausible sharp images is a blurry image. That's why the original from-scratch U-Net (trained
on L1+SSIM+perceptual only) softened detail even though its loss numbers looked fine. Adding
an adversarial loss changes the objective from "minimize average error" to "produce something
a discriminator can't distinguish from a real sharp photo" — which rewards committing to one
plausible sharp answer instead of hedging toward the blurry average.

## GAN vs. discriminator — these are not the same thing

A **GAN** is the whole two-network training *system*. A **discriminator** is just one of the two
networks inside it. Analogy: forger (generator) vs. art authenticator (discriminator) — they
train against each other, each pushing the other to improve. "GAN" names the competitive
setup as a whole; "discriminator" names the judge half specifically. In this project,
`RestoreUNet` is the generator (the model actually shipped and used), and
`UNetDiscriminatorSN` is the discriminator (training-only scaffolding, thrown away afterward —
it's not part of the deployed app at all). "Training with a GAN loss" means training partly
against a discriminator's judgment, not only against a fixed formula like L1 or SSIM.

## Discriminator architecture — UNetDiscriminatorSN

Real-ESRGAN's design, reused for our blur-GAN: a small U-Net (encoder/decoder with skips) that
outputs a per-pixel real/fake logit map instead of a single real/fake score for the whole
image — gives more localized gradient signal to the generator. Spectral normalization is
applied to every conv except the first/last: it constrains each layer's Lipschitz constant,
which keeps the discriminator's gradients well-behaved and is a big part of why GAN training
here didn't destabilize.

## Channels and N_BLOCKS (model capacity)

- **Channels**: the number of parallel learned feature maps at a given point in the network —
  each channel is one learned filter's output (edge detector, texture detector, etc). More
  channels = more distinct information carried per pixel location.
- **N_BLOCKS**: how many residual blocks are stacked *at the same resolution* before moving to
  the next stage — more sequential refinement passes at that scale.
- Both raise model capacity (representational ceiling), but conv cost scales roughly with
  channels², so doubling channels is much more expensive than adding one more block.
- More capacity only helps if the model is actually capacity-starved (underfitting) AND paired
  with enough data to use it — otherwise it's just slower for no quality gain, or overfits.
- Changing channels/N_BLOCKS changes every layer's parameter *shape*, so a checkpoint trained
  at one config can't warm-start a model at a different config — the state_dict simply won't
  fit. This is why we deliberately did NOT combine "more data" and "bigger model" in one
  experiment: they'd confound each other's effect, and the architecture change would have
  thrown away the warm-start.

## Why Real-ESRGAN and the from-scratch U-Net got dropped from the pipeline

Both over-smoothed real photos into a flattened, "fake" look on close inspection, despite
decent no-reference quality metrics (BRISQUE/MUSIQ-style scores reward smoothness, which can
hide the perceptual softening a human eye catches immediately). This is the practical reason a
purpose-built, narrowly-scoped model (blur-only) ended up beating a general-purpose one for
this specific job.

## Fix ordering matters, and it's DIFFERENT for classical vs. learned deblurring

- **Classical unsharp mask**: amplifies whatever edges/noise currently exist. Must run
  *after* denoising, or it amplifies noise into ugly sharpened-noise artifacts.
- **Learned blur-GAN**: the opposite. Non-local-means denoising *smooths texture* — and the
  GAN can't sharpen detail that's already been averaged away by an upstream denoiser. So the
  GAN must run *before* denoising, on the least-processed image available, with any residual
  noise cleaned up on its output afterward.
- This was a real bug introduced when Real-ESRGAN was removed and noise got folded into the
  same pre-blur loop as the tonal fixes — the fix was pulling noise back out to run last
  (except in the classical-fallback branch, which needs the opposite order).

## Probability-driven vs. threshold-gated fix strength

Original design: a fix only ran if the classifier's confidence crossed the 50% "flagged"
threshold, and its strength was rescaled from that threshold up to 1.0. This meant a photo at
30% blur confidence (real defect, just not confident enough to flag) got *zero* correction.
Current design: every fix's strength is directly proportional to the raw probability
(`strength_from_prob(p) = clip(p, 0, 1)`), with a small floor (`MIN_STRENGTH`) below which a
fix is skipped as a no-op. A genuinely clean photo ends up untouched because there's nothing to
scale up, not because a threshold excluded it.

## Gamma correction's hard limit — why "fix overexposure" can't undo clipping

`L_new = 255 * (L/255)^gamma`. Plug in `L = 255`: `(255/255)^gamma = 1` for ANY gamma, so
`L_new = 255` — unchanged. 0 and 255 are gamma's fixed points. A pixel that's genuinely clipped
to pure white (or pure black) has zero tonal information left to redistribute; no gamma curve,
classical or learned, can recover it. Only pixels *near* the extreme (245, 250) can be pulled
back — anything already sitting exactly at the extreme is gone for good.

## Why the exposure fix's severity measurement had to change (95th percentile → 75th)

First attempt: measure the 95th/5th percentile as "how overexposed/underexposed is this
photo," compute the exact gamma needed to pull that value to a target, apply globally. Bug:
gamma is applied to the WHOLE luminance channel at once, so an extreme gamma needed to fix a
small blown-highlight region also crushes every midtone in the photo. Measured on a test image:
a "mild" brightening (×1.05) still had 9.7% of pixels fully clipped (a bright sky in frame),
which alone was enough to demand gamma ≈ 20 (clamped to a then-too-loose bound of 3.0),
collapsing the median brightness from 176 to 95 — a wildly disproportionate global effect for a
supposedly mild case.

Fix: measure the 75th/25th percentile (upper/lower-*midtone*) instead of 95th/5th. This tracks
overall picture brightness — which is literally what "overexposed"/"underexposed" mean — rather
than being thrown off by a handful of already-blown highlight pixels that don't represent the
photo as a whole. Also tightened the gamma bound to 1.8 max. Lesson: when a single global
parameter (here, gamma) is computed from a measurement, that measurement must represent what
the parameter will actually affect globally — using an extreme/local statistic to drive a
global transform is a recipe for disproportionate side effects.

## The blur-GAN's working resolution vs. output resolution

`restore_blur_gan.py` caps its internal working resolution to `MAX_LONG_SIDE` (768px), since
the checkpoint was trained on 256px crops from images resized to ≤800px — going larger is
outside its training distribution. The function scales its output back up to match the
*input's* original size before returning, so calling it never shrinks a photo. But this means:
for photos larger than 768px long side, the actual deblurring computation happens on a
downscaled proxy, then the result is enlarged (Lanczos) back to full size — the output
dimensions are correct, but the fine detail the model could sharpen was capped by that lower
working resolution. Enlarging inherently reintroduces some softness. Same output size ≠
full-resolution deblurring.

## Reading GAN training curves: discriminator/generator balance

`train_d` (discriminator's own loss) and the generator's `adv` term (its loss for trying to
fool the discriminator) should hover near `ln(2) ≈ 0.693` — the loss value at which the
discriminator is exactly 50/50 guessing. In our runs, `train_d` sat slightly below 0.693 (D
can tell real from fake a bit better than chance) while `adv` sat above it (G isn't fully
fooling D) — a stable, mildly D-favored equilibrium. Danger signs would be `train_d` collapsing
toward 0 (discriminator totally dominant, generator gets no useful gradient — training stalls)
or generator loss exploding (mode collapse). Neither happened in any run here.

## Reading validation metrics together, not in isolation

PSNR and SSIM (pixel/structure fidelity) can plateau early while LPIPS (perceptual similarity,
closer to human judgment) keeps improving for many more epochs — this happened in both blur-GAN
training runs. That's the adversarial+perceptual loss terms still sharpening texture that
pixel-wise metrics are blind to. Conversely, watch for PSNR/SSIM *degrading* while training loss
keeps improving — that pattern (not seen here) would indicate overfitting.

## Real vs. synthetic training data — closing the domain gap

`degrade.py`'s synthetic blur (Gaussian/motion/defocus/aniso kernels) gives cheap, unlimited,
label-perfect training pairs, but a convolution kernel isn't identical to real camera-shake
statistics. RealBlur-J supplies genuinely real pairs: the same scene shot through a
beam-splitter rig simultaneously at long exposure (blurry) and short exposure (sharp),
pre-aligned via ECC (a geometric registration algorithm) + intensity correction. Fine-tuning
the COCO-trained checkpoint further on RealBlur-J (warm start, not from scratch) measurably
improved both the validation numbers and — more importantly — the visible sharpness on a real
test photo, confirming the domain gap was real and worth closing.

## Fast-forward merge

`git merge` produced no separate merge commit and no conflicts here because `main` hadn't
diverged from `restore-gan` at all since the branch point — Git just moved `main`'s pointer
forward to the tip of `restore-gan` ("fast-forward"). A merge commit (and possible conflicts)
only appears when both branches have their own new commits since diverging.
