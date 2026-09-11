"""
Realistic random degradation pipeline for the phase-2b restoration model.

`degrade(clean_rgb, rng)` takes a clean image and returns a plausibly-broken
version of it. It is called on the fly during training: every time a clean crop
is loaded, it gets a *fresh* random degradation, so the model never sees the same
(degraded, clean) pair twice.

The realism of this file is the single biggest lever on how well the trained
model works on real photos. A model trained only on `cv2.GaussianBlur` learns to
undo Gaussian blur and little else. So this pipeline mixes:

    blur    - Gaussian / motion / defocus / anisotropic, random strength
    noise   - Poisson (signal-dependent) + Gaussian (read noise)
    resize  - downscale then upscale, random interpolation -> aliasing + softness
    tone    - brightness / contrast / gamma shifts (the exposure & contrast defects)
    jpeg    - in-memory JPEG round-trip at random quality -> blocking + ringing

A random 1-3 of {blur, noise, resize, tone} are applied in random order; JPEG, if
chosen, always goes last (it is the final save step in real life). With a small
probability the whole thing runs twice ("high-order" degradation, like a photo
that has been through several lossy apps).

All maths is done in float32 [0, 1]. Input and output are uint8 RGB HxWx3.

Usage (eyeball a few):
    python src/degrade.py                       # degrades 8 images from data/raw/
    python src/degrade.py path/to/image.jpg     # degrades one image, 6 variants
    python src/degrade.py --blur-only           # preview degrade_blur_only() instead
"""
from pathlib import Path

import cv2
import numpy as np

# ---- small helpers -------------------------------------------------------- #

def _to_float(img):
    """uint8 [0,255] HxWx3 RGB  ->  float32 [0,1]."""
    return img.astype(np.float32) / 255.0


def _to_uint8(img):
    """float [0,1] (may be slightly out of range)  ->  uint8 [0,255]."""
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _skew(rng, lo, hi, power=2.0):
    """Random value in [lo, hi] biased toward `lo`. power=1 is uniform; higher
    packs more draws near the low end while the tail still reaches `hi`. Used so
    MILD blur is the common case and STRONG blur stays rare-but-present."""
    return lo + (hi - lo) * (rng.random() ** power)


# ---- individual degradations (float [0,1] in, float [0,1] out) ----------- #

def _blur(img, rng, power=2.0):
    """One of four blur types, random strength. `power` is passed to `_skew`:
    2.0 (default, used by the general pipeline) packs strength toward mild;
    lower values (e.g. 1.3, used by the blur-only pipeline below) spread more
    weight onto moderate/strong blur, since that pipeline's only job is blur."""
    kind = rng.choice(["gauss", "motion", "defocus", "aniso"])

    if kind == "gauss":
        sigma = _skew(rng, 0.4, 2.0, power)
        return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma)

    if kind == "motion":
        # A line of ones, rotated to a random angle -> smear along that direction.
        length = int(_skew(rng, 4, 17, power))
        angle = rng.uniform(0, 180)
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
        k = cv2.warpAffine(k, M, (length, length))
        k /= k.sum() + 1e-8
        return cv2.filter2D(img, -1, k)

    if kind == "defocus":
        # A filled disc -> the classic out-of-focus "circle of confusion".
        radius = max(2, int(_skew(rng, 2, 6, power)))
        d = 2 * radius + 1
        k = np.zeros((d, d), np.float32)
        cv2.circle(k, (radius, radius), radius, 1.0, -1)
        k /= k.sum() + 1e-8
        return cv2.filter2D(img, -1, k)

    # anisotropic Gaussian: different sigma on x and y -> directional softness
    sx, sy = _skew(rng, 0.4, 2.0, power), _skew(rng, 0.4, 2.0, power)
    return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sx, sigmaY=sy)


def _noise(img, rng):
    """Poisson (shot) noise scaled by brightness, plus a little Gaussian read noise."""
    # Poisson: model each pixel as a photon count. Fewer 'photons' (small scale)
    # => grainier. rng.poisson wants a rate array; divide back out to [0,1].
    scale = rng.uniform(30.0, 140.0)          # lower = noisier
    noisy = rng.poisson(np.clip(img, 0, 1) * scale) / scale

    # Gaussian read noise on top (sensor electronics, brightness-independent).
    sigma = rng.uniform(0.0, 0.04)
    noisy = noisy + rng.normal(0.0, sigma, img.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def _resize(img, rng):
    """Downscale then upscale back -> lost detail, aliasing, interpolation softness."""
    h, w = img.shape[:2]
    scale = rng.uniform(0.35, 0.9)
    interp_down = rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    interp_up = rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=int(interp_down))
    return cv2.resize(small, (w, h), interpolation=int(interp_up))


def _tone(img, rng):
    """Brightness / contrast / gamma shift -- the exposure and contrast defects."""
    if rng.random() < 0.7:                                   # exposure
        img = img * rng.uniform(0.35, 2.2)
    if rng.random() < 0.6:                                   # contrast about mid-grey
        img = 0.5 + (img - 0.5) * rng.uniform(0.5, 1.7)
    if rng.random() < 0.5:                                   # gamma
        img = np.clip(img, 1e-6, 1.0) ** rng.uniform(0.55, 1.8)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _jpeg(img, rng):
    """In-memory JPEG round-trip at random quality -> blocking + ringing."""
    quality = int(rng.integers(35, 96))
    bgr = cv2.cvtColor(_to_uint8(img), cv2.COLOR_RGB2BGR)    # the codec assumes BGR
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return _to_float(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB))


_ORDERABLE = {"blur": _blur, "noise": _noise, "resize": _resize, "tone": _tone}


# ---- the pipeline ------------------------------------------------------- #

def _one_pass(img, rng, strong):
    """Apply a random 1-3 of the orderable ops, then maybe JPEG."""
    n = int(rng.integers(1, 4)) if strong else 1
    ops = list(rng.choice(list(_ORDERABLE), size=min(n, len(_ORDERABLE)), replace=False))
    rng.shuffle(ops)
    recipe = []
    for name in ops:
        img = _ORDERABLE[name](img, rng)
        recipe.append(name)
    if rng.random() < (0.9 if strong else 0.5):
        img = _jpeg(img, rng)
        recipe.append("jpeg")
    return np.clip(img, 0.0, 1.0).astype(np.float32), recipe


def degrade(clean_rgb, rng, return_recipe=False):
    """Clean uint8 RGB image -> degraded uint8 RGB image (same size).

    rng: a numpy Generator (np.random.default_rng(...)).
    ~15% of calls apply only a mild single degradation, so the model also sees the
    'nearly clean' regime; ~10% run the pipeline twice (high-order).
    """
    img = _to_float(clean_rgb)
    roll = rng.random()

    if roll < 0.15:                          # gentle: one op, no JPEG
        name = rng.choice(list(_ORDERABLE))
        img = _ORDERABLE[name](img, rng)
        recipe = [name]
    else:
        img, recipe = _one_pass(img, rng, strong=True)
        if roll > 0.90:                      # high-order: a second, lighter pass
            img, r2 = _one_pass(img, rng, strong=False)
            recipe += ["|"] + r2

    out = _to_uint8(img)
    return (out, recipe) if return_recipe else out


# ---- blur-only pipeline (for the blur-specialist GAN fine-tune) ---------- #

def degrade_blur_only(clean_rgb, rng, return_recipe=False):
    """Clean uint8 RGB -> blurred uint8 RGB. Used only for the blur-specialist
    fine-tune (src/train_restore_gan.py) -- NOT the general `degrade()` above.

    Exposure/contrast are already handled by the classical fixes and noise by
    Real-ESRGAN in the shipped pipeline (see src/enhance.py), so this model's
    training data is scoped to just blur -- narrower task, all its capacity and
    the adversarial loss pointed at the one hard, ill-posed problem.

    Differences from the general pipeline's blur op:
      - power=1.3 (vs 2.0): flatter skew, more moderate/strong coverage, since
        this is the only degradation the model ever sees.
      - occasionally a second, lighter blur pass (camera shake + defocus can
        stack in a real photo).
      - JPEG kept (not a "defect", just how photos are actually saved) at high
        probability; no noise / tone / resize.
    """
    img = _to_float(clean_rgb)
    recipe = []

    img = _blur(img, rng, power=1.3)
    recipe.append("blur")

    if rng.random() < 0.15:                       # stacked blur (shake + defocus)
        img = _blur(img, rng, power=1.3)
        recipe.append("blur2")

    if rng.random() < 0.85:
        img = _jpeg(img, rng)
        recipe.append("jpeg")

    out = _to_uint8(np.clip(img, 0.0, 1.0))
    return (out, recipe) if return_recipe else out


if __name__ == "__main__":
    import sys

    blur_only = "--blur-only" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--blur-only"]
    fn = degrade_blur_only if blur_only else degrade

    rng = np.random.default_rng(0)
    out_dir = Path("outputs/degrade_preview" + ("_blur_only" if blur_only else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args:
        srcs = [Path(args[0])] * 6
    else:
        srcs = sorted(Path("data/raw").glob("raw_00[0-1][0-9].jpg"))[:8]

    for i, p in enumerate(srcs):
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        deg, recipe = fn(rgb, rng, return_recipe=True)
        stem = f"{p.stem}_deg{i}" if args else p.stem
        cv2.imwrite(str(out_dir / f"{stem}.png"), cv2.cvtColor(deg, cv2.COLOR_RGB2BGR))
        print(f"{stem:18s} {' -> '.join(recipe)}")

    print(f"\nwrote previews to {out_dir}/")
