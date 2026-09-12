"""
Enhancement -- classical, flag-driven fixes for every defect except blur:

    underexposed / overexposed  (tonal, gamma)
        -> low contrast          (percentile stretch)
            -> blur              (blur-specialist GAN, src/restore_blur_gan.py)
                -> noise         (non-local means, LAST)

Real uploads weren't degraded by our scripts, so every fix is blind: it nudges
the image toward a clean look, strength scaled DIRECTLY by the classifier's raw
per-defect probability -- not gated by the 50% "flagged" threshold. A photo the
classifier is only 20% suspicious of gets a light touch; one at 90% gets close
to the full fix; a genuinely clean photo (probabilities near 0 everywhere) gets
left alone because there's nothing to scale up. Fixes below MIN_STRENGTH are
skipped outright rather than run at a token, wasted-compute non-effect.

Blur is the one defect a plain filter can't really fix -- an unsharp mask can
only amplify edges that still exist, it can't recover detail actually lost to
blur -- so that step uses the trained GAN (see src/train_restore_gan.py)
instead. It runs on the tonal-fixed image but BEFORE denoising: non-local-means
smooths texture, and the GAN can't sharpen detail that's already been averaged
away, so denoising has to happen after deblurring, not before it. (Gamma
remapping doesn't have that problem -- it's a per-pixel curve, not a blur --
so the tonal fixes staying ahead of the GAN costs it nothing.) If the
checkpoint isn't present, blur falls back to the unsharp mask instead, which
DOES want denoising to happen first (it amplifies whatever noise is left).

We tried two general-purpose learned restorers here first -- a from-scratch
U-Net and pretrained Real-ESRGAN -- and dropped both: they over-smoothed real
photos into a flattened, "fake" look. See docs/writeup.html section 11.

Usage (smoke test):
    python src/enhance.py path/to/image
"""
import numpy as np
import cv2
from PIL import Image

import restore_blur_gan  # blur-specialist GAN fine-tune

DEFECT_COLUMNS = ["blur", "underexposed", "overexposed", "noise", "contrast"]

# Below this, a fix is skipped entirely -- not run at a strength so low it's
# indistinguishable from a no-op, just wasted compute (and, for blur, a real
# GAN forward pass we don't need to pay for).
MIN_STRENGTH = 0.02


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def strength_from_prob(p: float) -> float:
    """Classifier probability in [0, 1] -> fix strength in [0, 1], directly
    proportional -- NOT gated by the 50% "flagged" threshold. A photo the
    classifier is only mildly suspicious of gets a mild fix."""
    return float(max(0.0, min(1.0, p)))


def _on_luminance(rgb: np.ndarray, fn) -> np.ndarray:
    """Apply `fn` to the LAB L channel only, so colours are untouched."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    lab[:, :, 0] = np.clip(fn(L), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _blend_to_original(original: Image.Image, result: Image.Image, strength: float) -> Image.Image:
    """result <- strength -> a plain resize of the original, at result's size."""
    strength = float(min(1.0, max(0.0, strength)))
    if strength >= 1.0:
        return result
    base = np.asarray(original.convert("RGB").resize(result.size, Image.LANCZOS), np.float32)
    a = np.asarray(result, np.float32)
    return Image.fromarray((strength * a + (1.0 - strength) * base).round().clip(0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# per-defect fixes (rgb uint8 in/out)
# --------------------------------------------------------------------------- #
def _gamma_to_target(measured: float, target: float, gamma_bounds=(0.5, 1.8)) -> float:
    """The gamma that maps `measured` (0-255) onto `target` (0-255): solves
    255*(measured/255)**gamma == target for gamma. Bounds keep the correction
    a global tone shift rather than a crush -- a measured value pinned near 0
    or 255 (heavy clipping) would otherwise demand a huge exponent, and
    because gamma applies to the WHOLE luminance channel at once, an extreme
    exponent needed to fix a few blown pixels would also crush every midtone
    in the photo along with them. First measured value: 254/1, keeps both
    logs well-defined."""
    measured = min(max(measured, 1.0), 254.0)
    gamma = np.log(target / 255.0) / np.log(measured / 255.0)
    return float(np.clip(gamma, gamma_bounds[0], gamma_bounds[1]))


def fix_underexposed(rgb, strength, target=65.0, pct=25.0):
    """Gamma computed from the image's OWN lower-midtone level (25th
    percentile), not a fixed guess: a mildly dim photo and a very dark one get
    correspondingly different lifts. Using a lower-MIDtone percentile rather
    than a deep-shadow one (5th, 1st) keeps the measurement representative of
    overall brightness -- exactly what "underexposed" means -- instead of
    being thrown off by a small patch of near-black shadow that doesn't
    reflect the photo as a whole. Can't do anything for pixels truly clipped
    to 0 -- gamma leaves 0 and 255 unchanged no matter the exponent, there's
    no tonal data left to move."""
    def stretch(L):
        lo = np.percentile(L, pct)
        if lo >= target:                                # already bright enough down there
            return L
        gamma = _gamma_to_target(lo, target)
        corrected = 255.0 * (L / 255.0) ** gamma
        return L * (1.0 - strength) + corrected * strength
    return _on_luminance(rgb, stretch)


def fix_overexposed(rgb, strength, target=190.0, pct=75.0):
    """Mirrors fix_underexposed for the bright end (upper-midtone/75th
    percentile, not just the blown highlights) -- same clipping caveat
    applies to pixels already sitting at pure 255."""
    def stretch(L):
        hi = np.percentile(L, pct)
        if hi <= target:                                # already dark enough up there
            return L
        gamma = _gamma_to_target(hi, target)
        corrected = 255.0 * (L / 255.0) ** gamma
        return L * (1.0 - strength) + corrected * strength
    return _on_luminance(rgb, stretch)


def fix_low_contrast(rgb, strength):
    def stretch(L):
        lo, hi = np.percentile(L, 1.0), np.percentile(L, 99.0)
        if hi - lo < 1e-3:
            return L
        stretched = np.clip((L - lo) / (hi - lo) * 255.0, 0.0, 255.0)
        return L * (1.0 - strength) + stretched * strength
    return _on_luminance(rgb, stretch)


def fix_noise(rgb, strength):
    h = 12.0 * strength                                # non-local means filter strength
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    den = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
    return cv2.cvtColor(den, cv2.COLOR_BGR2RGB)


def fix_blur(rgb, strength):
    """Unsharp mask -- not a true deblur, just edge amplification. Fallback for
    when restore_blur_gan's checkpoint isn't available."""
    amount = 0.8 * strength
    blurred = cv2.GaussianBlur(rgb, (0, 0), 2.0).astype(np.float32)
    sharp = rgb.astype(np.float32) * (1.0 + amount) - blurred * amount
    return np.clip(sharp, 0, 255).astype(np.uint8)


# Tonal fixes run first -- a gamma curve doesn't destroy texture, so it costs
# the blur-GAN nothing to go ahead of it. Noise and blur are handled separately
# below: noise fix runs LAST (after deblurring via the GAN), so it never smooths
# away detail the GAN would otherwise have sharpened.
_FIXES = [
    ("underexposed", fix_underexposed),
    ("overexposed", fix_overexposed),
    ("contrast", fix_low_contrast),
]


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def _label(name: str, flags: dict) -> str:
    """'blur' if the classifier confidently flagged it, else 'blur (mild)' for a
    sub-threshold proportional touch-up."""
    return name if flags.get(name) else f"{name} (mild)"


def enhance(image: Image.Image, flags: dict, probs: dict, strength: float = 1.0):
    """image    : PIL image
       flags    : {defect: bool}  -- from predict(); used only to label results
                  ("mild" vs confidently-flagged), not to gate which fixes run
       probs    : {defect: float} -- from predict(); drives fix strength directly
       strength : 0-1, overall dial on top of that (1 = full, 0 = untouched)

    Every defect's fix runs, scaled by its own raw probability -- a genuinely
    clean photo (probabilities near 0 everywhere) ends up untouched because
    there's nothing to scale up, not because a threshold excluded it.

    Returns (enhanced PIL image, list of labels describing what was done).
    """
    src = np.array(image.convert("RGB"))
    rgb = src.copy()
    applied = []

    # tonal fixes -- safe ahead of deblurring, see module docstring
    for name, fn in _FIXES:
        s = strength_from_prob(probs.get(name, 0.0))
        if s > MIN_STRENGTH:
            rgb = fn(rgb, s)
            applied.append(_label(name, flags))

    # blur -- GAN runs on the tonal-fixed, NOT-yet-denoised image so it has the
    # most texture available to sharpen; the unsharp-mask fallback wants the
    # opposite (denoise first, then sharpen) so it doesn't amplify noise
    blur_s = strength_from_prob(probs.get("blur", 0.0))
    used_gan = False
    noise_handled = False
    if blur_s > MIN_STRENGTH:
        if restore_blur_gan.available():
            restored = restore_blur_gan.restore_image(Image.fromarray(rgb), strength=blur_s)
            rgb = np.array(restored)                    # may now be a different resolution
            used_gan = True
            applied.append(_label("blur", flags) + " (GAN)")
        else:
            noise_s = strength_from_prob(probs.get("noise", 0.0))
            if noise_s > MIN_STRENGTH:
                rgb = fix_noise(rgb, noise_s)
                applied.append(_label("noise", flags))
            noise_handled = True
            rgb = fix_blur(rgb, blur_s)
            applied.append(_label("blur", flags) + " (sharpen)")

    # noise -- last, so denoising never costs the GAN texture it needs (the
    # unsharp-mask fallback above already handled it in the opposite order)
    if not noise_handled:
        noise_s = strength_from_prob(probs.get("noise", 0.0))
        if noise_s > MIN_STRENGTH:
            rgb = fix_noise(rgb, noise_s)
            applied.append(_label("noise", flags))

    result = Image.fromarray(rgb)
    if used_gan:
        return _blend_to_original(image, result, strength), applied

    s = float(min(1.0, max(0.0, strength)))
    if s < 1.0:                                        # blend the whole result back toward the input
        blended = s * np.asarray(result, np.float32) + (1.0 - s) * src.astype(np.float32)
        result = Image.fromarray(blended.round().clip(0, 255).astype(np.uint8))
    return result, applied


if __name__ == "__main__":
    import sys
    from predict import predict

    path = sys.argv[1]
    im = Image.open(path).convert("RGB")
    pred = predict(im)
    enhanced, applied = enhance(im, pred["flags"], pred["probs"])
    save_to = path.rsplit(".", 1)[0] + "_enhanced.png"
    enhanced.save(save_to)

    print(f"{path}  {im.size} -> {enhanced.size}")
    print(f"  flagged : {[c for c in DEFECT_COLUMNS if pred['flags'][c]] or 'none'}")
    print(f"  applied : {applied}")
    print(f"  blur-GAN present: {restore_blur_gan.available()}")
    print(f"  wrote {save_to}")
