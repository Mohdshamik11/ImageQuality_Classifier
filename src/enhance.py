"""
Phase 2a: classical, flag-driven image enhancement.

Each defect the classifier flags has one fix function. `enhance()` runs only the
flagged fixes, in an order chosen so earlier steps don't sabotage later ones:

    underexposed / overexposed  (tonal)
        -> low contrast          (tonal)
            -> noise             (denoise BEFORE sharpening)
                -> blur          (sharpen last -- it amplifies noise)

All fixes are BLIND: real uploads weren't degraded by our scripts, so we can't
undo a known factor. Each fix nudges the image toward a well-exposed / clean look,
and its strength scales with the classifier's confidence (a borderline 0.5
detection barely touches the image; a 0.95 detection gets the full moderate fix).

Tonal fixes (brightness, contrast) are done on the L channel of LAB only, so they
don't shift the photo's colours.

Usage (smoke test):
    python src/enhance.py path/to/image
"""
import numpy as np
import cv2
from PIL import Image

DEFECT_COLUMNS = ["blur", "underexposed", "overexposed", "noise", "contrast"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def strength_from_prob(p: float) -> float:
    """Map a probability in [0.5, 1.0] onto a fix strength in [0.0, 1.0].
    A detection right at the 0.5 threshold -> ~0 strength (near no-op);
    a confident 1.0 detection -> full (moderate) strength."""
    return float(max(0.0, min(1.0, (p - 0.5) / 0.5)))


def _on_luminance(rgb: np.ndarray, fn) -> np.ndarray:
    """Apply `fn` to the lightness channel only, via LAB, so colours are untouched.
    cv2's LAB L channel is 0-255 for an 8-bit image."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    lab[:, :, 0] = np.clip(fn(L), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# --------------------------------------------------------------------------- #
# per-defect fixes  (rgb uint8 in, rgb uint8 out)
# --------------------------------------------------------------------------- #
def fix_underexposed(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Gamma correction with gamma < 1: lifts the midtones and shadows while
    leaving the highlights roughly in place. Moderate: gamma 1.0 -> 0.6."""
    gamma = 1.0 - 0.4 * strength
    return _on_luminance(rgb, lambda L: 255.0 * (L / 255.0) ** gamma)


def fix_overexposed(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Gamma > 1: pulls the bright end down. Pixels already clipped to pure white
    stay there -- that detail is gone -- but everything below recovers. Moderate:
    gamma 1.0 -> 1.5."""
    gamma = 1.0 + 0.5 * strength
    return _on_luminance(rgb, lambda L: 255.0 * (L / 255.0) ** gamma)


def fix_low_contrast(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Percentile stretch: find where the darkest 1% and brightest 1% of pixels
    sit, and remap that range to the full 0-255. Blended with the original by
    `strength` so a borderline case is barely changed."""
    def stretch(L):
        lo, hi = np.percentile(L, 1.0), np.percentile(L, 99.0)
        if hi - lo < 1e-3:                      # near-flat image, nothing to stretch
            return L
        stretched = np.clip((L - lo) / (hi - lo) * 255.0, 0.0, 255.0)
        return L * (1.0 - strength) + stretched * strength
    return _on_luminance(rgb, stretch)


def fix_noise(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Non-local means: for each patch, find similar-looking patches elsewhere in
    the image and average them -- random noise cancels, real structure survives.
    `h` is the filter strength; scales 3 -> 12 with confidence. Runs in BGR, which
    is what this OpenCV routine expects."""
    h = 3.0 + 9.0 * strength
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    den = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
    return cv2.cvtColor(den, cv2.COLOR_BGR2RGB)


def fix_blur(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Unsharp mask: subtract a blurred copy from the original to isolate the
    edges, then add them back amplified. Re-crisps edges; it is not true deblur,
    and it would amplify noise, which is why it runs after denoising. Moderate:
    amount 0 -> 0.8."""
    amount = 0.8 * strength
    blurred = cv2.GaussianBlur(rgb, (0, 0), 2.0).astype(np.float32)
    sharp = rgb.astype(np.float32) * (1.0 + amount) - blurred * amount
    return np.clip(sharp, 0, 255).astype(np.uint8)


# order matters -- see module docstring
_FIXES = [
    ("underexposed", fix_underexposed),
    ("overexposed", fix_overexposed),
    ("contrast", fix_low_contrast),
    ("noise", fix_noise),
    ("blur", fix_blur),
]


def enhance(image: Image.Image, flags: dict, probs: dict):
    """Run every flagged fix in order.

    image  : PIL image
    flags  : {defect: bool}   -- from predict()
    probs  : {defect: float}  -- from predict(); sets each fix's strength

    Returns (enhanced PIL image, list of fixes applied).
    """
    rgb = np.array(image.convert("RGB"))
    applied = []
    for name, fn in _FIXES:
        if flags.get(name):
            rgb = fn(rgb, strength_from_prob(probs.get(name, 1.0)))
            applied.append(name)
    return Image.fromarray(rgb), applied


if __name__ == "__main__":
    import sys
    from predict import predict

    path = sys.argv[1]
    im = Image.open(path).convert("RGB")
    out = predict(im)

    def mean_luma(pil):
        L = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2LAB)[:, :, 0]
        return L.mean()

    enhanced, applied = enhance(im, out["flags"], out["probs"])
    save_to = path.rsplit(".", 1)[0] + "_enhanced.png"
    enhanced.save(save_to)

    print(f"{path}")
    print(f"  flagged : {[c for c in DEFECT_COLUMNS if out['flags'][c]] or 'none'}")
    print(f"  applied : {applied or 'none'}")
    print(f"  mean luminance {mean_luma(im):.1f} -> {mean_luma(enhanced):.1f}")
    print(f"  wrote {save_to}")
