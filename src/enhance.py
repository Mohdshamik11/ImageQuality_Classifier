"""
Enhancement: a learned restoration model, with the classical fixes as fallback.

`enhance()` runs the phase-2b restoration U-Net (src/restore_infer.py) on the
whole image. The model is blind -- it takes only the pixels -- so the classifier
flags are used only to decide *whether* to enhance (done by the caller) and to
report what was targeted, not to steer the model.

If the restoration checkpoint is missing, `enhance()` falls back to the original
phase-2a classical, flag-driven fixes (kept below):

    underexposed / overexposed  (tonal, gamma)
        -> low contrast          (percentile stretch)
            -> noise             (non-local means, before sharpening)
                -> blur          (unsharp mask, last -- it amplifies noise)

Real uploads weren't degraded by our scripts, so every fix is blind: it nudges
the image toward a clean look, strength scaled by the classifier's confidence.

Usage (smoke test):
    python src/enhance.py path/to/image
"""
import numpy as np
import cv2
from PIL import Image

import restore_infer

DEFECT_COLUMNS = ["blur", "underexposed", "overexposed", "noise", "contrast"]


# --------------------------------------------------------------------------- #
# classical fallback -- helpers
# --------------------------------------------------------------------------- #
def strength_from_prob(p: float) -> float:
    """Probability in [0.5, 1.0] -> fix strength in [0.0, 1.0]. A detection at the
    0.5 threshold is a near no-op; a confident 1.0 gets the full moderate fix."""
    return float(max(0.0, min(1.0, (p - 0.5) / 0.5)))


def _on_luminance(rgb: np.ndarray, fn) -> np.ndarray:
    """Apply `fn` to the LAB L channel only, so colours are untouched."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    lab[:, :, 0] = np.clip(fn(L), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# --------------------------------------------------------------------------- #
# classical fallback -- per-defect fixes (rgb uint8 in/out)
# --------------------------------------------------------------------------- #
def fix_underexposed(rgb, strength):
    gamma = 1.0 - 0.4 * strength                      # lift midtones/shadows
    return _on_luminance(rgb, lambda L: 255.0 * (L / 255.0) ** gamma)


def fix_overexposed(rgb, strength):
    gamma = 1.0 + 0.5 * strength                      # pull the bright end down
    return _on_luminance(rgb, lambda L: 255.0 * (L / 255.0) ** gamma)


def fix_low_contrast(rgb, strength):
    def stretch(L):
        lo, hi = np.percentile(L, 1.0), np.percentile(L, 99.0)
        if hi - lo < 1e-3:
            return L
        stretched = np.clip((L - lo) / (hi - lo) * 255.0, 0.0, 255.0)
        return L * (1.0 - strength) + stretched * strength
    return _on_luminance(rgb, stretch)


def fix_noise(rgb, strength):
    h = 3.0 + 9.0 * strength                          # non-local means filter strength
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    den = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
    return cv2.cvtColor(den, cv2.COLOR_BGR2RGB)


def fix_blur(rgb, strength):
    amount = 0.8 * strength                           # unsharp mask -- not true deblur
    blurred = cv2.GaussianBlur(rgb, (0, 0), 2.0).astype(np.float32)
    sharp = rgb.astype(np.float32) * (1.0 + amount) - blurred * amount
    return np.clip(sharp, 0, 255).astype(np.uint8)


_FIXES = [
    ("underexposed", fix_underexposed),
    ("overexposed", fix_overexposed),
    ("contrast", fix_low_contrast),
    ("noise", fix_noise),
    ("blur", fix_blur),
]


def _enhance_classical(image: Image.Image, flags: dict, probs: dict):
    rgb = np.array(image.convert("RGB"))
    applied = []
    for name, fn in _FIXES:
        if flags.get(name):
            rgb = fn(rgb, strength_from_prob(probs.get(name, 1.0)))
            applied.append(name)
    return Image.fromarray(rgb), applied


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def enhance(image: Image.Image, flags: dict, probs: dict):
    """image : PIL image
       flags : {defect: bool}  -- from predict(); which defects were detected
       probs : {defect: float} -- from predict()

    Returns (enhanced PIL image, list of labels describing what was done).
    Uses the learned restoration model; falls back to the classical fixes if the
    checkpoint is not present.
    """
    flagged = [c for c in DEFECT_COLUMNS if flags.get(c)]

    if restore_infer.available():
        out = restore_infer.restore_image(image)
        return out, ["learned restoration"] + ([f"targets: {', '.join(flagged)}"] if flagged else [])

    return _enhance_classical(image, flags, probs)


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
    print(f"  restoration model present: {restore_infer.available()}")
    print(f"  wrote {save_to}")
