"""
Pretrained SOTA restoration inference.

Runs the published **Real-ESRGAN** general model (`realesr-general-x4v3`, a 1.2M-param
SRVGGNetCompact, ~5 MB) via `spandrel` -- a small modern loader that reads the weight
file and builds the right architecture, with none of `basicsr`'s install pain.

It's a x4 super-resolution model; used here as a *restorer*: the x4 pass removes
noise/compression/soft-focus and re-crisps edges, then the result is scaled back
down to a sane size. Nothing is trained -- one frozen forward pass per image.

Why this and not the from-scratch phase-2b U-Net: the U-Net (see src/train_restore.py,
src/restore_infer.py) softens detail -- a regression-loss artifact that needs an
adversarial-loss training run and far more data/compute than a free tier allows.
Real-ESRGAN was trained exactly that way (GAN loss, large data) by its authors.

Usage (smoke test):
    python src/restore_sota.py path/to/photo.jpg
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "realesr-general-x4v3.pth"

MAX_IN_LONG_SIDE = 512      # cap the model input -- bounds CPU time (~1.5 s) and RAM
OUT_LONG_SIDE = 1400        # scale the x4 output back down to this


def available() -> bool:
    return WEIGHTS.exists()


@lru_cache(maxsize=1)
def load_sota_model():
    from spandrel import ModelLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    md = ModelLoader().load_from_file(str(WEIGHTS))
    net = md.model.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net, int(md.scale), device


def restore_image(image: Image.Image, strength: float = 1.0,
                  max_in: int = MAX_IN_LONG_SIDE, out_long_side: int = OUT_LONG_SIDE) -> Image.Image:
    """PIL photo (any size) -> restored PIL photo.

    strength in [0, 1] blends the model output with a plain resize of the input:
    1.0 = full Real-ESRGAN, 0.0 = just a clean downscale (model untouched).
    """
    img = image.convert("RGB")

    # 1. downscale the model input -- keeps the CPU forward pass ~1-2 s
    lo = img
    if max(lo.size) > max_in:
        s = max_in / max(lo.size)
        lo = lo.resize((round(lo.size[0] * s), round(lo.size[1] * s)), Image.LANCZOS)

    # 2. one frozen x4 forward pass
    net, scale, device = load_sota_model()
    x = torch.from_numpy(np.asarray(lo, np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        y = net(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    restored = Image.fromarray((y * 255).round().astype(np.uint8))   # size = lo.size * scale

    # 3. bring the x4 result back to a sane size (never upsize past the original)
    target_long = min(out_long_side, max(img.size))
    if max(restored.size) != target_long:
        r = target_long / max(restored.size)
        restored = restored.resize((round(restored.size[0] * r), round(restored.size[1] * r)), Image.LANCZOS)

    # 4. strength blend against a plain resize of the ORIGINAL at the same size
    strength = float(min(1.0, max(0.0, strength)))
    if strength >= 1.0:
        return restored
    base = img.resize(restored.size, Image.LANCZOS)
    a = np.asarray(restored, np.float32)
    b = np.asarray(base, np.float32)
    out = (strength * a + (1.0 - strength) * b).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


if __name__ == "__main__":
    import sys
    import time
    p = sys.argv[1]
    im = Image.open(p)
    t0 = time.time()
    out = restore_image(im)
    save_to = p.rsplit(".", 1)[0] + "_sota.png"
    out.save(save_to)
    print(f"{p}  {im.size} -> {out.size}   {time.time() - t0:.1f}s   wrote {save_to}   "
          f"(weights present: {available()})")
