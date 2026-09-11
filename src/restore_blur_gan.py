"""
Inference for the blur-specialist GAN fine-tune (models/restore_blur_gan_lpips.pt).

Same RestoreUNet architecture as src/restore_infer.py -- this checkpoint is a
warm-started fine-tune of that model (see src/train_restore_gan.py), continued
with an adversarial + perceptual loss on BLUR-ONLY degradation. Reuses
restore_infer's tiling machinery (restore_tensor); only the checkpoint and
default max size differ.

Usage (smoke test):
    python src/restore_blur_gan.py path/to/photo.jpg
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from restore_model import RestoreUNet
from restore_infer import restore_tensor, MAX_LONG_SIDE

CKPT = Path(__file__).resolve().parent.parent / "models" / "restore_blur_gan_lpips.pt"

# must match src/train_restore_gan.py CONFIG (warm-started from restore_infer's checkpoint,
# so same architecture)
BASE_CHANNELS = 48
N_BLOCKS = 3


def available() -> bool:
    return CKPT.exists()


@lru_cache(maxsize=1)
def load_blur_gan_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RestoreUNet(base_channels=BASE_CHANNELS, n_blocks=N_BLOCKS).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()
    return model, device


def restore_image(image: Image.Image, max_long_side: int = MAX_LONG_SIDE,
                  strength: float = 1.0) -> Image.Image:
    """PIL photo (any size) -> deblurred PIL photo, SAME size as the input.

    The model only ever runs at <= max_long_side (matching its training scale),
    but the result is scaled back up to the input's original size before
    returning -- so calling this never costs an image resolution, only how much
    of that capped-resolution deblurring work makes it into the final pixels.

    strength in [0, 1] blends the (size-matched) model output with the input:
    1.0 = full model, 0.0 = untouched.
    """
    img = image.convert("RGB")
    orig_size = img.size

    work = img
    if max_long_side and max(work.size) > max_long_side:
        s = max_long_side / max(work.size)
        work = work.resize((round(work.size[0] * s), round(work.size[1] * s)), Image.LANCZOS)
    w, h = work.size
    work = work.crop((0, 0, w - w % 4, h - h % 4))   # U-Net needs H, W divisible by 4

    src = np.asarray(work, np.float32) / 255.0
    model, device = load_blur_gan_model()
    out = restore_tensor(model, torch.from_numpy(src).permute(2, 0, 1), device)
    out = out.permute(1, 2, 0).numpy()
    restored = Image.fromarray((out * 255).round().clip(0, 255).astype(np.uint8))
    if restored.size != orig_size:
        restored = restored.resize(orig_size, Image.LANCZOS)

    strength = float(min(1.0, max(0.0, strength)))
    if strength >= 1.0:
        return restored
    base = np.asarray(img, np.float32)
    a = np.asarray(restored, np.float32)
    return Image.fromarray((strength * a + (1.0 - strength) * base).round().clip(0, 255).astype(np.uint8))


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    src = Image.open(p)
    dst = restore_image(src)
    save_to = p.rsplit(".", 1)[0] + "_blurgan.png"
    dst.save(save_to)
    print(f"{p}  {src.size} -> {dst.size}   wrote {save_to}   (model present: {available()})")
