"""
Inference for the phase-2b restoration U-Net (models/restore_best_lpips.pt).

The model was trained on 256x256 crops, so a full photo is processed by TILING:
a 256 window slid across the frame, each tile restored, and the overlaps blended
with a raised-cosine window so there are no seams.

Scale note: training crops came from images resized to <=800 px long side, so we
downscale the input's long side to MAX_LONG_SIDE first. The model is blind -- it
takes only the degraded pixels, no defect flags.

Usage (smoke test):
    python src/restore_infer.py path/to/photo.jpg
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from restore_model import RestoreUNet   # same src/ package

CKPT = Path(__file__).resolve().parent.parent / "models" / "restore_best_lpips.pt"

# must match src/train_restore.py CONFIG that produced the checkpoint
BASE_CHANNELS = 48
N_BLOCKS = 3

TILE = 256
STRIDE = 192            # ~25% overlap
MAX_LONG_SIDE = 768     # downscale bigger uploads to near the training scale


def available() -> bool:
    return CKPT.exists()


@lru_cache(maxsize=1)
def load_restore_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RestoreUNet(base_channels=BASE_CHANNELS, n_blocks=N_BLOCKS).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()
    return model, device


@lru_cache(maxsize=1)
def _window(tile: int) -> torch.Tensor:
    """2-D raised-cosine blend weights, (1, tile, tile)."""
    r = torch.hann_window(tile, periodic=False).clamp(min=1e-3)
    return (r[:, None] * r[None, :])[None]


@torch.no_grad()
def restore_tensor(model, img_t: torch.Tensor, device: str,
                   tile: int = TILE, stride: int = STRIDE) -> torch.Tensor:
    """img_t: (3, H, W) float in [0, 1] -> restored (3, H, W) on CPU."""
    _, h, w = img_t.shape
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    if pad_h or pad_w:
        img_t = F.pad(img_t.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect")[0]
    _, h, w = img_t.shape

    def starts(length):
        s = list(range(0, length - tile + 1, stride))
        if not s or s[-1] != length - tile:
            s.append(length - tile)
        return s

    win = _window(tile).to(device)
    img_t = img_t.to(device)
    acc = torch.zeros(3, h, w, device=device)
    wsum = torch.zeros(1, h, w, device=device)
    for y in starts(h):
        for x in starts(w):
            out = model(img_t[:, y:y + tile, x:x + tile].unsqueeze(0))[0]
            acc[:, y:y + tile, x:x + tile] += out * win
            wsum[:, y:y + tile, x:x + tile] += win
    restored = (acc / wsum).clamp(0, 1)
    return restored[:, : h - pad_h, : w - pad_w].cpu()


def restore_image(image: Image.Image, max_long_side: int = MAX_LONG_SIDE,
                  strength: float = 1.0) -> Image.Image:
    """PIL photo (any size) -> restored PIL photo (long side <= max_long_side).

    strength in [0, 1] blends the model output with the (downscaled) input:
    1.0 = full model, 0.0 = untouched. Lower values trade some of the exposure
    fix back for the input's original sharpness and contrast.
    """
    img = image.convert("RGB")
    if max_long_side and max(img.size) > max_long_side:
        s = max_long_side / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
    w, h = img.size
    img = img.crop((0, 0, w - w % 4, h - h % 4))   # U-Net needs H, W divisible by 4

    src = np.asarray(img, np.float32) / 255.0
    model, device = load_restore_model()
    out = restore_tensor(model, torch.from_numpy(src).permute(2, 0, 1), device)
    out = out.permute(1, 2, 0).numpy()

    strength = float(min(1.0, max(0.0, strength)))
    blended = strength * out + (1.0 - strength) * src
    return Image.fromarray((blended * 255).round().clip(0, 255).astype(np.uint8))


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    src = Image.open(p)
    dst = restore_image(src)
    save_to = p.rsplit(".", 1)[0] + "_restored.png"
    dst.save(save_to)
    print(f"{p}  {src.size} -> {dst.size}   wrote {save_to}   (model present: {available()})")
