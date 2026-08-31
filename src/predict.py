"""
Inference for the frozen photo-quality classifier.

Turns a PIL image of any size into per-defect probabilities by TILING. The model
only ever saw 256x256 crops during training, so at inference we slide a 256x256
window across the whole image, run the model on every tile, and combine the
per-tile predictions.

Aggregation follows the defect taxonomy:
  - blur is LOCAL (a soft background, one blurred moving object) -> take the MAX
    tile probability: "is there blur anywhere in the frame?"
  - underexposed / overexposed / noise / contrast are GLOBAL -> take the MEAN
    tile probability: a global property shows up in every tile, and one odd tile
    should not flip the call.

Scale note: training crops came from images resized to short-side 256 (no scale
augmentation), so tiles must stay near that scale. We resize the upload's short
side to 320 -> each 256 tile is only ~1.25x zoomed vs training.

Frozen model: models/traincombo_best.pt (iteration 4). Threshold 0.5.

Usage (smoke test):
    python src/predict.py [path/to/image]
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import BaselineCNN  # same src/ package; callers put src/ on sys.path

DEFECT_COLUMNS = ["blur", "underexposed", "overexposed", "noise", "contrast"]
CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "traincombo_best.pt"

TILE = 256          # window size the model expects
SHORT_SIDE = 320    # resize the image's short side to this before tiling
STRIDE = 96         # window step (~60% overlap)
MAX_TILES = 16      # safety cap for extreme aspect ratios
THRESHOLD = 0.5

# True where the class aggregates by MAX across tiles (blur); False -> MEAN.
_AGG_MAX = np.array([c == "blur" for c in DEFECT_COLUMNS])

# Per-tile pixel prep: PIL crop -> float tensor (3,256,256) in [0,1]. Same as the
# training transform's final step.
_to_tensor = transforms.ToTensor()


@lru_cache(maxsize=1)
def load_model():
    """Load the frozen weights once. lru_cache means every later call hands back
    the same in-memory model instead of re-reading the checkpoint file."""
    model = BaselineCNN()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()  # inference mode
    return model


def _resize_short_side(img: Image.Image, target: int) -> Image.Image:
    """Scale so the shorter side == target, keeping aspect ratio. Never let either
    side fall below one tile."""
    w, h = img.size
    scale = target / min(w, h)
    new_w = max(TILE, round(w * scale))
    new_h = max(TILE, round(h * scale))
    return img.resize((new_w, new_h), Image.BILINEAR)


def _tile_starts(length: int, tile: int, stride: int) -> list:
    """Top-left offsets along one axis, always ending flush with the edge so the
    border is covered."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _make_tiles(img: Image.Image) -> list:
    img = _resize_short_side(img, SHORT_SIDE)
    w, h = img.size
    xs = _tile_starts(w, TILE, STRIDE)
    ys = _tile_starts(h, TILE, STRIDE)
    tiles = [img.crop((x, y, x + TILE, y + TILE)) for y in ys for x in xs]
    if len(tiles) > MAX_TILES:
        keep = np.linspace(0, len(tiles) - 1, MAX_TILES).round().astype(int)
        tiles = [tiles[i] for i in keep]
    return tiles


@torch.no_grad()
def predict(img: Image.Image) -> dict:
    """PIL image (any size) -> dict:
        probs     {defect: aggregated probability 0-1}
        flags     {defect: bool, True if probability >= 0.5}
        n_tiles   how many tiles were scored
        per_tile  {defect: [probability per tile]}  (for the UI's detail view)
    """
    img = img.convert("RGB")  # PIL is RGB; matches dataset.py
    tiles = _make_tiles(img)

    batch = torch.stack([_to_tensor(t) for t in tiles])   # (n_tiles, 3, 256, 256)
    logits = load_model()(batch)
    probs = torch.sigmoid(logits).numpy()                 # (n_tiles, 5)

    agg = np.where(_AGG_MAX, probs.max(axis=0), probs.mean(axis=0))  # (5,)

    return {
        "probs": {c: float(p) for c, p in zip(DEFECT_COLUMNS, agg)},
        "flags": {c: bool(p >= THRESHOLD) for c, p in zip(DEFECT_COLUMNS, agg)},
        "n_tiles": len(tiles),
        "per_tile": {c: [float(v) for v in probs[:, i]] for i, c in enumerate(DEFECT_COLUMNS)},
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        im = Image.open(sys.argv[1])
        src = sys.argv[1]
    else:
        im = Image.fromarray(np.random.randint(0, 256, (700, 1000, 3), dtype=np.uint8))
        src = "<random noise>"
    out = predict(im)
    print(f"{src}  ({im.size[0]}x{im.size[1]})  ->  {out['n_tiles']} tiles")
    for c in DEFECT_COLUMNS:
        flag = "  <-- flagged" if out["flags"][c] else ""
        print(f"  {c:<13} {out['probs'][c]:.3f}{flag}")
