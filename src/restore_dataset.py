"""
Training data for the phase-2b restoration model.

Each sample is a pair: (degraded 256x256 crop, clean 256x256 crop). The clean crop
is the target; the degraded crop is made on the fly by src/degrade.py, so the
model sees a fresh random degradation every time an image comes round.

  RestoreDataset       - one clean-image folder(s), mode "train" or "val"
  build_restore_loaders - split the clean images into train/val, return DataLoaders

train mode : random crop, random flip/rotate, fresh random degradation each call,
             ~12% of pairs are identity (degraded == clean) so the model learns
             not to touch good photos.
val mode   : deterministic centre crop, no augmentation, degradation seeded by the
             sample index so the val pairs are identical every epoch.

Usage (smoke test -- writes a few training pairs to outputs/):
    python src/restore_dataset.py
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from degrade import degrade  # same src/ package; callers put src/ on sys.path

PATCH = 256          # crop size fed to the model (speed/quality knob, revisit later)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _load_rgb(path: Path) -> np.ndarray:
    """Read an image file as uint8 RGB HxWx3 (OpenCV reads BGR)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _ensure_min_size(img: np.ndarray, size: int) -> np.ndarray:
    """If either side is smaller than `size`, scale up so the short side == size."""
    h, w = img.shape[:2]
    if min(h, w) >= size:
        return img
    scale = size / min(h, w)
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_CUBIC)


def _to_tensor(img_u8: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 RGB  ->  float32 CxHxW in [0,1]."""
    t = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1).float()
    return t / 255.0


class RestoreDataset(Dataset):
    def __init__(self, paths, mode: str = "train", patch: int = PATCH,
                 p_identity: float = 0.12, val_seed: int = 1234):
        assert mode in ("train", "val")
        self.paths = [Path(p) for p in paths]
        self.mode = mode
        self.patch = patch
        self.p_identity = p_identity
        self.val_seed = val_seed

    def __len__(self) -> int:
        return len(self.paths)

    def _crop(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        img = _ensure_min_size(img, self.patch)
        h, w = img.shape[:2]
        if self.mode == "train":
            y = int(rng.integers(0, h - self.patch + 1))
            x = int(rng.integers(0, w - self.patch + 1))
        else:                                   # deterministic centre crop
            y, x = (h - self.patch) // 2, (w - self.patch) // 2
        return img[y:y + self.patch, x:x + self.patch]

    def _augment(self, crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Flips + 90-degree rotations only -- safe because restoration is
        geometry-agnostic. Applied to the CLEAN crop, before degradation, so the
        pair stays pixel-aligned."""
        if rng.random() < 0.5:
            crop = np.fliplr(crop)
        if rng.random() < 0.5:
            crop = np.flipud(crop)
        crop = np.rot90(crop, k=int(rng.integers(0, 4)))
        return np.ascontiguousarray(crop)

    def __getitem__(self, idx: int):
        clean_full = _load_rgb(self.paths[idx])

        if self.mode == "train":
            rng = np.random.default_rng()        # fresh entropy -> new degradation every call
            clean = self._augment(self._crop(clean_full, rng), rng)
            if rng.random() < self.p_identity:
                degraded = clean.copy()          # identity pair: "leave good photos alone"
            else:
                degraded = degrade(clean, rng)
        else:
            rng = np.random.default_rng(self.val_seed + idx)  # stable pair every epoch
            clean = self._crop(clean_full, rng)
            degraded = degrade(clean, rng)

        return _to_tensor(degraded), _to_tensor(clean)


def _list_images(dirs):
    out = []
    for d in ([dirs] if isinstance(dirs, (str, Path)) else dirs):
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return out


def build_restore_loaders(clean_dirs, batch_size: int = 16, val_frac: float = 0.08,
                          test_frac: float = 0.06, num_workers: int = 0, seed: int = 42,
                          patch: int = PATCH):
    """Split clean images file-level into train/val/test (no image in more than one)
    and return {"train": loader, "val": loader, "test": loader}. File-level is
    enough here -- unlike the classifier there are no per-scene variants to leak.

    test is a SEALED split: deterministic degradation (like val), touched only once
    after all tuning is done, for the final synthetic-benchmark numbers.
    """
    paths = _list_images(clean_dirs)
    if not paths:
        raise SystemExit(f"no images found under {clean_dirs}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(paths))
    n_val = max(1, int(len(paths) * val_frac))
    n_test = max(1, int(len(paths) * test_frac))
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]

    train_ds = RestoreDataset([paths[i] for i in train_idx], mode="train", patch=patch)
    val_ds = RestoreDataset([paths[i] for i in val_idx], mode="val", patch=patch)
    test_ds = RestoreDataset([paths[i] for i in test_idx], mode="val", patch=patch)  # sealed = deterministic like val

    def _dl(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(), drop_last=shuffle)

    return {"train": _dl(train_ds, True), "val": _dl(val_ds, False), "test": _dl(test_ds, False)}


if __name__ == "__main__":
    loaders = build_restore_loaders("data/raw", batch_size=8)
    for split, ldr in loaders.items():
        deg, clean = next(iter(ldr))
        print(f"{split:5s} {len(ldr.dataset):5d} images | "
              f"deg {tuple(deg.shape)} {deg.dtype} [{deg.min():.2f},{deg.max():.2f}] | "
              f"clean {tuple(clean.shape)} [{clean.min():.2f},{clean.max():.2f}]")

    # save 6 training pairs stacked side by side for eyeballing
    out_dir = Path("outputs/restore_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)
    deg, clean = next(iter(loaders["train"]))
    for i in range(min(6, deg.shape[0])):
        d = (deg[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        c = (clean[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pair = np.concatenate([d, c], axis=1)  # [degraded | clean]
        cv2.imwrite(str(out_dir / f"pair_{i}.png"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
    print(f"wrote 6 [degraded | clean] pairs to {out_dir}/")
