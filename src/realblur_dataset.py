"""
Real paired blur/sharp fine-tuning data -- RealBlur-J (rimchang/RealBlur, CC BY
4.0, https://github.com/rimchang/RealBlur).

Unlike RestoreDataset (src/restore_dataset.py), which synthesizes blur on the
fly from clean images via src/degrade.py, every pair here is a REAL photo: the
same scene shot through a beam-splitter rig, one long exposure (genuine
camera-shake blur) and one short (sharp reference), pre-aligned pixel-for-pixel
by the dataset's own ECC + intensity-correction processing -- so a plain
same-coordinates crop from both sides stays aligned, no extra registration
needed. Used to fine-tune the already-COCO-trained blur-GAN checkpoint onto
real blur statistics -- a further fine-tune on top of that training, not a
replacement for it.

Expects the dataset laid out as extracted from the RealBlur Google Drive
archive (RealBlur.tar.gz), so that:

    <root>/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/sceneNNN/gt/gt_M.png
    <root>/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/sceneNNN/blur/blur_M.png

paired via the dataset authors' own list files (data/realblur_lists/
RealBlur_J_{train,test}_list.txt, pulled from rimchang/SRN-Deblur's datalist/),
each line "<gt_path> <blur_path>" relative to <root>.

Usage (smoke test -- writes a few pairs to outputs/):
    python src/realblur_dataset.py path/to/extracted/realblur/root
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PATCH = 256

_LISTS = Path(__file__).resolve().parent.parent / "data" / "realblur_lists"
TRAIN_LIST = _LISTS / "RealBlur_J_train_list.txt"
TEST_LIST = _LISTS / "RealBlur_J_test_list.txt"


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _to_tensor(img_u8: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 RGB -> float32 CxHxW in [0,1]."""
    t = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1).float()
    return t / 255.0


def _ensure_min_size_pair(gt, blur, size):
    """Scale BOTH sides by the same factor if either is smaller than `size` --
    a single-side resize would break the pixel alignment between them."""
    h, w = gt.shape[:2]
    if min(h, w) >= size:
        return gt, blur
    scale = size / min(h, w)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    resize = lambda im: cv2.resize(im, new_size, interpolation=cv2.INTER_CUBIC)
    return resize(gt), resize(blur)


def _parse_list(list_path: Path, root: Path):
    pairs = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        gt_rel, blur_rel = line.split()[:2]
        pairs.append((root / gt_rel, root / blur_rel))
    return pairs


class RealBlurDataset(Dataset):
    """Each sample: (blur 256x256 crop, gt 256x256 crop). The crop region and,
    in train mode, the flip/rotation augmentation are applied IDENTICALLY to
    both sides -- there's no degrade_fn regenerating one side from the other
    here, both images are real photos that have to move together or the pair
    stops being aligned."""

    def __init__(self, pairs, mode: str = "train", patch: int = PATCH, val_seed: int = 1234):
        assert mode in ("train", "val")
        self.pairs = pairs
        self.mode = mode
        self.patch = patch
        self.val_seed = val_seed

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        gt_path, blur_path = self.pairs[idx]
        gt = _load_rgb(gt_path)
        blur = _load_rgb(blur_path)
        if blur.shape[:2] != gt.shape[:2]:   # shouldn't happen, pairs are pre-aligned -- guard anyway
            blur = cv2.resize(blur, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        gt, blur = _ensure_min_size_pair(gt, blur, self.patch)

        rng = (np.random.default_rng() if self.mode == "train"
               else np.random.default_rng(self.val_seed + idx))

        h, w = gt.shape[:2]
        if self.mode == "train":
            y = int(rng.integers(0, h - self.patch + 1))
            x = int(rng.integers(0, w - self.patch + 1))
        else:                                # deterministic centre crop
            y, x = (h - self.patch) // 2, (w - self.patch) // 2
        gt_crop = gt[y:y + self.patch, x:x + self.patch]
        blur_crop = blur[y:y + self.patch, x:x + self.patch]

        if self.mode == "train":             # same flip/rotation on BOTH sides
            if rng.random() < 0.5:
                gt_crop, blur_crop = np.fliplr(gt_crop), np.fliplr(blur_crop)
            if rng.random() < 0.5:
                gt_crop, blur_crop = np.flipud(gt_crop), np.flipud(blur_crop)
            k = int(rng.integers(0, 4))
            gt_crop, blur_crop = np.rot90(gt_crop, k), np.rot90(blur_crop, k)
            gt_crop = np.ascontiguousarray(gt_crop)
            blur_crop = np.ascontiguousarray(blur_crop)

        return _to_tensor(blur_crop), _to_tensor(gt_crop)


def build_realblur_loaders(root, batch_size: int = 8, val_frac: float = 0.08,
                           num_workers: int = 0, seed: int = 42, patch: int = PATCH,
                           max_train_pairs: int | None = None):
    """RealBlur-J's own train list, further split train/val; its own test list
    stays a sealed held-out split, same pattern as build_restore_loaders.

    max_train_pairs caps how many of the 3,757 official train pairs to use --
    a lever for keeping a first fine-tune run fast on a small local GPU."""
    root = Path(root)
    train_pairs = _parse_list(TRAIN_LIST, root)
    test_pairs = _parse_list(TEST_LIST, root)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_pairs))
    if max_train_pairs:
        perm = perm[:max_train_pairs]
    n_val = max(1, int(len(perm) * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = RealBlurDataset([train_pairs[i] for i in train_idx], mode="train", patch=patch)
    val_ds = RealBlurDataset([train_pairs[i] for i in val_idx], mode="val", patch=patch)
    test_ds = RealBlurDataset(test_pairs, mode="val", patch=patch)   # official sealed test split

    def _dl(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(), drop_last=shuffle)

    return {"train": _dl(train_ds, True), "val": _dl(val_ds, False), "test": _dl(test_ds, False)}


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/realblur"

    loaders = build_realblur_loaders(root, batch_size=8)
    for split, ldr in loaders.items():
        blur, gt = next(iter(ldr))
        print(f"{split:5s} {len(ldr.dataset):5d} pairs | "
              f"blur {tuple(blur.shape)} {blur.dtype} [{blur.min():.2f},{blur.max():.2f}] | "
              f"gt {tuple(gt.shape)} [{gt.min():.2f},{gt.max():.2f}]")

    out_dir = Path("outputs/realblur_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)
    blur, gt = next(iter(loaders["train"]))
    for i in range(min(6, blur.shape[0])):
        b = (blur[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        g = (gt[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pair = np.concatenate([b, g], axis=1)   # [blur | gt]
        cv2.imwrite(str(out_dir / f"pair_{i}.png"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
    print(f"wrote 6 [blur | gt] pairs to {out_dir}/")
