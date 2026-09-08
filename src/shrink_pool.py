"""
One-time: shrink the clean training pool so on-the-fly data loading is fast.

The restoration model only ever sees 256x256 crops, so keeping ~2000px source
images (DIV2K) just makes every __getitem__ decode a multi-MB file to pull a tiny
crop out of it -- that starves the GPU (seen at ~27% utilisation). This resizes
every image so its long side is at most MAX_SIDE px (smaller images are copied
unchanged), into data/clean_pool_small/, keeping the coco/ and div2k/ subfolders
and each file's original format.

Usage:
    python src/shrink_pool.py
"""
from pathlib import Path

import cv2
from tqdm import tqdm

SRC = Path("data/clean_pool")
DST = Path("data/clean_pool_small")
MAX_SIDE = 800
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main():
    files = [p for p in SRC.rglob("*") if p.suffix.lower() in EXTS]
    if not files:
        raise SystemExit(f"no images under {SRC}/ -- run the download scripts first")
    print(f"{len(files)} images -> {DST}/  (long side <= {MAX_SIDE}px)")

    n_resized = n_copied = n_skip = 0
    for p in tqdm(files):
        out = DST / p.relative_to(SRC)          # keep subfolder + original extension
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            n_skip += 1
            continue

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            n_resized += 1
        else:
            n_copied += 1
        cv2.imwrite(str(out), img)

    total = len([p for p in DST.rglob("*") if p.suffix.lower() in EXTS])
    print(f"\ndone. {n_resized} resized, {n_copied} copied, {n_skip} already present. "
          f"{total} images in {DST}/")
    print("Point build_restore_loaders at "
          "['data/clean_pool_small/coco', 'data/clean_pool_small/div2k'].")


if __name__ == "__main__":
    main()
