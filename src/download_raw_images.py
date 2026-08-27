"""
Downloads the COCO val2017 image set, randomly samples a diverse subset,
filters out images that are already too dark/blurry/small (since our raw
images need to be clean baselines before we deliberately corrupt them),
and copies the selected images into data/raw/.

Usage:
    python src/download_raw_images.py --num-images 750
"""
import argparse
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm

COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"


def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        print(f"{dest_path.name} already downloaded, skipping.")
        return

    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path):
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"{extract_to} already extracted, skipping.")
        return

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def passes_quality_check(image_path: Path, min_size=300, min_brightness=40,
                          max_brightness=220, min_sharpness=60):
    """Filters out images that are already too dark, too bright, too blurry,
    or too small — we want CLEAN baseline images so our synthetic defects
    are the only source of degradation, not pre-existing quality issues."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    h, w = img.shape[:2]
    if min(h, w) < min_size:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()
    if brightness < min_brightness or brightness > max_brightness:
        return False

    # Laplacian variance is a standard cheap sharpness proxy — low variance
    # means few sharp edges, i.e. the image is already blurry.
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness < min_sharpness:
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=750,
                         help="Number of raw images to select")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--temp-dir", type=str, default="data/_coco_temp")
    parser.add_argument("--skip-quality-check", action="store_true",
                         help="Skip filtering already-poor-quality images (faster, less strict)")
    args = parser.parse_args()

    random.seed(args.seed)

    temp_dir = Path(args.temp_dir)
    output_dir = Path(args.output_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = temp_dir / "val2017.zip"
    extract_dir = temp_dir / "val2017"

    # 1. Download
    download_file(COCO_VAL2017_URL, zip_path)

    # 2. Extract
    extract_zip(zip_path, temp_dir)

    # 3. Sample and filter
    all_images = sorted(extract_dir.glob("*.jpg"))
    if not all_images:
        # COCO's zip structure sometimes nests one level differently
        all_images = sorted(temp_dir.glob("**/*.jpg"))

    print(f"Found {len(all_images)} candidate images in COCO val2017.")

    random.shuffle(all_images)

    selected = []
    checked = 0

    for img_path in tqdm(all_images, desc="Selecting images"):
        if len(selected) >= args.num_images:
            break
        checked += 1

        if args.skip_quality_check or passes_quality_check(img_path):
            selected.append(img_path)

    print(f"Checked {checked} images, selected {len(selected)} that passed quality filtering.")

    if len(selected) < args.num_images:
        print(f"Warning: only found {len(selected)} images passing quality checks "
              f"(wanted {args.num_images}). Consider --skip-quality-check or lowering thresholds.")

    # 4. Copy to data/raw/
    for i, img_path in enumerate(tqdm(selected, desc="Copying to data/raw/")):
        dest_name = f"raw_{i:04d}{img_path.suffix}"
        shutil.copy(img_path, output_dir / dest_name)

    print(f"\nDone. {len(selected)} raw images saved to {output_dir}/")
    print(f"You can now delete the temp folder to save disk space: rm -rf {temp_dir}")


if __name__ == "__main__":
    main()