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


def download_file(url: str, dest_path: Path, min_expected_bytes: int = 500_000_000,
                   max_retries: int = 5):
    """Downloads with resume support: if a previous attempt left a partial
    file, this continues from where it left off using an HTTP Range request
    instead of starting over. Retries automatically on connection drops,
    which are common on slow/unstable connections for a file this large."""
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if actual_size >= min_expected_bytes:
            print(f"{dest_path.name} already downloaded ({actual_size / 1e6:.0f} MB), skipping.")
            return

    for attempt in range(1, max_retries + 1):
        resume_pos = dest_path.stat().st_size if dest_path.exists() else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

        try:
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()

            # Total size: for a resumed (206 Partial Content) request, add
            # what we already have to what's left to download.
            total_size = int(response.headers.get("content-length", 0)) + resume_pos
            mode = "ab" if resume_pos > 0 else "wb"

            if resume_pos > 0:
                print(f"Resuming download from {resume_pos / 1e6:.0f} MB "
                      f"(attempt {attempt}/{max_retries}) ...")
            else:
                print(f"Downloading {url} (attempt {attempt}/{max_retries}) ...")

            with open(dest_path, mode) as f, tqdm(
                total=total_size, initial=resume_pos, unit="B", unit_scale=True,
                desc=dest_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            final_size = dest_path.stat().st_size
            if final_size >= min_expected_bytes:
                return  # success
            else:
                print(f"Download ended early ({final_size / 1e6:.1f} MB) — will retry.")

        except (requests.exceptions.RequestException, ConnectionError) as e:
            print(f"Download interrupted on attempt {attempt}/{max_retries}: {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download after {max_retries} attempts. "
                    "Consider downloading manually in a browser instead (see notes below)."
                )
            print("Retrying, resuming from where it left off ...")


def extract_zip(zip_path: Path, extract_to: Path, expected_subfolder: str = "val2017"):
    """Extracts the zip into extract_to. Checks specifically for the expected
    image subfolder (not just any file in extract_to) to decide whether
    extraction already happened — extract_to also holds the zip file itself,
    so a naive 'is this folder non-empty' check would always skip extraction."""
    expected_dir = extract_to / expected_subfolder

    if expected_dir.exists() and any(expected_dir.glob("*.jpg")):
        print(f"{expected_dir} already contains images, skipping extraction.")
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