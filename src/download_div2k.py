"""
Downloads DIV2K's training set: 800 high-resolution images curated specifically
for image-restoration research, and extracts them into data/clean_pool/div2k/.

Unlike the COCO subset, DIV2K images are already curated to be sharp and clean --
no quality filter needed here, just download, extract, done. This is a second,
different source of clean training images for the phase-2b restoration model
(alongside the expanded COCO pull), so the model sees more scene variety than
750 COCO photos alone.

Usage:
    python src/download_div2k.py
"""
import argparse
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"


def download_file(url: str, dest_path: Path, min_expected_bytes: int = 3_000_000_000,
                   max_retries: int = 5):
    """Same resumable-download pattern as download_raw_images.py: continues a
    partial file via an HTTP Range request, retries on connection drops."""
    if dest_path.exists() and dest_path.stat().st_size >= min_expected_bytes:
        print(f"{dest_path.name} already downloaded, skipping.")
        return

    for attempt in range(1, max_retries + 1):
        resume_pos = dest_path.stat().st_size if dest_path.exists() else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

        try:
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0)) + resume_pos
            mode = "ab" if resume_pos > 0 else "wb"

            print(f"{'Resuming' if resume_pos else 'Downloading'} {url} "
                  f"(attempt {attempt}/{max_retries}) ...")
            with open(dest_path, mode) as f, tqdm(
                total=total_size, initial=resume_pos, unit="B", unit_scale=True,
                desc=dest_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            if dest_path.stat().st_size >= min_expected_bytes:
                return
            print(f"Download ended early -- will retry.")

        except (requests.exceptions.RequestException, ConnectionError) as e:
            print(f"Download interrupted on attempt {attempt}/{max_retries}: {e}")
            if attempt == max_retries:
                raise RuntimeError(f"Failed to download after {max_retries} attempts.")
            # Back off before retrying -- a DNS/connection blip often needs a few
            # seconds to clear. The original version retried instantly 5 times,
            # which doesn't help if the blip outlasts a fraction of a second.
            wait = 5 * attempt  # 5s, 10s, 15s, 20s
            print(f"Waiting {wait}s before retrying, then resuming from where it left off ...")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/clean_pool/div2k")
    parser.add_argument("--temp-dir", type=str, default="data/_div2k_temp")
    args = parser.parse_args()

    temp_dir = Path(args.temp_dir)
    output_dir = Path(args.output_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = temp_dir / "DIV2K_train_HR.zip"

    download_file(DIV2K_URL, zip_path)

    # The zip contains one top-level folder (DIV2K_train_HR/*.png). We extract it
    # straight into output_dir -- restore_dataset.py's file lister recurses into
    # subfolders, so the nested structure doesn't need flattening.
    marker = output_dir / "DIV2K_train_HR"
    if marker.exists() and any(marker.glob("*.png")):
        print(f"{marker} already contains images, skipping extraction.")
    else:
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)

    n_images = len(list(output_dir.rglob("*.png")))
    print(f"\nDone. {n_images} DIV2K images in {output_dir}/")
    print(f"You can delete the temp zip to save disk space: rm -rf {temp_dir}")


if __name__ == "__main__":
    main()
