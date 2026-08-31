"""
Turns the clean baseline images in data/raw/ into the synthetic defect dataset
in data/synthetic/.

For every raw image we write SIX output images:
  - one clean pass-through copy   -> label row of all zeros
  - one blurred version           -> blur=1
  - one underexposed version      -> underexposed=1
  - one overexposed version       -> overexposed=1
  - one noisy version             -> noise=1
  - one low-contrast version      -> contrast=1

So 750 raw images produce ~4,500 synthetic images (750 clean + 750 x 5 defects).
Every generated file gets one row in data/synthetic/labels.csv with a binary
column per defect class.

Severity (how strong each defect is) is randomised within a per-defect range so
the model sees mild and severe examples of each defect, not one fixed amount.
All randomness comes from a single seeded numpy Generator, so re-running with the
same --seed reproduces a byte-identical dataset.

Usage:
    python src/generate_synthetic.py                 # full run over data/raw/
    python src/generate_synthetic.py --limit 5       # quick test on 5 images
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# The defect classes, in the fixed column order used everywhere downstream
# (labels CSV, model outputs, evaluation). Change this list in ONE place if the
# class set ever changes.
DEFECT_COLUMNS = ["blur", "underexposed", "overexposed", "noise", "contrast"]


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Clamp a floating-point image back into the valid 0-255 range and cast it
    to uint8 (the dtype OpenCV expects for saving).

    Every transform below does its maths in float, which can push pixel values
    above 255 or below 0. If we cast straight to uint8 without clamping, values
    WRAP AROUND (256 -> 0), so an over-bright pixel would turn black. np.clip
    pins anything out of range to the nearest edge, which is also physically what
    a real sensor does when it saturates or bottoms out."""
    return np.clip(arr, 0, 255).astype(np.uint8)


def apply_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Out-of-focus look via a Gaussian blur: each pixel becomes a weighted
    average of its neighbours, with nearer neighbours weighted more heavily.
    `sigma` is the width of that weighting curve in pixels -- bigger sigma spreads
    the average over a wider area, so the image looks more smeared."""
    sigma = rng.uniform(1.0, 4.0)
    # ksize=(0, 0) tells OpenCV to derive the kernel size from sigma itself,
    # so we only have to reason about one knob.
    blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return blurred  # GaussianBlur keeps uint8 in -> uint8 out, no _to_uint8 needed


def apply_underexposure(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Too-little-light look: scale every pixel DOWN by a constant factor < 1.
    Multiplying (rather than subtracting a constant) mirrors how a real camera in
    low light collects proportionally less light everywhere, and keeps true black
    at 0 instead of lifting or crushing the shadows unnaturally."""
    factor = rng.uniform(0.3, 0.6)
    darker = img.astype(np.float32) * factor
    return _to_uint8(darker)


def apply_overexposure(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blown-out look: scale every pixel UP by a factor > 1. Values that would
    exceed 255 are clipped to 255 by _to_uint8 -- those flat white regions with
    no detail ARE the overexposed effect, matching a saturated sensor."""
    factor = rng.uniform(1.6, 2.6)
    brighter = img.astype(np.float32) * factor
    return _to_uint8(brighter)


def apply_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """High-ISO grain: add an independent random value to every pixel (and every
    colour channel), drawn from a bell curve centred on 0. Mean 0 means a pixel
    is equally likely to be nudged brighter or darker; `sigma` is how big those
    nudges typically are, so larger sigma = coarser grain."""
    sigma = rng.uniform(8.0, 30.0)
    # rng.normal fills an array the same shape as the image with noise samples.
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape)
    noisy = img.astype(np.float32) + noise
    return _to_uint8(noisy)


def apply_low_contrast(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flat / washed-out look: pull every pixel toward the image's average
    brightness. `mean` is a single number (the grand average over all pixels and
    channels). (pixel - mean) is how far that pixel sits from average; scaling
    that gap by factor < 1 shrinks the spread of values, which is exactly what
    "low contrast" means. factor = 1 leaves the image unchanged; factor = 0 would
    collapse it to a solid grey."""
    factor = rng.uniform(0.3, 0.6)
    mean = img.astype(np.float32).mean()
    flattened = mean + (img.astype(np.float32) - mean) * factor
    return _to_uint8(flattened)


# Registry: map each defect's column name to the function that produces it.
# The main loop iterates over this dict, so it never mentions individual defects
# by name -- add or remove a class here and everything else follows.
DEFECT_FUNCS = {
    "blur": apply_blur,
    "underexposed": apply_underexposure,
    "overexposed": apply_overexposure,
    "noise": apply_noise,
    "contrast": apply_low_contrast,
}


def zero_label_row(filename: str) -> dict:
    """Build a label row with every defect column set to 0. Callers flip the one
    column they need to 1 (or leave all 0 for a clean image)."""
    row = {col: 0 for col in DEFECT_COLUMNS}
    row["filename"] = filename
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                        help="Folder of clean baseline images to corrupt")
    parser.add_argument("--out-dir", type=str, default="data/synthetic",
                        help="Where to write generated images + labels.csv")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the severity RNG; same seed = same dataset")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N raw images (for quick tests)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One seeded generator, created once and threaded into every transform call.
    # Because the raw images are processed in sorted order and each transform
    # draws from this same generator in the same sequence, the whole run is
    # reproducible from the seed alone.
    rng = np.random.default_rng(args.seed)

    # Sorted so the processing order (and therefore the RNG draw order) is stable
    # regardless of how the filesystem lists the directory.
    raw_paths = sorted(raw_dir.glob("raw_*.jpg"))
    if args.limit is not None:
        raw_paths = raw_paths[:args.limit]

    if not raw_paths:
        raise SystemExit(f"No raw images found in {raw_dir}/ (expected files like raw_0000.jpg)")

    label_rows = []  # accumulates one dict per generated image; written to CSV at the end

    for raw_path in tqdm(raw_paths, desc="Generating synthetic defects"):
        img = cv2.imread(str(raw_path))  # loads as a HxWx3 uint8 BGR array
        if img is None:
            print(f"Skipping unreadable file: {raw_path}")
            continue

        base = raw_path.stem  # e.g. "raw_0000" (filename without extension)

        # 1. Clean pass-through: re-save the untouched image as PNG so the model
        #    also learns what "no defect" looks like. Label row is all zeros.
        clean_name = f"{base}_clean.png"
        cv2.imwrite(str(out_dir / clean_name), img)
        label_rows.append(zero_label_row(clean_name))

        # 2. One image per defect class.
        for defect_name, transform in DEFECT_FUNCS.items():
            degraded = transform(img, rng)
            out_name = f"{base}_{defect_name}.png"
            cv2.imwrite(str(out_dir / out_name), degraded)

            row = zero_label_row(out_name)
            row[defect_name] = 1  # only this defect is present
            label_rows.append(row)

    # Write every collected row to a single CSV. newline="" is the documented
    # requirement for the csv module on Windows so it doesn't insert blank lines.
    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"] + DEFECT_COLUMNS)
        writer.writeheader()
        writer.writerows(label_rows)

    print(f"\nDone. {len(label_rows)} images written to {out_dir}/")
    print(f"Labels: {csv_path}")


if __name__ == "__main__":
    main()
