"""
Generates the multi-defect ("combo") images for the validation and test sets and
appends them to data/synthetic/labels.csv.

Why this exists (from the project plan): training data is deliberately kept to one
defect per image. But real photos stack defects -- low light raises sensor noise,
darkness hides softness. We don't KNOW the model generalises from single-defect
training to multi-defect photos, so val/test get a small batch of genuine combos
to measure it empirically instead of assuming.

Design (settled with discussion):
  - 50 combos in val + 50 in test, each from a different randomly-chosen scene
    that is ALREADY assigned to that split (so no split boundary is crossed).
  - 60% pairs / 40% triples  ->  30 pairs + 20 triples per split.
  - Valid combos exclude any that contain BOTH `underexposed` and `overexposed`
    (an image can't be globally too dark and too bright). -> 9 pairs, 7 triples.
  - Round-robin over a shuffled list of valid combos, so each combo type gets
    roughly equal coverage and none gets zero.
  - The two/three transforms are applied in a fixed CANONICAL ORDER that mirrors
    a real camera: tonal changes (contrast, exposure) first, then lens blur, then
    sensor noise LAST. Adding noise before blur would smear the grain and look
    fake.
  - Transform functions are imported from generate_synthetic.py, not reimplemented
    -- that's exactly why they were written as standalone image->image functions.
  - New rows are appended to labels.csv with a multi-hot label and the scene's
    existing `split` value. Files are named like
    `raw_0700_combo_blur_noise.png`.

Run AFTER split_dataset.py (it needs the `split` column). Run once; re-running is
refused unless --force (which first deletes the previous combo rows + files).

Usage:
    python src/generate_combos.py
    python src/generate_combos.py --force
"""
import argparse
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Reuse the exact degradation functions used for the single-defect dataset.
from generate_synthetic import DEFECT_FUNCS, DEFECT_COLUMNS

# Order in which stacked defects are applied. Tonal first, optics next, sensor
# noise last. This is NOT the CSV column order (that's DEFECT_COLUMNS); it's the
# physically-motivated application order, and it's also the order used in filenames.
CANONICAL_ORDER = ["contrast", "underexposed", "overexposed", "blur", "noise"]

# A combo is invalid if it asks for both of these at once.
CONTRADICTORY = {"underexposed", "overexposed"}


def valid_combos(size: int) -> list[tuple]:
    """All defect combinations of the given size, in canonical order, minus the
    physically impossible ones. size=2 -> 9 combos, size=3 -> 7 combos."""
    out = []
    for combo in combinations(CANONICAL_ORDER, size):  # already in canonical order
        if CONTRADICTORY.issubset(combo):
            continue
        out.append(combo)
    return out


def round_robin(items: list, n: int, rng: np.random.Generator) -> list:
    """Return n picks from `items`, cycling through a shuffled copy so every item
    is used about n/len(items) times and none is skipped."""
    shuffled = list(rng.permutation(np.array(items, dtype=object)))
    return [shuffled[i % len(shuffled)] for i in range(n)]


def apply_combo(img: np.ndarray, combo: tuple, rng: np.random.Generator) -> np.ndarray:
    """Apply each defect in `combo` (already canonical order) one after another,
    feeding the output of one transform straight into the next."""
    out = img
    for defect_name in combo:
        out = DEFECT_FUNCS[defect_name](out, rng)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-csv", type=str, default="data/synthetic/labels.csv")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-split", type=int, default=50,
                        help="Combo images to make for each of val and test")
    parser.add_argument("--n-train", type=int, default=0,
                        help="Combo images to add to the TRAIN split (0 = none). "
                             "These teach the model that defects stack; val/test stay a held-out probe.")
    parser.add_argument("--pair-frac", type=float, default=0.60,
                        help="Fraction of a split's combos that are 2-defect (rest are 3-defect)")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing combo rows/files for the splits being generated, then redo them")
    args = parser.parse_args()

    csv_path = Path(args.labels_csv)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found -- run generate_synthetic.py then split_dataset.py first.")

    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise SystemExit("labels.csv has no `split` column -- run split_dataset.py first.")

    is_combo = df["filename"].str.contains("_combo_", regex=False)

    # --- Decide which splits to (re)generate -------------------------------------
    # A split is a "job" if it has no combos yet (or --force). This makes adding
    # train combos later a purely additive step that never disturbs the existing
    # val/test probe set.
    jobs = []  # list of (split_name, n_combos)
    for split_name, n_combos in [("val", args.per_split),
                                 ("test", args.per_split),
                                 ("train", args.n_train)]:
        if n_combos <= 0:
            continue
        already = bool((is_combo & (df["split"] == split_name)).any())
        if already and not args.force:
            print(f"{split_name}: combos already present, skipping (use --force to redo).")
            continue
        jobs.append((split_name, n_combos))

    if not jobs:
        raise SystemExit("Nothing to do.")

    # --force: drop existing combo rows/files only for the splits we're redoing.
    if args.force:
        redo = is_combo & df["split"].isin([s for s, _ in jobs])
        for fname in df.loc[redo, "filename"]:
            (out_dir / fname).unlink(missing_ok=True)
        if redo.any():
            print(f"--force: removed {int(redo.sum())} existing combo rows/files "
                  f"for {sorted({s for s, _ in jobs})}.")
        df = df[~redo].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    pairs = valid_combos(2)    # 9
    triples = valid_combos(3)  # 7

    new_rows = []
    made = {}  # split -> (n_pairs, n_triples)

    for split_name, n_combos in jobs:
        n_pairs = round(n_combos * args.pair_frac)
        n_triples = n_combos - n_pairs
        made[split_name] = (n_pairs, n_triples)

        # Unique scenes in this split. sorted() so the pre-shuffle order is
        # deterministic regardless of CSV row order.
        scenes = sorted(df.loc[df["split"] == split_name, "filename"]
                        .str.extract(r"^(raw_\d+)_")[0].unique())
        if len(scenes) < n_combos:
            raise SystemExit(f"{split_name} has only {len(scenes)} scenes, need {n_combos}.")

        # Pick scenes, then split them into pair-scenes / triple-scenes.
        chosen = rng.choice(scenes, size=n_combos, replace=False)
        rng.shuffle(chosen)
        pair_scenes, triple_scenes = chosen[:n_pairs], chosen[n_pairs:]

        pair_assign = round_robin(pairs, n_pairs, rng)
        triple_assign = round_robin(triples, n_triples, rng)

        for scene, combo in list(zip(pair_scenes, pair_assign)) + list(zip(triple_scenes, triple_assign)):
            img = cv2.imread(str(raw_dir / f"{scene}.jpg"))
            if img is None:
                raise SystemExit(f"Could not read {raw_dir / f'{scene}.jpg'}")

            degraded = apply_combo(img, combo, rng)
            fname = f"{scene}_combo_{'_'.join(combo)}.png"
            cv2.imwrite(str(out_dir / fname), degraded)

            row = {col: 0 for col in DEFECT_COLUMNS}
            for defect_name in combo:
                row[defect_name] = 1          # multi-hot: 2 or 3 columns set
            row["filename"] = fname
            row["split"] = split_name         # inherit the scene's split
            new_rows.append(row)

    # Append and write back, keeping the original column order.
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)[
        ["filename"] + DEFECT_COLUMNS + ["split"]]
    df.to_csv(csv_path, index=False)

    # --- Report -----------------------------------------------------------------
    combo_df = df[df["filename"].str.contains("_combo_", regex=False)]
    print(f"\nWrote {len(new_rows)} new combo images.")
    for split_name, (n_pairs, n_triples) in made.items():
        print(f"  {split_name}: requested {n_pairs} pairs + {n_triples} triples")
    print("\nCombo rows now in labels.csv:")
    for split_name in ["train", "val", "test"]:
        sub = combo_df[combo_df["split"] == split_name]
        n_pair_rows = int((sub[DEFECT_COLUMNS].sum(axis=1) == 2).sum())
        n_triple_rows = int((sub[DEFECT_COLUMNS].sum(axis=1) == 3).sum())
        print(f"  {split_name:<6} {len(sub):>4} combos  ({n_pair_rows} pairs, {n_triple_rows} triples)")

    print("\nPer-class positives per split (all images, singles + combos):")
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        counts = "  ".join(f"{c}={int(sub[c].sum())}" for c in DEFECT_COLUMNS)
        print(f"  {split_name:<6} n={len(sub):<5} {counts}")

    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
