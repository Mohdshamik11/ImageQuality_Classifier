"""
Adds a `split` column (train / val / test) to data/synthetic/labels.csv.

The split is done at the SCENE level, not the image level. Every synthetic image
whose filename starts with `raw_0123_` came from the same original COCO photo
`raw_0123.jpg`, so all of them must land in the same split -- otherwise the model
could train on `raw_0123_blur.png` and then be "validated" on `raw_0123_noise.png`,
a scene it has effectively already seen. That inflated validation score is called
data leakage (specifically group/scene leakage).

Method:
  1. Read labels.csv.
  2. Recover each row's scene id from its filename (raw_0123_blur.png -> raw_0123).
  3. Shuffle the unique scene ids with a seeded RNG (reproducible every run).
  4. First 70% of scenes -> train, next 15% -> val, last 15% -> test.
  5. Map every row to its scene's split and write labels.csv back out.

Usage:
    python src/split_dataset.py
    python src/split_dataset.py --seed 42 --train-frac 0.70 --val-frac 0.15
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Same defect columns / order as generate_synthetic.py. Used here only to print a
# per-class balance check per split.
DEFECT_COLUMNS = ["blur", "underexposed", "overexposed", "noise", "contrast"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-csv", type=str, default="data/synthetic/labels.csv",
                        help="CSV to read and rewrite with a `split` column")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the scene shuffle; same seed = same split")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    # test-frac is implied: whatever scenes are left after train + val.
    args = parser.parse_args()

    csv_path = Path(args.labels_csv)
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found -- run generate_synthetic.py first.")

    # pandas reads the CSV into a DataFrame: an in-memory table with named columns
    # you can filter, group, and assign to. df["filename"] is one column (a Series).
    df = pd.read_csv(csv_path)

    # Recover the scene id from each filename. str.extract runs the regex on every
    # row at once; the capture group (raw_ followed by digits) becomes the value.
    # e.g. "raw_0123_blur.png" -> "raw_0123", "raw_0123_clean.png" -> "raw_0123".
    df["scene_id"] = df["filename"].str.extract(r"^(raw_\d+)_")

    if df["scene_id"].isna().any():
        bad = df.loc[df["scene_id"].isna(), "filename"].tolist()
        raise SystemExit(f"Could not parse a scene id from: {bad[:5]} ...")

    # The unique scenes, sorted first so the starting order is deterministic
    # BEFORE we shuffle (otherwise it depends on row order in the CSV).
    scene_ids = np.sort(df["scene_id"].unique())
    n_scenes = len(scene_ids)

    # One seeded generator -> permutation() returns the ids in a shuffled order
    # that is identical on every run with the same seed.
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(scene_ids)

    # Slice the shuffled scenes into the three splits. int() truncates, and test
    # takes the remainder, so the three counts always sum back to n_scenes.
    n_train = int(n_scenes * args.train_frac)
    n_val = int(n_scenes * args.val_frac)
    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train:n_train + n_val])
    test_ids = set(shuffled[n_train + n_val:])

    # Guard: the three id sets must be disjoint. If they weren't, a scene would be
    # in two splits -- the leakage this whole script exists to prevent.
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Map each scene id to its split label, then tag every row via its scene_id.
    split_of = {}
    for sid in train_ids:
        split_of[sid] = "train"
    for sid in val_ids:
        split_of[sid] = "val"
    for sid in test_ids:
        split_of[sid] = "test"
    df["split"] = df["scene_id"].map(split_of)

    # scene_id was a working column only; drop it so the CSV schema stays
    # filename + defect columns + split.
    df = df.drop(columns="scene_id")

    df.to_csv(csv_path, index=False)  # index=False: don't write pandas' row numbers

    # --- Report -----------------------------------------------------------------
    print(f"Split {n_scenes} scenes -> "
          f"{len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test "
          f"(seed {args.seed})\n")

    print(f"{'split':<8}{'images':>8}   " + "".join(f"{c:>13}" for c in DEFECT_COLUMNS)
          + f"{'clean':>8}")
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        # Per-class positive counts + count of all-zero (clean) rows, to confirm
        # every class is represented proportionally in every split.
        class_counts = [int(sub[c].sum()) for c in DEFECT_COLUMNS]
        clean_count = int((sub[DEFECT_COLUMNS].sum(axis=1) == 0).sum())
        print(f"{split_name:<8}{len(sub):>8}   "
              + "".join(f"{n:>13}" for n in class_counts)
              + f"{clean_count:>8}")

    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
