"""
Real-photo evaluation for the phase-2b restoration model.

There is no clean ground truth for real photos, so this does three things per image:
  1. Runs the restoration U-Net (tiled, feathered blend) and saves a
     [ before | after ] pair.
  2. Runs the FROZEN defect classifier on before and after -- the flagged
     problems should get weaker.
  3. Reports no-reference quality metrics (BRISQUE, NIQE lower=better;
     MUSIQ higher=better) plus a Laplacian-variance sharpness proxy,
     before vs after.

Scale note: the model was trained on 256 crops from images resized to <=800 px.
Real photos are much larger, so by default we downscale the long side to
--long-side before restoring, keeping the model near its training scale.

Usage:
    conda run -n imageQuality_Classifier python src/eval_restore.py
    python src/eval_restore.py --photos data/real_test --ckpt models/restore_best_lpips.pt --long-side 768
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from restore_model import RestoreUNet          # noqa: E402
from restore_infer import restore_tensor       # noqa: E402  (shared tiled inference)
from predict import predict, DEFECT_COLUMNS    # noqa: E402  (frozen classifier)

# must match src/train_restore.py CONFIG that produced the checkpoint
BASE_CHANNELS = 48
N_BLOCKS = 3

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", default="data/real_test", help="folder of input photos")
    ap.add_argument("--ckpt", default="", help="restoration checkpoint (default: auto)")
    ap.add_argument("--out", default="outputs/real_eval", help="where to write pairs")
    ap.add_argument("--long-side", type=int, default=768,
                    help="downscale each photo's long side to this before restoring "
                         "(closest to training scale). 0 = full resolution.")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    return ap.parse_args()


def pick_ckpt(arg: str) -> Path:
    if arg:
        return Path(arg)
    for name in ("restore_best_lpips.pt", "restore_best.pt", "restore_best_ep44.pt"):
        p = ROOT / "models" / name
        if p.exists():
            return p
    raise SystemExit("no restoration checkpoint found in models/ -- pass --ckpt")


def load_model(ckpt: Path, device: str) -> RestoreUNet:
    m = RestoreUNet(base_channels=BASE_CHANNELS, n_blocks=N_BLOCKS).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m


def load_photo(path: Path, long_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if long_side and max(img.size) > long_side:
        s = long_side / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
    # restoration U-Net needs H, W divisible by 4
    w, h = img.size
    img = img.crop((0, 0, w - w % 4, h - h % 4))
    return img


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def to_pil(t: torch.Tensor) -> Image.Image:
    a = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a)


def lap_var(img: Image.Image) -> float:
    """Variance of the Laplacian -- a cheap sharpness proxy (higher = sharper)."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(g, (3, 3))
    lap = (win * k).sum(axis=(-1, -2))
    return float(lap.var())


def build_noref():
    """Return {name: (metric_fn, lower_is_better)} or {} if pyiqa is unavailable."""
    try:
        import pyiqa
    except Exception:
        print("  (pyiqa not installed -- skipping BRISQUE/NIQE/MUSIQ. "
              "`pip install pyiqa` to enable.)")
        return {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for name in ("brisque", "niqe", "musiq"):
        try:
            m = pyiqa.create_metric(name, device=dev)
            out[name] = (m, bool(m.lower_better))
        except Exception as e:
            print(f"  (could not load {name}: {e})")
    return out


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = pick_ckpt(args.ckpt)
    print(f"device: {device}\ncheckpoint: {ckpt.name}\n")

    photos = sorted(p for p in Path(args.photos).iterdir()
                    if p.suffix.lower() in IMG_EXTS)
    if not photos:
        raise SystemExit(f"no images in {args.photos} -- drop your test photos there")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(ckpt, device)
    noref = build_noref()

    @torch.no_grad()
    def score_noref(img: Image.Image) -> dict:
        if not noref:
            return {}
        t = to_tensor(img).unsqueeze(0).to(device)
        return {n: float(fn(t).item()) for n, (fn, _) in noref.items()}

    rows = []
    for path in photos:
        before = load_photo(path, args.long_side)
        bt = to_tensor(before)
        after = to_pil(restore_tensor(model, bt, device, tile=args.tile, stride=args.stride))

        # side-by-side
        pair = Image.new("RGB", (before.width * 2, before.height))
        pair.paste(before, (0, 0))
        pair.paste(after, (before.width, 0))
        pair.save(out_dir / f"{path.stem}_pair.png")

        # classifier before/after
        pb, pa = predict(before)["probs"], predict(after)["probs"]
        # metrics
        nb, na = score_noref(before), score_noref(after)
        sb, sa = lap_var(before), lap_var(after)

        print(f"=== {path.name}  ({before.width}x{before.height}) ===")
        print("  classifier defect probability (before -> after, lower is better):")
        for c in DEFECT_COLUMNS:
            arrow = "  worse" if pa[c] > pb[c] + 0.02 else ("  better" if pa[c] < pb[c] - 0.02 else "")
            print(f"    {c:<13} {pb[c]:.3f} -> {pa[c]:.3f}{arrow}")
        print(f"  sharpness (Laplacian var, higher=sharper): {sb:8.1f} -> {sa:8.1f}")
        for n, (_, lo) in noref.items():
            better = (na[n] < nb[n]) if lo else (na[n] > nb[n])
            print(f"  {n:<8} {nb[n]:7.2f} -> {na[n]:7.2f}   "
                  f"({'lower' if lo else 'higher'} better) {'OK' if better else 'WORSE'}")
        print()

        row = {"file": path.name, "sharp_b": sb, "sharp_a": sa}
        for c in DEFECT_COLUMNS:
            row[f"{c}_b"] = pb[c]
            row[f"{c}_a"] = pa[c]
        for n in noref:
            row[f"{n}_b"] = nb[n]
            row[f"{n}_a"] = na[n]
        rows.append(row)

    # ---- summary ----
    print("========== SUMMARY (mean over", len(rows), "photos) ==========")
    print(f"  sharpness      {np.mean([r['sharp_b'] for r in rows]):8.1f} -> "
          f"{np.mean([r['sharp_a'] for r in rows]):8.1f}")
    for c in DEFECT_COLUMNS:
        mb = np.mean([r[f'{c}_b'] for r in rows])
        ma = np.mean([r[f'{c}_a'] for r in rows])
        print(f"  {c:<13} {mb:.3f} -> {ma:.3f}   (delta {ma - mb:+.3f})")
    for n, (_, lo) in noref.items():
        mb = np.mean([r[f'{n}_b'] for r in rows])
        ma = np.mean([r[f'{n}_a'] for r in rows])
        print(f"  {n:<13} {mb:7.2f} -> {ma:7.2f}   ({'lower' if lo else 'higher'} better)")

    print(f"\npairs written to {out_dir}")


if __name__ == "__main__":
    main()
