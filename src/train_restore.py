"""
Phase 2b restoration training -- a plain script so it runs the SAME way locally
and on a rented GPU (Kaggle / Colab). No notebook, nothing to clobber.

    # local (uses data/clean_pool_small by default)
    conda run -n imageQuality_Classifier python src/train_restore.py

    # Kaggle: point at the uploaded dataset, write outputs to /kaggle/working
    python src/train_restore.py \
        --clean-dirs /kaggle/input/imagequality-clean-pool/coco \
                     /kaggle/input/imagequality-clean-pool/div2k \
        --out-dir /kaggle/working --epochs 45 --workers 4

Writes (under --out-dir):
    models/restore_best.pt       best-by-val-PSNR checkpoint (state_dict)
    models/restore_history.csv   per-epoch metrics -- rewritten EVERY epoch, so a
                                 killed run still leaves a complete log
    outputs/restore_samples/*.png   [ degraded | restored | clean ] grids
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_msssim import ssim as ssim_fn
import lpips as lpips_pkg
import matplotlib
matplotlib.use("Agg")            # headless: no display on a cloud box
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from restore_dataset import build_restore_loaders
from restore_model import RestoreUNet

# ============================ CONFIG ============================
# CHANGED FROM THE LAST RUN: model widened 32 -> 48 channels, 2 -> 3 blocks/level.
# Reason: L1 + SSIM(0.1) was fully STABLE for 28 epochs but plateaued at ~22 dB val
# PSNR and the sample grids showed almost no denoising/deblurring. Three runs with
# different losses / learning rates all stall at the same place -> the 1.4M-param
# 2-level net is the limit, not the loss recipe. This bumps it to ~5M params.
BASE_CHANNELS = 48
N_BLOCKS      = 3

BATCH_SIZE    = 16
NUM_EPOCHS    = 45
LR            = 1e-4
BETAS         = (0.9, 0.99)
EPS           = 1e-8            # default; the 1e-4 anti-divergence brake is not needed
WARMUP_ITERS  = 300            # LinearLR warmup, start at 0.1x LR
GRAD_CLIP     = 1.0

W_L1, W_SSIM, W_PERC = 1.0, 0.1, 0.0   # perceptual still off -- add after capacity is proven

SEED          = 42
SAMPLE_EVERY  = 5              # save a [deg|restored|clean] grid every N epochs
# ==============================================================


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dirs", nargs="+",
                    default=["data/clean_pool_small/coco", "data/clean_pool_small/div2k"],
                    help="one or more folders of clean images")
    ap.add_argument("--out-dir", default=".",
                    help="root for models/ and outputs/ (use /kaggle/working on Kaggle)")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=2,
                    help="DataLoader workers -- 0 on Windows if it errors, 4 on Kaggle")
    ap.add_argument("--resume", default="",
                    help="path to a checkpoint to load before training (optimizer state is NOT restored)")
    return ap.parse_args()


class PerceptualLoss(nn.Module):
    """Frozen VGG16 feature-space L1. Slices at relu1_2 / relu2_2 / relu3_3
    (low + mid texture, not semantics). Inputs get ImageNet-normalised inside."""

    def __init__(self, layer_idx=(3, 8, 15)):
        super().__init__()
        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        prev = 0
        for idx in layer_idx:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, idx + 1)]))
            prev = idx + 1
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        x = (pred - self.mean) / self.std
        y = (target - self.mean) / self.std
        loss = 0.0
        for s in self.slices:
            x, y = s(x), s(y)
            loss = loss + F.l1_loss(x, y)
        return loss


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | torch {torch.__version__} | "
          f"{torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}", flush=True)

    out = Path(args.out_dir)
    ckpt_dir = out / "models"
    samp_dir = out / "outputs" / "restore_samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "restore_best.pt"
    hist_path = ckpt_dir / "restore_history.csv"

    loaders = build_restore_loaders(
        args.clean_dirs, batch_size=args.batch_size, num_workers=args.workers)
    for split, ldr in loaders.items():
        print(f"{split:5s} {len(ldr.dataset):5d} images | {len(ldr):4d} batches", flush=True)

    # ---- loss ----
    perceptual = PerceptualLoss().to(device) if W_PERC else None

    def criterion(pred, target):
        l1 = F.l1_loss(pred, target)
        ssim_term = ((1.0 - ssim_fn(pred, target, data_range=1.0))
                     if W_SSIM else torch.zeros((), device=pred.device))
        perc = perceptual(pred, target) if W_PERC else torch.zeros((), device=pred.device)
        total = W_L1 * l1 + W_SSIM * ssim_term + W_PERC * perc
        return total, {"l1": l1.item(), "ssim": float(ssim_term), "perc": float(perc)}

    # ---- model / optimiser ----
    model = RestoreUNet(base_channels=BASE_CHANNELS, n_blocks=N_BLOCKS).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"resumed weights from {args.resume}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} parameters (base_channels={BASE_CHANNELS}, n_blocks={N_BLOCKS})", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, eps=EPS)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_ITERS)

    # ---- metrics ----
    lpips_fn = lpips_pkg.LPIPS(net="alex").to(device)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    def psnr(pred, target):
        mse = ((pred - target) ** 2).mean(dim=[1, 2, 3]).clamp(min=1e-10)
        return (10 * torch.log10(1.0 / mse)).mean()

    @torch.no_grad()
    def evaluate(m, loader):
        m.eval()
        tp = ts = tl = 0.0
        n = 0
        for deg, clean in loader:
            deg, clean = deg.to(device), clean.to(device)
            o = m(deg)
            bs = deg.size(0)
            tp += psnr(o, clean).item() * bs
            ts += ssim_fn(o, clean, data_range=1.0).item() * bs
            tl += lpips_fn(o * 2 - 1, clean * 2 - 1).mean().item() * bs
            n += bs
        return tp / n, ts / n, tl / n

    @torch.no_grad()
    def save_grid(m, loader, tag, n_rows=4):
        m.eval()
        deg, clean = next(iter(loader))        # val loader is seeded -> same images every call
        deg, clean = deg[:n_rows].to(device), clean[:n_rows].to(device)
        res = m(deg)
        rows = [torch.cat([deg[i], res[i], clean[i]], dim=2).cpu() for i in range(n_rows)]
        grid = torch.cat(rows, dim=1).clamp(0, 1).permute(1, 2, 0).numpy()
        plt.imsave(samp_dir / f"{tag}.png", grid)

    # ---- train ----
    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run = {"total": 0.0, "l1": 0.0, "ssim": 0.0, "perc": 0.0}
        nb = 0
        t0 = time.time()
        for deg, clean in loaders["train"]:
            deg, clean = deg.to(device), clean.to(device)
            optimizer.zero_grad()
            out = model(deg)
            loss, parts = criterion(out, clean)
            if not torch.isfinite(loss) or loss.item() > 5.0:
                print(f"  epoch {epoch} step {nb}: bad loss {loss.item():.3f}, skipped", flush=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            run["total"] += loss.item()
            for k in ("l1", "ssim", "perc"):
                run[k] += parts[k]
            nb += 1
        for k in run:
            run[k] /= max(nb, 1)

        vp, vs, vl = evaluate(model, loaders["val"])
        history.append({"epoch": epoch,
                        "train_total": run["total"], "train_l1": run["l1"],
                        "train_ssim": run["ssim"], "train_perc": run["perc"],
                        "val_psnr": vp, "val_ssim": vs, "val_lpips": vl})

        flag = ""
        if vp > best:
            best = vp
            torch.save(model.state_dict(), ckpt_path)
            flag = "  <- saved"
        print(f"epoch {epoch:2d} | loss {run['total']:.4f} "
              f"(l1 {run['l1']:.4f}  ssim {run['ssim']:.4f}  perc {run['perc']:.3f}) | "
              f"val PSNR {vp:5.2f}  SSIM {vs:.3f}  LPIPS {vl:.3f} | {time.time() - t0:4.0f}s{flag}",
              flush=True)

        # rewrite the whole CSV every epoch -> a killed run still leaves a full log
        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

        if epoch == 1 or epoch % SAMPLE_EVERY == 0:
            save_grid(model, loaders["val"], f"epoch_{epoch:02d}")

    save_grid(model, loaders["val"], "final_lastepoch")
    print(f"\nbest val PSNR {best:.2f} -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
