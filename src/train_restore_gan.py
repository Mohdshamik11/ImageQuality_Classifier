"""
Blur-specialist fine-tune: adds an adversarial (GAN) loss on top of the already-
trained restoration generator, using ONLY blur degradation (src/degrade.py's
degrade_blur_only). Exposure/contrast are handled by the classical fixes and
noise by Real-ESRGAN in the shipped pipeline (src/enhance.py) -- this run's only
job is to stop the generator's blurry-average output on deblurring specifically.

Two networks, trained together:
    generator     RestoreUNet (src/restore_model.py) -- WARM-STARTED from
                  models/restore_best_lpips.pt, not trained from scratch.
    discriminator UNetDiscriminatorSN (src/discriminator.py) -- fresh, judges
                  "real clean crop, or generator output?" per pixel.

Each training step alternates:
    1. generator step   -- L1 + perceptual + (small weight) adversarial loss.
                            The adversarial term rewards fooling the discriminator,
                            which is what pushes the generator off the smooth
                            average and toward committing to sharp detail.
    2. discriminator step -- real/fake classification loss on real clean crops
                              vs. the generator's (detached) output.

Only the generator is saved -- the discriminator is scaffolding, thrown away
after training, exactly like Real-ESRGAN's own release ships only its generator.

Usage:
    python src/train_restore_gan.py                          # local defaults
    python src/train_restore_gan.py \
        --clean-dirs /path/to/coco /path/to/div2k \
        --out-dir /workspace --epochs 25 --workers 4
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
from pytorch_msssim import ssim as ssim_fn
import lpips as lpips_pkg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import degrade_blur_only
from restore_dataset import build_restore_loaders
from restore_model import RestoreUNet
from discriminator import UNetDiscriminatorSN
from train_restore import PerceptualLoss   # reuse the VGG-feature loss, unchanged

# ============================ CONFIG ============================
BASE_CHANNELS = 48       # generator -- MUST match the warm-start checkpoint
N_BLOCKS = 3
DISC_CHANNELS = 64       # discriminator -- Real-ESRGAN's proven default

WARM_START = "models/restore_best_lpips.pt"   # the L1+SSIM+perceptual generator

BATCH_SIZE = 16
NUM_EPOCHS = 25          # warm-started -> needs far fewer epochs than training from scratch
LR_G = 3e-5              # fine-tune LR -- lower than the 1e-4 used to train the base model
LR_D = 3e-5
BETAS = (0.9, 0.99)
EPS = 1e-8
WARMUP_ITERS = 300       # LinearLR warmup on BOTH optimizers -- the discriminator starts
                        # from scratch, so its early steps are as unstable as the
                        # generator's were on the very first training run
GRAD_CLIP = 1.0

# no SSIM here -- it fights sharpness, which is the whole point of this run.
# adversarial weight starts small: enough to matter, not enough to dominate L1/perceptual
# and cause the instability we fought before.
W_L1, W_PERC, W_ADV = 1.0, 0.05, 0.05

SEED = 42
SAMPLE_EVERY = 5
# ==============================================================


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dirs", nargs="+",
                    default=["data/clean_pool_small/coco", "data/clean_pool_small/div2k"])
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--warm-start", default=WARM_START)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | torch {torch.__version__}", flush=True)

    out = Path(args.out_dir)
    ckpt_dir = out / "models"
    samp_dir = out / "outputs" / "restore_gan_samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "restore_blur_gan.pt"
    lpips_ckpt_path = ckpt_dir / "restore_blur_gan_lpips.pt"
    hist_path = ckpt_dir / "restore_gan_history.csv"

    # blur-only degradation -- this run's whole point
    loaders = build_restore_loaders(
        args.clean_dirs, batch_size=args.batch_size, num_workers=args.workers,
        degrade_fn=degrade_blur_only)
    for split, ldr in loaders.items():
        print(f"{split:5s} {len(ldr.dataset):5d} images | {len(ldr):4d} batches", flush=True)

    # ---- generator: warm-started from the L1+SSIM+perceptual checkpoint ----
    gen = RestoreUNet(base_channels=BASE_CHANNELS, n_blocks=N_BLOCKS).to(device)
    gen.load_state_dict(torch.load(args.warm_start, map_location=device, weights_only=True))
    print(f"generator warm-started from {args.warm_start}", flush=True)

    # ---- discriminator: fresh ----
    disc = UNetDiscriminatorSN(base_channels=DISC_CHANNELS).to(device)

    n_g = sum(p.numel() for p in gen.parameters())
    n_d = sum(p.numel() for p in disc.parameters())
    print(f"generator {n_g:,} params | discriminator {n_d:,} params", flush=True)

    opt_g = torch.optim.Adam(gen.parameters(), lr=LR_G, betas=BETAS, eps=EPS)
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR_D, betas=BETAS, eps=EPS)
    sched_g = torch.optim.lr_scheduler.LinearLR(opt_g, start_factor=0.1, total_iters=WARMUP_ITERS)
    sched_d = torch.optim.lr_scheduler.LinearLR(opt_d, start_factor=0.1, total_iters=WARMUP_ITERS)

    perceptual = PerceptualLoss().to(device)
    bce = nn.BCEWithLogitsLoss()

    def criterion_g(fake, clean, pred_fake_for_g):
        l1 = F.l1_loss(fake, clean)
        perc = perceptual(fake, clean)
        # the generator WANTS the discriminator to say "real" (label 1) about its output
        adv = bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
        total = W_L1 * l1 + W_PERC * perc + W_ADV * adv
        return total, {"l1": l1.item(), "perc": perc.item(), "adv": adv.item()}

    # ---- metrics (same shape as train_restore.py) ----
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
        deg, clean = next(iter(loader))
        deg, clean = deg[:n_rows].to(device), clean[:n_rows].to(device)
        res = m(deg)
        rows = [torch.cat([deg[i], res[i], clean[i]], dim=2).cpu() for i in range(n_rows)]
        grid = torch.cat(rows, dim=1).clamp(0, 1).permute(1, 2, 0).numpy()
        plt.imsave(samp_dir / f"{tag}.png", grid)

    # ---- train ----
    history = []
    best_psnr = -1.0
    best_lpips = float("inf")
    for epoch in range(1, args.epochs + 1):
        gen.train()
        disc.train()
        run = {"total": 0.0, "l1": 0.0, "perc": 0.0, "adv": 0.0, "d": 0.0}
        nb = 0
        t0 = time.time()
        for deg, clean in loaders["train"]:
            deg, clean = deg.to(device), clean.to(device)

            # --- generator step ---
            opt_g.zero_grad()
            fake = gen(deg)
            pred_fake_for_g = disc(fake)                         # gradient must flow into gen
            loss_g, parts = criterion_g(fake, clean, pred_fake_for_g)
            if not torch.isfinite(loss_g) or loss_g.item() > 5.0:
                print(f"  epoch {epoch} step {nb}: bad generator loss {loss_g.item():.3f}, "
                      f"batch skipped", flush=True)
                continue
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP)
            opt_g.step()
            sched_g.step()

            # --- discriminator step ---
            opt_d.zero_grad()
            pred_real = disc(clean)
            pred_fake_for_d = disc(fake.detach())                # detached -- no grad into gen
            loss_d = 0.5 * (bce(pred_real, torch.ones_like(pred_real)) +
                            bce(pred_fake_for_d, torch.zeros_like(pred_fake_for_d)))
            if not torch.isfinite(loss_d):
                print(f"  epoch {epoch} step {nb}: bad discriminator loss, step skipped", flush=True)
            else:
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), GRAD_CLIP)
                opt_d.step()
            sched_d.step()

            run["total"] += loss_g.item()
            for k in ("l1", "perc", "adv"):
                run[k] += parts[k]
            run["d"] += loss_d.item() if torch.isfinite(loss_d) else 0.0
            nb += 1
        for k in run:
            run[k] /= max(nb, 1)

        vp, vs, vl = evaluate(gen, loaders["val"])
        history.append({"epoch": epoch, "train_total": run["total"], "train_l1": run["l1"],
                        "train_perc": run["perc"], "train_adv": run["adv"], "train_d": run["d"],
                        "val_psnr": vp, "val_ssim": vs, "val_lpips": vl})

        flag = ""
        if vp > best_psnr:
            best_psnr = vp
            torch.save(gen.state_dict(), ckpt_path)
            flag += "  <-psnr"
        if vl < best_lpips:
            best_lpips = vl
            torch.save(gen.state_dict(), lpips_ckpt_path)
            flag += "  <-lpips"

        print(f"epoch {epoch:2d} | G {run['total']:.4f} "
              f"(l1 {run['l1']:.4f}  perc {run['perc']:.3f}  adv {run['adv']:.4f}) | "
              f"D {run['d']:.4f} | val PSNR {vp:5.2f}  SSIM {vs:.3f}  LPIPS {vl:.3f} | "
              f"{time.time() - t0:4.0f}s{flag}", flush=True)

        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

        if epoch == 1 or epoch % SAMPLE_EVERY == 0:
            save_grid(gen, loaders["val"], f"epoch_{epoch:02d}")

    save_grid(gen, loaders["val"], "final_lastepoch")
    print(f"\nbest val PSNR  {best_psnr:.2f}   -> {ckpt_path}", flush=True)
    print(f"best val LPIPS {best_lpips:.3f}  -> {lpips_ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
