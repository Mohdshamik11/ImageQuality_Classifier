"""
Second-stage blur-GAN fine-tune: same generator/discriminator setup as
src/train_restore_gan.py, but on REAL paired blur/sharp photos (RealBlur-J,
src/realblur_dataset.py) instead of synthetic blur-only degradation of COCO.

WARM-STARTS from models/restore_blur_gan_lpips.pt -- the already COCO-trained
blur-GAN checkpoint (committed at git commit 5ae5a37) -- not from scratch and
not from the pre-GAN base model. This is meant to be the ONLY thing that
changes relative to that run: same generator/discriminator architecture, same
loss weights, same optimizer settings, so that any quality difference in the
result is attributable to the real-vs-synthetic data, not a confound from also
having changed the training recipe.

Writes to DIFFERENT checkpoint files than the COCO run (restore_blur_gan_real*,
not restore_blur_gan*) so the existing, known-good, committed checkpoint is
never at risk of being silently overwritten by an unproven run -- compare the
two explicitly before switching src/restore_blur_gan.py's CKPT over.

Usage:
    python src/train_realblur_gan.py --data-root data/realblur
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
from realblur_dataset import build_realblur_loaders
from restore_model import RestoreUNet
from discriminator import UNetDiscriminatorSN
from train_restore import PerceptualLoss   # reuse the VGG-feature loss, unchanged

# ============================ CONFIG ============================
# unchanged from train_restore_gan.py -- see that file's docstring for why
BASE_CHANNELS = 48
N_BLOCKS = 3
DISC_CHANNELS = 64

WARM_START = "models/restore_blur_gan_lpips.pt"   # the COCO-trained blur-GAN, not the pre-GAN base

BATCH_SIZE = 4            # local 6GB-VRAM GPU, not a rented 24GB one -- the one deliberately
                          # different setting, forced by hardware rather than chosen for the experiment
NUM_EPOCHS = 15           # smaller dataset (3.7k real pairs vs 4k synthetic) and second-stage
                          # fine-tune -- likely needs fewer passes, revisit from the loss curve
LR_G = 3e-5
LR_D = 3e-5
BETAS = (0.9, 0.99)
EPS = 1e-8
WARMUP_ITERS = 300
GRAD_CLIP = 1.0

W_L1, W_PERC, W_ADV = 1.0, 0.05, 0.05

SEED = 42
SAMPLE_EVERY = 5
# ==============================================================


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="extracted RealBlur root, containing "
                         "RealBlur-J_ECC_IMCORR_centroid_itensity_ref/")
    ap.add_argument("--max-train-pairs", type=int, default=None,
                    help="cap on the 3,757 official train pairs, for a faster first run")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--warm-start", default=WARM_START)
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames (e.g. 'ext' -> restore_blur_gan_real_ext.pt), "
                         "so a continuation run doesn't overwrite the checkpoint it warm-started from")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | torch {torch.__version__}", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    out = Path(args.out_dir)
    ckpt_dir = out / "models"
    samp_dir = out / "outputs" / f"restore_gan_real{suffix}_samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"restore_blur_gan_real{suffix}.pt"
    lpips_ckpt_path = ckpt_dir / f"restore_blur_gan_real{suffix}_lpips.pt"
    hist_path = ckpt_dir / f"restore_gan_real{suffix}_history.csv"

    loaders = build_realblur_loaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.workers,
        max_train_pairs=args.max_train_pairs)
    for split, ldr in loaders.items():
        print(f"{split:5s} {len(ldr.dataset):5d} pairs | {len(ldr):4d} batches", flush=True)

    # ---- generator: warm-started from the COCO-trained blur-GAN ----
    gen = RestoreUNet(base_channels=BASE_CHANNELS, n_blocks=N_BLOCKS).to(device)
    gen.load_state_dict(torch.load(args.warm_start, map_location=device, weights_only=True))
    print(f"generator warm-started from {args.warm_start}", flush=True)

    # ---- discriminator: fresh, same as the COCO run (thrown away after training there too) ----
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
        adv = bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
        total = W_L1 * l1 + W_PERC * perc + W_ADV * adv
        return total, {"l1": l1.item(), "perc": perc.item(), "adv": adv.item()}

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
        for blur, gt in loader:
            blur, gt = blur.to(device), gt.to(device)
            o = m(blur)
            bs = blur.size(0)
            tp += psnr(o, gt).item() * bs
            ts += ssim_fn(o, gt, data_range=1.0).item() * bs
            tl += lpips_fn(o * 2 - 1, gt * 2 - 1).mean().item() * bs
            n += bs
        return tp / n, ts / n, tl / n

    @torch.no_grad()
    def save_grid(m, loader, tag, n_rows=4):
        m.eval()
        blur, gt = next(iter(loader))
        n_rows = min(n_rows, blur.size(0))
        blur, gt = blur[:n_rows].to(device), gt[:n_rows].to(device)
        res = m(blur)
        rows = [torch.cat([blur[i], res[i], gt[i]], dim=2).cpu() for i in range(n_rows)]
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
        for blur, gt in loaders["train"]:
            blur, gt = blur.to(device), gt.to(device)

            # --- generator step ---
            opt_g.zero_grad()
            fake = gen(blur)
            pred_fake_for_g = disc(fake)
            loss_g, parts = criterion_g(fake, gt, pred_fake_for_g)
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
            pred_real = disc(gt)
            pred_fake_for_d = disc(fake.detach())
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
