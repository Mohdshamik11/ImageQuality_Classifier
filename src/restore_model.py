"""
The phase-2b restoration network: a small residual U-Net.

Predicts a CORRECTION to add to the degraded input (global residual learning),
not the whole clean image from scratch:

    restored = clamp(degraded + RestoreUNet(degraded), 0, 1)

Architecture -- a 2-level encoder/decoder, channels double each level down:

    head (3->C)
      enc1 [res blocks] (C,  256) --down--> enc2 [res blocks] (2C, 128)
                                    --down--> bottleneck [res blocks] (4C, 64)
      dec1 [res blocks] (2C, 128) <--up-- (skip: enc2)
      dec2 [res blocks] (C,  256) <--up-- (skip: enc1)
    tail (C->3)  -- this is the predicted correction

(A 3rd level -- doubling channels again to 8C at a 32x32 bottleneck -- was tried
first and measured: 13.25M params, 305ms/tile on CPU, too slow for the free-tier
deploy target. A plain conv's cost scales with channels^2, so that deepest level
alone was over half the total. Two levels cuts it to ~3.2M params while keeping
enough receptive field for our blur range, max 16px motion / 5px defocus.)

No BatchNorm anywhere: it normalises using batch statistics, which fights the
pixel-exact fidelity restoration needs (and behaves differently at inference,
one image at a time, than during training). Modern restoration nets drop it.

Usage (smoke test):
    python src/restore_model.py
"""
import torch
import torch.nn as nn

IN_CHANNELS = 3
BASE_CHANNELS = 32   # C. Measured: 32ch/2blocks ~1.4M params, ~131ms/tile CPU (vs 48ch: 295ms).
N_BLOCKS = 2          # residual blocks per resolution level (rounds of refinement)


class ResidualBlock(nn.Module):
    """Conv -> ReLU -> Conv, then add the block's own input back (LOCAL residual).
    Gives backprop a direct shortcut path so we can stack many of these without
    the gradient vanishing through them.

    res_scale: the block's contribution is multiplied by a small number before
    being added back. Early in training the conv weights are random, so this
    keeps every block's effect small at first (closer to "do nothing") instead
    of many random corrections compounding into an unstable start."""

    def __init__(self, channels: int, res_scale: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.act(self.conv1(x)))
        return x + self.res_scale * out


def _block_stack(channels: int, n: int = N_BLOCKS) -> nn.Sequential:
    """n ResidualBlocks in a row, all at the same channel count/resolution."""
    return nn.Sequential(*[ResidualBlock(channels) for _ in range(n)])


class Down(nn.Module):
    """Stride-2 convolution: extracts features AND halves H, W in one learned
    step (unlike the classifier's Conv-then-separate-MaxPool)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class Up(nn.Module):
    """PixelShuffle upsampling: a normal conv produces 4x the target channel
    count at the SAME resolution, then PixelShuffle rearranges every pixel's
    extra channels into a 2x2 block of new pixels -- no overlapping math, so no
    checkerboard artefacts (the problem with transposed convolution)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class RestoreUNet(nn.Module):
    def __init__(self, base_channels: int = BASE_CHANNELS, n_blocks: int = N_BLOCKS):
        super().__init__()
        C = base_channels

        self.head = nn.Conv2d(IN_CHANNELS, C, kernel_size=3, padding=1)

        # --- encoder: each level processes at its resolution, then halves ---
        # level 0
        self.enc1 = _block_stack(C, n_blocks)
        self.down1 = Down(C, 2 * C)
        # level 1
        self.enc2 = _block_stack(2 * C, n_blocks)
        self.down2 = Down(2 * C, 4 * C)
        # level 2 -- the bottleneck itself (no separate deeper level; see docstring)
        self.bottleneck = _block_stack(4 * C, n_blocks)

        # --- decoder: upsample, fuse with the matching encoder skip, process ---
        self.up1 = Up(4 * C, 2 * C)
        self.fuse1 = nn.Conv2d(4 * C, 2 * C, kernel_size=1)  # 2C (up) + 2C (skip) -> 2C
        self.dec1 = _block_stack(2 * C, n_blocks)

        self.up2 = Up(2 * C, C)
        self.fuse2 = nn.Conv2d(2 * C, C, kernel_size=1)      # C (up) + C (skip) -> C
        self.dec2 = _block_stack(C, n_blocks)

        self.tail = nn.Conv2d(C, IN_CHANNELS, kernel_size=3, padding=1)  # -> the correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) float in [0, 1]. H and W must be divisible by 4 (two
        # halvings) -- our 256x256 tiles satisfy this (256 -> 128 -> 64).
        h = self.head(x)

        e1 = self.enc1(h)                    # (B, C,  H,   W)
        e2 = self.enc2(self.down1(e1))       # (B, 2C, H/2, W/2)
        b = self.bottleneck(self.down2(e2))  # (B, 4C, H/4, W/4)

        d1 = self.up1(b)                                        # -> (B, 2C, H/2, W/2)
        d1 = self.dec1(self.fuse1(torch.cat([d1, e2], dim=1)))  # concat on the CHANNEL axis

        d2 = self.up2(d1)                                       # -> (B, C, H, W)
        d2 = self.dec2(self.fuse2(torch.cat([d2, e1], dim=1)))

        correction = self.tail(d2)                               # (B, 3, H, W)

        # Global residual: predict the CHANGE, add it to the original, clamp
        # back into the valid pixel range.
        return torch.clamp(x + correction, 0.0, 1.0)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RestoreUNet().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"total parameters: {n_params:,}")

    dummy = torch.rand(2, 3, 256, 256, device=device)  # fake batch, valid [0,1] pixels
    out = model(dummy)
    print(f"input {tuple(dummy.shape)}  ->  output {tuple(out.shape)}")
    print(f"output range [{out.min().item():.3f}, {out.max().item():.3f}]  (should be within [0,1])")
