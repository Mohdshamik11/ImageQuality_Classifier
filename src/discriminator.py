"""
U-Net discriminator with spectral normalization -- Real-ESRGAN's design, used
here to fine-tune the phase-2b generator (RestoreUNet) with an adversarial loss
(src/train_restore_gan.py).

What a discriminator is and why THIS shape:
  During GAN training, the discriminator is a second network whose only job is
  to look at an image and guess "real clean photo, or generator output?" Its
  gradient becomes an extra loss term for the generator: to fool an improving
  discriminator, the generator has to stop producing the safe, blurry average
  and commit to output that genuinely looks like a real sharp photo.

  A plain classifier discriminator (one real/fake score for the whole image)
  only gives coarse feedback. This is a U-NET discriminator: it downsamples
  then upsamples back to a real/fake score AT EVERY PIXEL. That's much richer
  feedback -- it can tell the generator exactly *where* a patch still looks
  fake (e.g. one soft corner), not just "somewhere in this image."

Spectral normalization (`torch.nn.utils.spectral_norm`) rescales each conv
layer's weights so the layer can't amplify its input by more than a bounded
factor. Discriminators that get too powerful too fast are the classic cause of
GAN instability (the generator's gradient either vanishes or explodes); this
is the standard fix, used by Real-ESRGAN and most modern GANs.

Usage (smoke test):
    python src/discriminator.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

IN_CHANNELS = 3
BASE_CHANNELS = 64   # matches Real-ESRGAN's proven default -- a known-stable recipe


class UNetDiscriminatorSN(nn.Module):
    def __init__(self, in_channels: int = IN_CHANNELS, base_channels: int = BASE_CHANNELS,
                skip_connection: bool = True):
        super().__init__()
        C = base_channels
        self.skip_connection = skip_connection

        # --- encoder: three stride-2 downsamples, doubling channels each time ---
        # (no spectral norm on the very first conv -- standard practice, it's not
        # the layer that causes runaway discriminator gradients)
        self.conv0 = nn.Conv2d(in_channels, C, 3, 1, 1)
        self.conv1 = spectral_norm(nn.Conv2d(C, C * 2, 4, 2, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(C * 2, C * 4, 4, 2, 1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(C * 4, C * 8, 4, 2, 1, bias=False))

        # --- decoder: bilinear upsample + conv back to full resolution ---
        self.conv4 = spectral_norm(nn.Conv2d(C * 8, C * 4, 3, 1, 1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(C * 4, C * 2, 3, 1, 1, bias=False))
        self.conv6 = spectral_norm(nn.Conv2d(C * 2, C, 3, 1, 1, bias=False))

        # --- two more full-res convs, then the per-pixel real/fake score ---
        self.conv7 = spectral_norm(nn.Conv2d(C, C, 3, 1, 1, bias=False))
        self.conv8 = spectral_norm(nn.Conv2d(C, C, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(C, 1, 3, 1, 1)   # 1 channel: a realness map, not a class score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) float in [0, 1]. H, W should be divisible by 8 (three
        # halvings); the restoration generator's 256x256 tiles satisfy this.
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)     # (C,    H,   W  )
        x1 = F.leaky_relu(self.conv1(x0), 0.2, inplace=True)    # (2C,   H/2, W/2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2, inplace=True)    # (4C,   H/4, W/4)
        x3 = F.leaky_relu(self.conv3(x2), 0.2, inplace=True)    # (8C,   H/8, W/8)

        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), 0.2, inplace=True)    # (4C, H/4, W/4)
        if self.skip_connection:
            x4 = x4 + x2                                        # U-Net skip (additive)

        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), 0.2, inplace=True)    # (2C, H/2, W/2)
        if self.skip_connection:
            x5 = x5 + x1

        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), 0.2, inplace=True)    # (C, H, W)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), 0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), 0.2, inplace=True)
        return self.conv9(out)                                  # (B, 1, H, W) real/fake logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    disc = UNetDiscriminatorSN().to(device)

    n_params = sum(p.numel() for p in disc.parameters())
    print(f"device: {device}")
    print(f"total parameters: {n_params:,}")

    dummy = torch.rand(2, 3, 256, 256, device=device)
    out = disc(dummy)
    print(f"input {tuple(dummy.shape)}  ->  output {tuple(out.shape)}  (per-pixel real/fake logit)")
