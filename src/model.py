"""
The baseline CNN for the photo-quality classifier.

This file is the model's BLUEPRINT only — which layers exist and how data flows
through them. It holds no learned weights: creating a BaselineCNN() gives you a
model with random weights, which the training loop then adjusts.

Architecture (deliberately simple — a first model to beat, not a tuned one):

    input (B, 3, 256, 256)
      stage 1:  Conv 3->32,   ReLU,  MaxPool 2   -> (B, 32, 128, 128)
      stage 2:  Conv 32->64,  ReLU,  MaxPool 2   -> (B, 64,  64,  64)
      stage 3:  Conv 64->128, ReLU,  MaxPool 2   -> (B, 128, 32,  32)
      stage 4:  Conv 128->256,ReLU,  MaxPool 2   -> (B, 256, 16,  16)
      flatten                                    -> (B, 65536)
      Linear 65536->128, ReLU                    -> (B, 128)
      Linear 128->5                              -> (B, 5)   raw scores (logits)

Note: NO sigmoid at the end. The model outputs raw scores; BCEWithLogitsLoss
applies the sigmoid internally (more numerically stable). At evaluation time we
call torch.sigmoid() ourselves to get 0-1 probabilities.

Concepts new here (all from torch.nn):
  nn.Module     - base class for models/layers; define layers in __init__,
                  define the data path in forward(). Backward pass is automatic.
  nn.Conv2d     - one convolution layer. padding=1 with a 3x3 kernel keeps
                  height/width unchanged, so only the pooling shrinks the image.
  nn.ReLU       - clip negatives to 0.
  nn.MaxPool2d  - 2x2 "keep the largest" downsample; halves height and width.
  nn.Linear     - dense layer; every input connects to every output.
  nn.Flatten    - (B, C, H, W) -> (B, C*H*W) so nn.Linear can take it.
  nn.Sequential - runs a list of layers in order, keeping forward() short.

Usage (smoke test):
    python src/model.py
"""
import torch
import torch.nn as nn

NUM_CLASSES = 5     # blur, underexposed, overexposed, noise, contrast
IMAGE_SIZE = 256    # must match dataset.py
NUM_POOLS = 4       # one MaxPool per conv stage


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, image_size: int = IMAGE_SIZE):
        super().__init__()  # required: lets nn.Module set itself up

        # --- Feature extractor: 4 stages, each halves H and W ------------------
        self.features = nn.Sequential(
            # stage 1: 3 -> 32 channels,  256x256 -> 128x128
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # stage 2: 32 -> 64,  128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # stage 3: 64 -> 128,  64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # stage 4: 128 -> 256,  32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # After NUM_POOLS halvings: a 256px image is 256 / 2**4 = 16 px per side,
        # with 256 channels -> 256 * 16 * 16 numbers once flattened.
        final_spatial = image_size // (2 ** NUM_POOLS)
        flat_features = 256 * final_spatial * final_spatial

        # --- Classifier head -------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # (B, 256, 16, 16) -> (B, 65536)
            nn.Linear(flat_features, 128),    # dense layer (this holds most of the model's weights)
            nn.ReLU(),
            nn.Linear(128, num_classes),      # -> 5 raw scores; NO sigmoid here
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 256, 256), float, pixels in [0, 1]
        x = self.features(x)      # -> (B, 256, 16, 16)
        x = self.classifier(x)   # -> (B, 5) logits
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaselineCNN().to(device)

    # numel() = number of elements in a weight tensor; sum over all of them.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"total parameters: {n_params:,}")

    # Fake batch of 2 images, just to confirm shapes flow through end to end.
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    out = model(dummy)
    print(f"input {tuple(dummy.shape)}  ->  output {tuple(out.shape)}")
    print("output row 0 (raw logits):", out[0].detach().cpu().numpy().round(3))
