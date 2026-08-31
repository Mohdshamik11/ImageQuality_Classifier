"""
PyTorch data pipeline for the photo-quality classifier.

Turns data/synthetic/ + labels.csv into batches of (image_tensor, label_tensor)
ready to feed the CNN.

  ImageQualityDataset  - fetches ONE (image, labels) pair by index, loading the
                         image file from disk on demand.
  build_dataloaders()  - reads labels.csv and builds one Dataset + DataLoader per
                         split (train / val / test).

Concepts new here vs. the data-generation scripts:
  tensor      - like a numpy array, but can live on the GPU and records the
                operations applied to it so gradients can flow back in training.
  Dataset     - a class exposing __len__ and __getitem__(i); PyTorch's standard
                "here is how to get example i" interface.
  DataLoader  - wraps a Dataset and yields BATCHES (groups of N examples),
                optionally shuffled, optionally loaded on worker processes.
  transforms  - a per-image preprocessing pipeline. transforms.ToTensor() also
                rescales pixels 0-255 -> 0.0-1.0 and moves the colour channel to
                the front: (C, H, W), which is what conv layers expect.

Usage (smoke test):
    python src/dataset.py
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Must stay in sync with the columns written by generate_synthetic.py / split_dataset.py.
DEFECT_COLUMNS = ["blur", "underexposed", "overexposed", "noise", "contrast"]

IMAGE_SIZE = 256  # every image is resized to IMAGE_SIZE x IMAGE_SIZE before the model


class ImageQualityDataset(Dataset):
    """One split (train / val / test) of the synthetic dataset."""

    def __init__(self, df: pd.DataFrame, image_dir: str, image_size: int = IMAGE_SIZE,
                 augment: bool = False, train: bool = False):
        # df holds ONLY this split's rows. reset_index so this dataset's index i
        # maps cleanly to self.df.iloc[i].
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)

        if not augment:
            # Baseline pipeline: squish the whole image into a fixed square, then a
            # 0-1 float tensor. Simple, but the downscale averages away fine grain,
            # which is why the noise class scores poorly.
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),  # squish to a fixed square
                transforms.ToTensor(),                        # PIL uint8 HxWxC -> float (C,H,W) in [0,1]
            ])
        elif train:
            # Train, augment on: gently resize the SHORT side to image_size (keeps
            # aspect ratio, minimal averaging), then take a RANDOM image_size square
            # crop. The crop keeps native pixel detail so grain survives, and the
            # randomness is free data augmentation against overfitting.
            self.transform = transforms.Compose([
                transforms.Resize(image_size),        # short side -> image_size
                transforms.RandomCrop(image_size),    # random square patch at native detail
                transforms.ToTensor(),
            ])
        else:
            # Val / test, augment on: same gentle resize, but a FIXED centre crop so
            # evaluation is deterministic and repeatable.
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # PIL opens images in RGB order (OpenCV uses BGR). We stay in RGB from
        # here on, including the eventual Streamlit app, so there's no mismatch.
        img_path = self.image_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # tensor, shape (3, 256, 256)

        # The 5 defect columns as a float tensor of shape (5,). Float (not int)
        # because BCEWithLogitsLoss expects float targets.
        label = torch.from_numpy(row[DEFECT_COLUMNS].to_numpy(dtype="float32"))

        return image, label


def build_dataloaders(labels_csv: str = "data/synthetic/labels.csv",
                      image_dir: str = "data/synthetic",
                      batch_size: int = 32,
                      num_workers: int = 0,
                      augment: bool = False):
    """Read labels.csv and return {"train": loader, "val": loader, "test": loader}.

    augment=False (default): every split is resized to a fixed square (the baseline).
    augment=True: train uses Resize(short side) + RandomCrop (grain-preserving +
    augmentation); val/test use Resize(short side) + CenterCrop (deterministic).

    Train is shuffled every epoch; val and test keep CSV row order, so predictions
    from those loaders line up with loaders[split].dataset.df row-for-row (used
    later to slice out the combo images for their own metrics).

    num_workers=0 by default: on Windows, DataLoader workers are spawned as new
    processes and can be slow/awkward from a notebook. Raise it later only if
    disk loading becomes the training bottleneck.
    """
    df = pd.read_csv(labels_csv)

    loaders = {}
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        dataset = ImageQualityDataset(split_df, image_dir,
                                      augment=augment, train=(split == "train"))

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # faster host->GPU copy when on CUDA
            drop_last=False,
        )
    return loaders


if __name__ == "__main__":
    # Quick check: describe one batch from each split, in both modes.
    for aug in (False, True):
        print(f"\n--- augment={aug} ---")
        loaders = build_dataloaders(augment=aug)
        for split, loader in loaders.items():
            images, labels = next(iter(loader))
            print(f"{split:5s}: {len(loader.dataset):5d} images | "
                  f"batch images {tuple(images.shape)} {images.dtype} "
                  f"range [{images.min():.2f}, {images.max():.2f}] | "
                  f"batch labels {tuple(labels.shape)} {labels.dtype}")
