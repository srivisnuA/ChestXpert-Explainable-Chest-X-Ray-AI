"""Train the ChestXpert baseline.

This script expects NIH ChestX-ray14 to have been downloaded separately.
It uses the official train/test image lists to avoid mixing the held-out
test set into training.

Usage:
    python scripts/train.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from app.inference.model import build_model, save_checkpoint
from app.preprocessing.dataset import LABELS, ChestXrayDataset, expand_labels, load_metadata
from app.preprocessing.transforms import evaluation_transforms, training_transforms


ROOT = Path("data/raw/NIH_ChestXray14")
METADATA = ROOT / "Data_Entry_2017_v2020.csv"
TRAIN_LIST = ROOT / "train_val_list.txt"
TEST_LIST = ROOT / "test_list.txt"
IMAGE_DIR = ROOT / "images"
CHECKPOINT_DIR = Path("models/checkpoints")


def read_image_list(path: Path) -> set[str]:
    return set(path.read_text().splitlines())


def make_splits(df: pd.DataFrame):
    train_names = read_image_list(TRAIN_LIST)
    test_names = read_image_list(TEST_LIST)

    train_df = df[df["Image Index"].isin(train_names)].copy()
    test_df = df[df["Image Index"].isin(test_names)].copy()

    overlap = set(train_df["Image Index"]) & set(test_df["Image Index"])
    if overlap:
        raise RuntimeError(f"Train/test overlap detected: {len(overlap)} images")

    return train_df, test_df


def compute_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    positives = train_df[list(LABELS)].sum().to_numpy(dtype=np.float32)
    negatives = len(train_df) - positives
    positives = np.maximum(positives, 1.0)
    return torch.tensor(negatives / positives, dtype=torch.float32)


def main() -> None:
    required = [METADATA, TRAIN_LIST, TEST_LIST, IMAGE_DIR]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset is not configured. Missing: " + ", ".join(missing)
        )

    torch.manual_seed(42)
    np.random.seed(42)

    df = expand_labels(load_metadata(METADATA))
    train_df, test_df = make_splits(df)

    train_ds = ChestXrayDataset(
        train_df,
        IMAGE_DIR,
        transform=training_transforms(),
    )
    test_ds = ChestXrayDataset(
        test_df,
        IMAGE_DIR,
        transform=evaluation_transforms(),
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(LABELS), pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_df).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        print(f"Epoch {epoch:02d}/10 - train_loss={train_loss:.4f}")

        save_checkpoint(
            model,
            str(CHECKPOINT_DIR / "resnet18_latest.pt"),
            epoch,
            list(LABELS),
        )

    print(f"Training complete. Held-out test images: {len(test_ds):,}")


if __name__ == "__main__":
    main()
