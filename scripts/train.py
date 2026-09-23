"""Train the ChestXpert baseline.

The NIH ChestX-ray14 dataset must be downloaded separately. Training uses
the published train_val_list.txt pool and keeps test_list.txt held out.

Examples:
    python scripts/train.py
    python scripts/train.py --epochs 3 --batch-size 16
    python scripts/train.py --epochs 10 --batch-size 32 --num-workers 4
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

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
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def make_splits(df: pd.DataFrame):
    train_names = read_image_list(TRAIN_LIST)
    test_names = read_image_list(TEST_LIST)

    train_df = df[df["Image Index"].isin(train_names)].copy()
    test_df = df[df["Image Index"].isin(test_names)].copy()

    overlap = set(train_df["Image Index"]) & set(test_df["Image Index"])
    if overlap:
        raise RuntimeError(f"Train/test overlap detected: {len(overlap)} images")

    missing_train = train_names - set(train_df["Image Index"])
    missing_test = test_names - set(test_df["Image Index"])
    if missing_train or missing_test:
        raise RuntimeError(
            "Split manifest contains images missing from metadata: "
            f"train={len(missing_train)}, test={len(missing_test)}"
        )

    return train_df, test_df


def compute_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    positives = train_df[list(LABELS)].sum().to_numpy(dtype=np.float32)
    negatives = len(train_df) - positives
    positives = np.maximum(positives, 1.0)
    return torch.tensor(negatives / positives, dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ChestXpert ResNet-18.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "resnet18_latest.pt"))
    parser.add_argument("--data-root", default=str(ROOT), help="Dataset root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global ROOT, METADATA, TRAIN_LIST, TEST_LIST, IMAGE_DIR
    ROOT = Path(args.data_root)
    METADATA = ROOT / "Data_Entry_2017_v2020.csv"
    TRAIN_LIST = ROOT / "train_val_list.txt"
    TEST_LIST = ROOT / "test_list.txt"
    IMAGE_DIR = ROOT / "images"

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    required = [METADATA, TRAIN_LIST, TEST_LIST, IMAGE_DIR]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset is not configured. Missing: " + ", ".join(missing)
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    print(f"Train images: {len(train_ds):,}")
    print(f"Held-out test images: {len(test_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(num_classes=len(LABELS), pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=compute_pos_weight(train_df).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        print(
            f"Epoch {epoch:02d}/{args.epochs} - "
            f"train_loss={train_loss:.4f}"
        )

        save_checkpoint(
            model,
            str(checkpoint_path),
            epoch,
            list(LABELS),
        )

    print(f"Training complete. Checkpoint: {checkpoint_path}")
    print("Next: run scripts/evaluate.py on the held-out test split.")


if __name__ == "__main__":
    main()
