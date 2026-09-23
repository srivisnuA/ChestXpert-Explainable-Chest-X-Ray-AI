"""Evaluate a trained ChestXpert checkpoint on the official test split.

Usage:
    python scripts/evaluate.py --checkpoint models/checkpoints/resnet18_latest.pt
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from app.inference.evaluate import collect_predictions, per_label_metrics
from app.inference.model import build_model
from app.preprocessing.dataset import LABELS, ChestXrayDataset, expand_labels, load_metadata
from app.preprocessing.transforms import evaluation_transforms


ROOT = Path("data/raw/NIH_ChestXray14")
METADATA = ROOT / "Data_Entry_2017_v2020.csv"
TEST_LIST = ROOT / "test_list.txt"
IMAGE_DIR = ROOT / "images"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not all(path.exists() for path in [METADATA, TEST_LIST, IMAGE_DIR]):
        raise FileNotFoundError("NIH ChestX-ray14 test data is not configured.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    df = expand_labels(load_metadata(METADATA))
    test_names = set(TEST_LIST.read_text().splitlines())
    test_df = df[df["Image Index"].isin(test_names)].copy()

    dataset = ChestXrayDataset(
        test_df,
        IMAGE_DIR,
        transform=evaluation_transforms(),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(LABELS), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    y_true, y_prob = collect_predictions(model, loader, device)
    metrics = per_label_metrics(
        y_true,
        y_prob,
        LABELS,
        threshold=args.threshold,
    )

    output = {
        "checkpoint": args.checkpoint,
        "test_images": len(test_df),
        "threshold": args.threshold,
        "metrics": metrics,
    }

    Path("outputs").mkdir(exist_ok=True)
    output_path = Path("outputs/test_metrics.json")
    output_path.write_text(json.dumps(output, indent=2))

    pd.DataFrame(metrics).to_csv("outputs/test_metrics.csv", index=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
