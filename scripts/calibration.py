"""Assess probability calibration on the development subset.

Reports per-label Brier score and a simple expected calibration error (ECE)
using fixed probability bins. This is an engineering diagnostic only; it is
not clinical calibration or validation.

Example:
    python scripts/calibration.py \
        --checkpoint models/checkpoints/resnet18_latest.pt \
        --data-root data/raw/NIH_ChestXray14_subset
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader

from app.inference.evaluate import collect_predictions
from app.inference.model import build_model
from app.preprocessing.dataset import LABELS, ChestXrayDataset, expand_labels, load_metadata
from app.preprocessing.transforms import evaluation_transforms


def expected_calibration_error(y_true, y_prob, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = len(y_true)

    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lower) & (y_prob < upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / total) * abs(accuracy - confidence)

    return float(ece)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.data_root)
    metadata = root / "Data_Entry_2017_v2020.csv"
    test_list = root / "test_list.txt"
    image_dir = root / "images"

    if not all(path.exists() for path in [metadata, test_list, image_dir]):
        raise FileNotFoundError(f"Test data is not configured under {root}.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    df = expand_labels(load_metadata(metadata))
    test_names = {
        line.strip() for line in test_list.read_text().splitlines() if line.strip()
    }
    test_df = df[df["Image Index"].isin(test_names)].copy()

    dataset = ChestXrayDataset(
        test_df,
        image_dir,
        transform=evaluation_transforms(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(LABELS), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_true, y_prob = collect_predictions(model, loader, device)

    rows = []
    for index, label in enumerate(LABELS):
        truth = y_true[:, index]
        probability = y_prob[:, index]
        rows.append(
            {
                "label": label,
                "brier_score": float(brier_score_loss(truth, probability)),
                "ece": expected_calibration_error(
                    truth, probability, bins=args.bins
                ),
                "mean_probability": float(np.mean(probability)),
                "positive_rate": float(np.mean(truth)),
            }
        )

    output = {
        "checkpoint": args.checkpoint,
        "data_root": str(root),
        "test_images": len(test_df),
        "bins": args.bins,
        "metrics": rows,
        "note": (
            "Calibration diagnostics are engineering measurements on the "
            "development subset, not clinical calibration."
        ),
    }

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/calibration.json").write_text(json.dumps(output, indent=2))
    pd.DataFrame(rows).to_csv("outputs/calibration.csv", index=False)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
