"""Find development-set thresholds that maximize per-label F1.

This is threshold analysis for engineering validation on the configured
development subset. It is not clinical calibration or clinical validation.

Example:
    python scripts/threshold_analysis.py \
        --checkpoint models/checkpoints/resnet18_latest.pt \
        --data-root data/raw/NIH_ChestXray14_subset
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from app.inference.evaluate import collect_predictions
from app.inference.model import build_model
from app.preprocessing.dataset import LABELS, ChestXrayDataset, expand_labels, load_metadata
from app.preprocessing.transforms import evaluation_transforms
from app.preprocessing.splits import split_metadata, verify_validation_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--validation-list", type=Path, required=True,
                        help="Explicit held-out training-pool manifest also excluded during training.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-threshold", type=float, default=0.05)
    parser.add_argument("--max-threshold", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.min_threshold <= args.max_threshold <= 1 or not np.isfinite(args.step) or args.step <= 0:
        raise ValueError("Threshold range must be within [0, 1] with a finite positive step")

    root = Path(args.data_root)
    metadata = root / "Data_Entry_2017_v2020.csv"
    test_list = root / "test_list.txt"
    image_dir = root / "images"

    if not all(path.exists() for path in [metadata, test_list, image_dir]):
        raise FileNotFoundError(
            f"Test data is not configured under {root}."
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    df = expand_labels(load_metadata(metadata))
    read_names = lambda path: {line.strip() for line in path.read_text().splitlines() if line.strip()}
    validation_names = read_names(args.validation_list)
    if not validation_names:
        raise ValueError("Validation manifest must be non-empty")
    verify_validation_manifest(checkpoint, validation_names)
    _, test_df, _ = split_metadata(df, read_names(root / "train_val_list.txt"),
                                  read_names(test_list), validation_names)

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

    thresholds = np.arange(
        args.min_threshold,
        args.max_threshold + args.step / 2,
        args.step,
    )

    rows = []
    for index, label in enumerate(LABELS):
        truth = y_true[:, index]
        probability = y_prob[:, index]

        best = None
        for threshold in thresholds:
            prediction = (probability >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                truth,
                prediction,
                average="binary",
                zero_division=0,
            )
            candidate = {
                "label": label,
                "best_threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
            if best is None or candidate["f1"] > best["f1"]:
                best = candidate

        rows.append(best)

    output = {
        "checkpoint": args.checkpoint,
        "data_root": str(root),
        "validation_images": len(test_df),
        "validation_manifest": str(args.validation_list),
        "threshold_search": {
            "min": args.min_threshold,
            "max": args.max_threshold,
            "step": args.step,
        },
        "metrics": rows,
        "note": (
            "Thresholds were selected on the development subset for engineering "
            "analysis only. They are not clinically calibrated thresholds."
        ),
    }

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/threshold_analysis.json").write_text(json.dumps(output, indent=2))
    pd.DataFrame(rows).to_csv("outputs/threshold_analysis.csv", index=False)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
