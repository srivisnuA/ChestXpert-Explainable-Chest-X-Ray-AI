"""Evaluate a trained ChestXpert checkpoint.

The default path targets the full NIH dataset. A development subset can be
evaluated by supplying --data-root.

Usage:
    python scripts/evaluate.py --checkpoint models/checkpoints/resnet18_latest.pt
    python scripts/evaluate.py --checkpoint models/checkpoints/resnet18_latest.pt --data-root data/raw/NIH_ChestXray14_subset
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
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.data_root)
    metadata = root / "Data_Entry_2017_v2020.csv"
    test_list = root / "test_list.txt"
    image_dir = root / "images"

    if not all(path.exists() for path in [metadata, test_list, image_dir]):
        raise FileNotFoundError(
            f"Test data is not configured under {root}. "
            "Provide --data-root pointing to the dataset directory."
        )

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
    metrics = per_label_metrics(
        y_true,
        y_prob,
        LABELS,
        threshold=args.threshold,
    )

    output = {
        "checkpoint": args.checkpoint,
        "data_root": str(root),
        "test_images": len(test_df),
        "threshold": args.threshold,
        "metrics": metrics,
        "note": (
            "Development-subset evaluation is for engineering validation only; "
            "it is not clinical validation or a benchmark result."
        ),
    }

    Path("outputs").mkdir(exist_ok=True)
    output_path = Path("outputs/test_metrics.json")
    output_path.write_text(json.dumps(output, indent=2))
    pd.DataFrame(metrics).to_csv("outputs/test_metrics.csv", index=False)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
