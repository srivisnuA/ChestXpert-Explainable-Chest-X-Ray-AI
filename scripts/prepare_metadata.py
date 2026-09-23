"""Validate and prepare ChestX-ray14 metadata.

Usage:
    python scripts/prepare_metadata.py
"""

from pathlib import Path

import pandas as pd

from app.preprocessing.dataset import LABELS, expand_labels, load_metadata


ROOT = Path("data/raw/NIH_ChestXray14")
INPUT = ROOT / "Data_Entry_2017_v2020.csv"
OUTPUT = Path("data/processed/labels.csv")


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Download the research dataset separately; "
            "do not commit it to Git."
        )

    df = expand_labels(load_metadata(INPUT))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Keep only fields needed for reproducible image-level classification.
    columns = ["Image Index", "Finding Labels", *LABELS]
    df[columns].to_csv(OUTPUT, index=False)

    print(f"Rows: {len(df):,}")
    print(f"Labels: {len(LABELS)}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
