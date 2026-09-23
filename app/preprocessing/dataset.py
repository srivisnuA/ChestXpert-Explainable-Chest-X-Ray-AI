"""Dataset utilities for NIH ChestX-ray14.

The dataset is multi-label: one image may contain multiple findings.
This module keeps the official image-level labels separate from model
training so preprocessing remains reproducible and testable.
"""

from pathlib import Path
from typing import Sequence

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


LABELS: Sequence[str] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
)


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Image Index", "Finding Labels"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")
    return df


def expand_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    label_text = result["Finding Labels"].fillna("")
    for label in LABELS:
        result[label] = label_text.map(
            lambda value, target=label: float(target in str(value).split("|"))
        )
    return result


class ChestXrayDataset(Dataset):
    """Minimal image dataset returning an image and a 14-label target vector."""

    def __init__(self, dataframe: pd.DataFrame, image_dir: str | Path, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = self.image_dir / row["Image Index"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        target = row[list(LABELS)].to_numpy(dtype="float32")
        return image, target
