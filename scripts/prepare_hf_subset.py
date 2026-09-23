"""Build a small development subset from the Hugging Face NIH ChestX-ray14 mirror.

This is for pipeline development only. It does NOT reproduce the official
NIH train/test benchmark and must not be presented as clinical validation.

Usage:
    python scripts/prepare_hf_subset.py --samples 2000
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image

from app.preprocessing.dataset import LABELS


HF_DATASET = "chehablab/NIHChestXR"
OUTPUT_ROOT = Path("data/raw/NIH_ChestXray14_subset")


HF_TO_PROJECT = {
    1: "Atelectasis",
    2: "Cardiomegaly",
    3: "Effusion",
    4: "Infiltration",
    5: "Mass",
    6: "Nodule",
    7: "Pneumonia",
    8: "Pneumothorax",
    9: "Consolidation",
    10: "Edema",
    11: "Emphysema",
    12: "Fibrosis",
    13: "Pleural Thickening",
    14: "Hernia",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples < 100:
        raise ValueError("--samples should be at least 100")

    output_images = OUTPUT_ROOT / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {args.samples:,} samples from {HF_DATASET}...")
    dataset = load_dataset(HF_DATASET, split="train", streaming=True)

    rows = []
    for index, example in enumerate(dataset):
        if index >= args.samples:
            break

        image = example["image"].convert("RGB")
        filename = f"hf_{index:06d}.png"
        image.save(output_images / filename)

        labels = [
            HF_TO_PROJECT[label_index]
            for label_index in example["labels"]
            if label_index in HF_TO_PROJECT
        ]

        row = {
            "Image Index": filename,
            "Finding Labels": "|".join(labels) if labels else "No Finding",
            "Patient ID": str(example.get("patient_id", index)),
        }
        for label in LABELS:
            row[label] = int(label in labels)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_ROOT / "Data_Entry_2017_v2020.csv", index=False)

    # For a development experiment only, split by patient ID.
    rng = __import__("numpy").random.default_rng(args.seed)
    patients = frame["Patient ID"].unique()
    rng.shuffle(patients)
    cutoff = max(1, int(len(patients) * 0.8))
    train_patients = set(patients[:cutoff])
    test_patients = set(patients[cutoff:])

    train_names = frame.loc[frame["Patient ID"].isin(train_patients), "Image Index"]
    test_names = frame.loc[frame["Patient ID"].isin(test_patients), "Image Index"]

    (OUTPUT_ROOT / "train_val_list.txt").write_text(
        "\n".join(train_names) + "\n"
    )
    (OUTPUT_ROOT / "test_list.txt").write_text(
        "\n".join(test_names) + "\n"
    )

    print(f"Saved {len(frame):,} images to {output_images}")
    print(f"Development train images: {len(train_names):,}")
    print(f"Development test images: {len(test_names):,}")
    print("This subset is for engineering validation, not clinical benchmarking.")


if __name__ == "__main__":
    main()
