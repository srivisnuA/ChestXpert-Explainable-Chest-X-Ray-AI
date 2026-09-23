"""Run ChestXpert inference from the command line.

Usage:
    python scripts/predict.py --image path/to/xray.png --checkpoint models/checkpoints/resnet18_latest.pt
"""

import argparse
import json

from app.inference.predict import ChestXpertPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="models/checkpoints/resnet18_latest.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predictor = ChestXpertPredictor(args.checkpoint)
    print(json.dumps(
        predictor.predict(args.image, threshold=args.threshold),
        indent=2,
    ))


if __name__ == "__main__":
    main()
