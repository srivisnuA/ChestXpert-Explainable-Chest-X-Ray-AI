"""Generate a Grad-CAM explanation for one ChestXpert image.

Example:
    python scripts/explain.py \
        --checkpoint models/checkpoints/resnet18_latest.pt \
        --image path/to/xray.png \
        --finding Edema
"""

import argparse
from pathlib import Path

from PIL import Image

from app.explainability.overlay import cam_to_overlay
from app.inference.predict import ChestXpertPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--finding", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = Image.open(image_path).convert("RGB")
    predictor = ChestXpertPredictor(args.checkpoint)

    results = predictor.predict_from_image(image)
    result = next((row for row in results if row["finding"] == args.finding), None)
    if result is None:
        raise ValueError(
            f"Unknown finding: {args.finding}. "
            f"Choose from: {', '.join(predictor.labels)}"
        )

    cam = predictor.explain_finding(image, args.finding)
    overlay = cam_to_overlay(
        image,
        cam.numpy(),
        alpha=0.45,
    )

    output = Path(args.output) if args.output else Path(
        Path("outputs") / f"gradcam_{args.finding.lower().replace(' ', '_')}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output)

    print(f"Finding: {args.finding}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Flagged at threshold 0.5: {result['flagged']}")
    print(f"Grad-CAM overlay: {output}")


if __name__ == "__main__":
    main()
