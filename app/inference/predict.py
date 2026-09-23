"""Single-image inference for ChestXpert."""

from pathlib import Path

import torch
from PIL import Image

from app.inference.model import build_model
from app.preprocessing.dataset import LABELS
from app.preprocessing.transforms import evaluation_transforms


class ChestXpertPredictor:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        labels = checkpoint.get("labels", list(LABELS))
        self.labels = labels

        self.model = build_model(
            num_classes=len(self.labels),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = evaluation_transforms()

    def predict(self, image_path: str, threshold: float = 0.5) -> list[dict]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(path)

        image = Image.open(path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(tensor))[0].cpu().tolist()

        results = []
        for label, probability in zip(self.labels, probabilities):
            results.append(
                {
                    "finding": label,
                    "probability": round(float(probability), 6),
                    "flagged": bool(probability >= threshold),
                }
            )

        return sorted(results, key=lambda item: item["probability"], reverse=True)
