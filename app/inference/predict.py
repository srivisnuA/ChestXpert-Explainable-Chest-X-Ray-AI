"""Single-image inference and explainability for ChestXpert."""

from pathlib import Path

import math
import threading

import torch
from PIL import Image

from app.explainability.gradcam import GradCAM
from app.inference.model import build_model
from app.preprocessing.dataset import LABELS
from app.preprocessing.dicom import dicom_to_pil
from app.preprocessing.transforms import evaluation_transforms


class ChestXpertPredictor:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self._lock = threading.RLock()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint must contain model_state_dict")
        self.labels = checkpoint.get("labels", list(LABELS))
        if (not isinstance(self.labels, (list, tuple)) or not self.labels or
                any(not isinstance(label, str) or not label.strip() for label in self.labels) or
                len(set(self.labels)) != len(self.labels)):
            raise ValueError("Checkpoint labels must be unique, non-empty strings")
        self.labels = list(self.labels)
        if not isinstance(checkpoint["model_state_dict"], dict):
            raise ValueError("model_state_dict must be a mapping")
        self.model = build_model(
            num_classes=len(self.labels),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = evaluation_transforms()

    def _predict_tensor(self, tensor: torch.Tensor, threshold: float = 0.5) -> list[dict]:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a finite number between 0 and 1")
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3 or not torch.isfinite(tensor).all():
            raise ValueError("Expected one finite RGB image tensor [1, 3, H, W]")
        with self._lock, torch.no_grad():
            logits = self.model(tensor)
            if logits.shape != (1, len(self.labels)) or not torch.isfinite(logits).all():
                raise ValueError("Model outputs do not match the label contract")
            probabilities = torch.sigmoid(logits)[0].cpu().tolist()

        results = [
            {
                "finding": label,
                "probability": round(float(probability), 6),
                "flagged": bool(probability >= threshold),
            }
            for label, probability in zip(self.labels, probabilities)
        ]
        return sorted(results, key=lambda item: item["probability"], reverse=True)

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        if image.width * image.height > 25_000_000:
            raise ValueError("Image exceeds 25 million pixels")
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict_from_image(
        self, image: Image.Image, threshold: float = 0.5
    ) -> list[dict]:
        """Run inference directly on a PIL image."""
        tensor = self._prepare_image(image)
        return self._predict_tensor(tensor, threshold=threshold)

    def predict(self, image_path: str, threshold: float = 0.5) -> list[dict]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix.lower() == ".dcm":
            image = dicom_to_pil(path)
        else:
            with Image.open(path) as source:
                if source.format not in {"PNG", "JPEG"}:
                    raise ValueError("Only PNG and JPEG raster images are supported")
                if source.width * source.height > 25_000_000:
                    raise ValueError("Image exceeds 25 million pixels")
                image = source.convert("RGB")

        return self.predict_from_image(image, threshold=threshold)

    def explain(self, image: Image.Image, class_index: int) -> torch.Tensor:
        """Generate a normalized Grad-CAM map for one model output."""
        if isinstance(class_index, bool) or not isinstance(class_index, int) or not 0 <= class_index < len(self.labels):
            raise ValueError(f"class_index must be in [0, {len(self.labels) - 1}]")

        tensor = self._prepare_image(image)
        with self._lock:
            cam = GradCAM(self.model, self.model.layer4[-1])
            try:
                return cam.generate(tensor, class_index=class_index)[0, 0].cpu()
            finally:
                cam.close()

    def explain_finding(self, image: Image.Image, finding: str) -> torch.Tensor:
        """Generate Grad-CAM for a finding name."""
        if finding not in self.labels:
            raise ValueError(f"Unknown finding: {finding}")
        return self.explain(image, self.labels.index(finding))
