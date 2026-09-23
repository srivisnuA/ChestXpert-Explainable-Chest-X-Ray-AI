"""Single-image inference and explainability for ChestXpert."""

from pathlib import Path

import torch
from PIL import Image

from app.explainability.gradcam import GradCAM
from app.inference.model import build_model
from app.preprocessing.dataset import LABELS
from app.preprocessing.dicom import dicom_to_pil
from app.preprocessing.transforms import evaluation_transforms


class ChestXpertPredictor:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.labels = checkpoint.get("labels", list(LABELS))
        self.model = build_model(
            num_classes=len(self.labels),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = evaluation_transforms()

    def _predict_tensor(self, tensor: torch.Tensor, threshold: float = 0.5) -> list[dict]:
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(tensor))[0].cpu().tolist()

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
            image = Image.open(path).convert("RGB")

        return self.predict_from_image(image, threshold=threshold)

    def explain(self, image: Image.Image, class_index: int) -> torch.Tensor:
        """Generate a normalized Grad-CAM map for one model output."""
        if not 0 <= class_index < len(self.labels):
            raise ValueError(f"class_index must be in [0, {len(self.labels) - 1}]")

        tensor = self._prepare_image(image)
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
