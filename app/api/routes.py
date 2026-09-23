"""REST endpoints for ChestXpert inference."""

import os
import tempfile

from flask import Blueprint, jsonify, request

from app.inference.predict import ChestXpertPredictor


api = Blueprint("api", __name__)
_predictor = None


def get_predictor():
    global _predictor

    checkpoint = os.getenv(
        "CHESTXPERT_CHECKPOINT",
        "models/checkpoints/resnet18_latest.pt",
    )

    if _predictor is None:
        if not os.path.exists(checkpoint):
            return None
        _predictor = ChestXpertPredictor(checkpoint)

    return _predictor


@api.get("/health")
def health():
    predictor = get_predictor()
    return jsonify(
        {
            "status": "ok",
            "service": "chestxpert",
            "model_loaded": predictor is not None,
        }
    )


@api.post("/predict")
def predict():
    predictor = get_predictor()

    if predictor is None:
        return jsonify(
            {
                "error": "Model checkpoint is not available.",
                "hint": "Train the model first or set CHESTXPERT_CHECKPOINT.",
            }
        ), 503

    if "image" not in request.files:
        return jsonify({"error": "Upload an image using the 'image' field."}), 400

    uploaded = request.files["image"]

    if not uploaded.filename:
        return jsonify({"error": "No filename supplied."}), 400

    suffix = os.path.splitext(uploaded.filename)[1].lower() or ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        uploaded.save(temp.name)
        results = predictor.predict(temp.name)

    return jsonify(
        {
            "disclaimer": (
                "Research/educational output only. "
                "Requires qualified clinician review and is not a diagnosis."
            ),
            "findings": results,
        }
    )
