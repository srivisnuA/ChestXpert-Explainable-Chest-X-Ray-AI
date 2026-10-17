"""REST endpoints for ChestXpert inference."""

import os
import tempfile
from pathlib import Path

from flask import Blueprint, jsonify, request, current_app
from PIL import UnidentifiedImageError, Image
from pydicom.errors import InvalidDicomError

from app.inference.predict import ChestXpertPredictor


api = Blueprint("api", __name__)


def get_predictor():
    state = current_app.extensions["chestxpert"]
    with state["lock"]:
        if state["predictor"] is None:
            checkpoint = current_app.config["CHESTXPERT_CHECKPOINT"]
            if not os.path.isfile(checkpoint):
                return None
            try:
                state["predictor"] = ChestXpertPredictor(checkpoint)
            except Exception:
                current_app.logger.error("Model initialization failed; verify the configured checkpoint.")
                return None
        return state["predictor"]


@api.get("/health")
def health():
    predictor = current_app.extensions["chestxpert"]["predictor"]
    return jsonify(
        {
            "status": "ok",
            "service": "chestxpert",
            "model_loaded": predictor is not None,
        }
    )


@api.get("/ready")
def ready():
    loaded = get_predictor() is not None
    return jsonify(ready=loaded), 200 if loaded else 503


@api.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image using the 'image' field."}), 400

    uploaded = request.files["image"]

    if not uploaded.filename:
        return jsonify({"error": "No filename supplied."}), 400

    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".dcm"}:
        return jsonify(error="Supported formats: PNG, JPEG, DICOM (.dcm)."), 415
    predictor = get_predictor()
    if predictor is None:
        return jsonify(error="Model is unavailable. Configure a valid trained checkpoint."), 503

    # A closed file in a temporary directory is reopenable on Windows too.
    # Never use a supplied filename as a filesystem path; cleanup runs on errors.
    try:
        with tempfile.TemporaryDirectory(prefix="chestxpert-") as directory:
            path = Path(directory) / ("upload" + suffix)
            uploaded.save(path)
            results = predictor.predict(str(path))
    except (UnidentifiedImageError, InvalidDicomError, Image.DecompressionBombError, ValueError, OSError):
        return jsonify(error="Invalid or unsupported image content."), 400
    except Exception:
        current_app.logger.error("Inference failed; inspect the model configuration.")
        return jsonify(error="Inference failed."), 500

    return jsonify(
        {
            "disclaimer": (
                "Research/educational output only. "
                "Requires qualified clinician review and is not a diagnosis."
            ),
            "findings": results,
        }
    )
