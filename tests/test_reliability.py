"""Synthetic correctness tests; these do not measure medical performance."""
import io
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from app import create_app
from app.inference.evaluate import collect_predictions, per_label_metrics
from app.inference.metrics import multilabel_metrics
from app.inference.model import build_model, save_checkpoint
from app.inference.predict import ChestXpertPredictor
from app.preprocessing.splits import split_metadata, manifest_hash, verify_validation_manifest


@pytest.fixture(scope="module")
def predictor(tmp_path_factory):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    path = tmp_path_factory.mktemp("weights") / "synthetic.pt"
    model = build_model(num_classes=2, pretrained=False)
    save_checkpoint(model, str(path), 0, ["A", "B"])
    return ChestXpertPredictor(str(path), device="cpu")


def png_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(stream, format="PNG")
    stream.seek(0)
    return stream


def test_actual_checkpoint_inference_and_resnet_gradcam(predictor):
    image = Image.new("RGB", (64, 64), "gray")
    first = predictor.predict_from_image(image)
    assert first == predictor.predict_from_image(image)
    assert {row["finding"] for row in first} == {"A", "B"}
    assert all(0 <= row["probability"] <= 1 for row in first)
    before = len(predictor.model.layer4[-1]._forward_hooks)
    for _ in range(2):
        cam = predictor.explain(image, 0)
        assert cam.shape == (224, 224)
        assert torch.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1
    assert len(predictor.model.layer4[-1]._forward_hooks) == before


@pytest.mark.parametrize("labels", [[], ["A", "A"], [""], [None], [2], "A"])
def test_invalid_checkpoint_labels(labels, tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {}, "labels": labels}, path)
    with pytest.raises(ValueError):
        ChestXpertPredictor(str(path), "cpu")


def test_wrong_model_output_and_invalid_tensor_fail(predictor):
    with pytest.raises(ValueError):
        predictor._predict_tensor(torch.zeros(2, 3, 16, 16))
    with patch.object(predictor.model, "forward", return_value=torch.tensor([[float("nan"), 0.0]])):
        with pytest.raises(ValueError, match="output"):
            predictor._predict_tensor(torch.zeros(1, 3, 16, 16))
    with pytest.raises(ValueError):
        predictor.explain(Image.new("RGB", (64, 64)), True)


def test_api_upload_works_on_windows_and_cleans_up(predictor):
    application = create_app({"TESTING": True})
    application.extensions["chestxpert"]["predictor"] = predictor
    client = application.test_client()
    assert client.get("/api/ready").status_code == 200
    response = client.post("/api/predict", data={"image": (png_bytes(), "../../scan.png")})
    assert response.status_code == 200
    assert len(response.json["findings"]) == 2
    assert response.headers["Cache-Control"] == "no-store"
    paths = []
    def failing(path):
        paths.append(Path(path))
        raise ValueError("private metadata must not appear in responses")
    with patch.object(predictor, "predict", side_effect=failing):
        response = client.post("/api/predict", data={"image": (png_bytes(), "scan.png")})
    assert response.status_code == 400
    assert "private metadata" not in response.text
    assert paths and not paths[0].parent.exists()


def test_api_validation_size_and_unavailable_model(tmp_path):
    application = create_app({"TESTING": True, "CHESTXPERT_CHECKPOINT": str(tmp_path / "missing.pt")})
    client = application.test_client()
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/ready").status_code == 503
    assert client.post("/api/predict").status_code == 400
    assert client.post("/api/predict", data={"image": (io.BytesIO(b"bad"), "scan.exe")}).status_code == 415
    assert client.post("/api/predict", data={"image": (png_bytes(), "scan.png")}).status_code == 503
    application.config["MAX_CONTENT_LENGTH"] = 10
    response = client.post("/api/predict", data={"image": (png_bytes(), "scan.png")})
    assert response.status_code == 413 and response.is_json


def test_bad_image_and_model_failure_are_controlled(predictor):
    application = create_app({"TESTING": True})
    application.extensions["chestxpert"]["predictor"] = predictor
    client = application.test_client()
    response = client.post("/api/predict", data={"image": (io.BytesIO(b"not an image"), "scan.png")})
    assert response.status_code == 400
    with patch.object(predictor, "predict", side_effect=RuntimeError("private path")):
        response = client.post("/api/predict", data={"image": (png_bytes(), "scan.png")})
    assert response.status_code == 500 and "private path" not in response.text


def test_app_instances_do_not_share_models():
    first, second = create_app(), create_app()
    first.extensions["chestxpert"]["predictor"] = Mock()
    assert second.extensions["chestxpert"]["predictor"] is None


def test_undefined_metrics_are_json_null_with_support():
    truth = np.array([[0, 1, 1], [0, 1, 0]])
    probabilities = np.array([[.1, .9, .8], [.2, .8, .2]])
    rows = per_label_metrics(truth, probabilities, ["absent", "always", "mixed"])
    assert rows[0]["auroc"] is None and rows[0]["average_precision"] is None
    assert rows[1]["auroc"] is None and rows[1]["positive_count"] == 2
    result = multilabel_metrics(truth, probabilities)
    assert result["macro_auroc"] == 1.0 and result["auroc_label_count"] == 1
    json.dumps({"rows": rows, "macro": result}, allow_nan=False)


@pytest.mark.parametrize("truth,probability", [([], []), ([[1]], [[float("nan")]]),
    ([[2]], [[.5]]), ([[1]], [[1.1]]), ([[1]], [[.1, .2]])])
def test_invalid_evaluation_arrays_are_rejected(truth, probability):
    with pytest.raises(ValueError):
        multilabel_metrics(truth, probability)


def test_empty_loader_is_rejected():
    with pytest.raises(ValueError, match="no samples"):
        collect_predictions(Mock(), [], "cpu")


def test_split_validation_prevents_patient_and_image_leakage():
    df = pd.DataFrame({"Image Index": ["a", "b", "c"], "Patient ID": [1, 2, 3]})
    train, validation, test = split_metadata(df, {"a", "b"}, {"c"}, {"b"})
    assert list(train["Image Index"]) == ["a"]
    assert list(validation["Image Index"]) == ["b"]
    assert list(test["Image Index"]) == ["c"]
    with pytest.raises(ValueError, match="overlap"):
        split_metadata(df, {"a", "b"}, {"b", "c"})
    with pytest.raises(ValueError, match="training pool"):
        split_metadata(df, {"a", "b"}, {"c"}, {"c"})
    df["Patient ID"] = [1, 1, 3]
    with pytest.raises(ValueError, match="Patient overlap"):
        split_metadata(df, {"a", "b"}, {"c"}, {"b"})


def test_threshold_tuning_requires_matching_training_provenance():
    checkpoint = {"split_hashes": {"validation": manifest_hash(["b", "a"])}}
    verify_validation_manifest(checkpoint, {"a", "b"})
    for invalid in [{}, {"split_hashes": {"validation": "wrong"}}]:
        with pytest.raises(ValueError, match="reserved during training"):
            verify_validation_manifest(invalid, {"a", "b"})
