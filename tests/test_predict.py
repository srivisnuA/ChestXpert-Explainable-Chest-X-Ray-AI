import pytest

from app.inference.predict import ChestXpertPredictor


def test_predictor_rejects_missing_checkpoint():
    with pytest.raises(FileNotFoundError):
        ChestXpertPredictor("does-not-exist.pt")
