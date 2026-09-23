"""Tests for inference image preprocessing without requiring a checkpoint."""

from PIL import Image

from app.preprocessing.transforms import evaluation_transforms


def test_evaluation_transform_returns_batched_compatible_tensor():
    image = Image.new("RGB", (320, 240), color="black")
    tensor = evaluation_transforms()(image)

    assert tuple(tensor.shape) == (3, 224, 224)
    assert tensor.dtype.is_floating_point
