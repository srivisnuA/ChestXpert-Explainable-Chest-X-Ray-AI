"""Utilities for turning a Grad-CAM map into a viewable overlay."""

import numpy as np
from PIL import Image


def cam_to_overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a normalized CAM with the original RGB image.

    The function intentionally keeps visualization separate from model
    inference so the explanation can be inspected independently.
    """
    image = image.convert("RGB")
    cam = np.asarray(cam).squeeze()
    cam = np.clip(cam, 0.0, 1.0)

    heat = np.zeros((*cam.shape, 3), dtype=np.uint8)
    heat[..., 0] = (255 * cam).astype(np.uint8)
    heat[..., 1] = (180 * (1.0 - cam)).astype(np.uint8)

    heat_image = Image.fromarray(heat).resize(image.size)
    return Image.blend(image, heat_image, alpha=alpha)
