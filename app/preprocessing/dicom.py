"""Minimal DICOM-to-PIL conversion for research/demo inference."""

from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def dicom_to_pil(path: str | Path) -> Image.Image:
    """Read DICOM pixel data and convert it to an 8-bit RGB PIL image.

    This intentionally ignores patient metadata and returns only image pixels.
    It is a research/demo loader, not a full clinical DICOM presentation-state
    implementation.
    """
    dataset = pydicom.dcmread(str(path))
    pixels = dataset.pixel_array.astype(np.float32)

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept

    low, high = np.percentile(pixels, [1, 99])
    if high <= low:
        low, high = float(pixels.min()), float(pixels.max())

    pixels = np.clip((pixels - low) / (high - low + 1e-8), 0.0, 1.0)

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 1.0 - pixels

    image = Image.fromarray((pixels * 255).astype(np.uint8), mode="L")
    return image.convert("RGB")
