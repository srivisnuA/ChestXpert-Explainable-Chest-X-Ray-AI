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
    if (int(getattr(dataset, "NumberOfFrames", 1)) != 1 or
            int(getattr(dataset, "SamplesPerPixel", 1)) != 1 or
            getattr(dataset, "PhotometricInterpretation", "") not in {"MONOCHROME1", "MONOCHROME2"}):
        raise ValueError("Only single-frame monochrome DICOM images are supported")
    if int(dataset.Rows) * int(dataset.Columns) > 25_000_000:
        raise ValueError("DICOM exceeds 25 million pixels")
    pixels = dataset.pixel_array.astype(np.float32)
    if pixels.ndim != 2 or not pixels.size:
        raise ValueError("Expected a non-empty 2D pixel array")

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept
    if not np.isfinite(pixels).all():
        raise ValueError("DICOM pixel values must be finite")

    low, high = np.percentile(pixels, [1, 99])
    if high <= low:
        low, high = float(pixels.min()), float(pixels.max())

    pixels = np.clip((pixels - low) / (high - low + 1e-8), 0.0, 1.0)

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 1.0 - pixels

    image = Image.fromarray((pixels * 255).astype(np.uint8), mode="L")
    return image.convert("RGB")
