"""Basic DICOM preprocessing tests."""

import numpy as np

from app.preprocessing.dicom import dicom_to_pil


def test_dicom_module_imports():
    assert np is not None
    assert callable(dicom_to_pil)
