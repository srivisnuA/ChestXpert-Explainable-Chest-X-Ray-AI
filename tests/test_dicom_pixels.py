"""Synthetic DICOM fixtures: no patient data or external downloads."""
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from app.preprocessing.dicom import dicom_to_pil


def write_dicom(path, *, photo="MONOCHROME2", frames=1, constant=False):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows = ds.Columns = 16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photo
    ds.NumberOfFrames = frames
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    pixels = np.zeros((16, 16), dtype=np.uint16) if constant else np.arange(256, dtype=np.uint16).reshape(16, 16)
    ds.PixelData = pixels.tobytes() * frames
    ds.save_as(path, enforce_file_format=True)


def test_monochrome_inversion_and_constant_image(tmp_path):
    p1, p2, p3 = [tmp_path / f"{i}.dcm" for i in range(3)]
    write_dicom(p1)
    write_dicom(p2, photo="MONOCHROME1")
    write_dicom(p3, constant=True)
    normal, inverted = np.asarray(dicom_to_pil(p1)), np.asarray(dicom_to_pil(p2))
    assert normal.shape == (16, 16, 3)
    assert normal[0, 0, 0] == 0 and inverted[0, 0, 0] == 255
    assert np.isfinite(np.asarray(dicom_to_pil(p3))).all()


def test_multiframe_is_rejected(tmp_path):
    path = tmp_path / "multiframe.dcm"
    write_dicom(path, frames=2)
    with pytest.raises(ValueError, match="single-frame"):
        dicom_to_pil(path)
