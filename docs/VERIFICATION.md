# Engineering verification

Validated on Windows, Python 3.10, 18 October 2026. The verification environment used
PyTorch 2.14.1, torchvision 0.29.1, NumPy 1.26.4 and pydicom 3.0.2.

`python -m pytest tests -q` passed 30 tests (plus 6 threshold subtests). Coverage includes:

- Save and restricted-load a real ResNet-18 checkpoint with synthetic weights.
- Deterministic CPU inference, finite output validation and label-schema validation.
- Real ResNet Grad-CAM generation and cleanup of hooks after repeated explanations.
- Multipart API uploads on Windows, invalid image content, missing model, oversized
  requests, unsupported extensions, private error responses and temporary-file cleanup.
- Isolated Flask app instances and separately testable health/readiness endpoints.
- Synthetic DICOM pixel decoding, MONOCHROME1 inversion, constant pixels and rejection
  of unsupported multiframe images. No patient data is used.
- Finite probability validation, undefined single-class metrics and explicit support.
- Training/validation/test image and patient isolation and checkpoint split provenance.

No trained checkpoint or research dataset was available in the checkout. Historical
README metrics are retained as historical claims and were not verified in this run.
The tests neither train a useful model nor validate medical accuracy. DICOM conversion
still uses percentile normalization; it is not a clinical presentation-state renderer.
Compressed DICOM requires a supported pixel decoder. Actual GPU inference, optional
Streamlit UI and Docker builds were not exercised in this run.

References: [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html),
[Flask upload limits](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/).
