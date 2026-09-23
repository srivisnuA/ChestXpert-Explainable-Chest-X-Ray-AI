# ChestXpert — Explainable Chest X-Ray AI

ChestXpert is a research/educational prototype for **multi-label chest X-ray abnormality classification with explainability**. It demonstrates an end-to-end healthcare-AI engineering workflow: dataset preparation, transfer learning, reproducible training, evaluation, Grad-CAM explanations, DICOM handling, REST API integration, and a Streamlit clinician-review prototype.

> **Medical safety:** ChestXpert is **not a medical device** and must not be used to diagnose, triage, or treat patients. Predictions and heatmaps require qualified clinician review. The development-subset metrics below are engineering validation results, not clinical performance claims.

## Why this project

The project was built to explore practical components relevant to medical-imaging AI:

- Multi-label abnormality prediction rather than single-class classification
- Transfer learning with PyTorch/ResNet-18
- Per-finding AUROC, average precision, precision, recall and F1
- Explainability with Grad-CAM
- DICOM/PNG/JPEG input
- Flask REST API for downstream integration
- Streamlit research/demo interface
- Reproducible training and evaluation configuration
- Docker deployment scaffold

## Architecture

```text
Chest X-ray / DICOM
        |
        v
Image preprocessing
(224x224 + ImageNet normalization)
        |
        v
ResNet-18 transfer-learning model
        |
        v
14 abnormality probabilities
        |
   +----+------------------+
   |                       |
   v                       v
Evaluation             Grad-CAM
AUROC / AP             visual explanation
Precision / Recall/F1
   |                       |
   +-----------+-----------+
               v
        Flask REST API
               |
               v
       Streamlit demo UI
```

## Abnormalities

The baseline predicts 14 NIH ChestX-ray14 labels:

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax.

## Dataset

The initial experiment uses a **2,000-image development subset** prepared from NIH ChestX-ray14:

- 1,582 development-train images
- 418 held-out development-test images
- 14 multi-label findings
- No patient-identifiable dataset files are committed to the repository

This subset is intentionally used for rapid engineering validation. It is **not the official full-dataset benchmark** and is not sufficient to establish clinical validity.

## Training

The model uses a torchvision ResNet-18 initialized with ImageNet weights and replaces the final layer with a 14-output classification head.

Example:

```powershell
$env:PYTHONPATH = (Get-Location).Path

python scripts/train.py `
  --data-root data/raw/NIH_ChestXray14_subset `
  --epochs 1 `
  --batch-size 16 `
  --num-workers 0
```

The completed development training run produced:

```text
Train images: 1,582
Held-out test images: 418
Device: cpu
Epoch 01/1 - train_loss=1.2652
```

Checkpoint:

```text
models/checkpoints/resnet18_latest.pt
```

## Held-out development evaluation

The trained checkpoint was evaluated on the 418-image held-out development subset.

| Finding | AUROC | Average Precision | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Atelectasis | 0.575 | 0.099 | 0.096 | 0.735 | 0.170 |
| Cardiomegaly | 0.740 | 0.183 | 0.107 | 0.970 | 0.193 |
| Consolidation | 0.764 | 0.135 | 0.129 | 0.480 | 0.203 |
| Edema | 0.873 | 0.292 | 0.201 | 0.933 | 0.331 |
| Effusion | 0.677 | 0.133 | 0.134 | 0.676 | 0.223 |
| Emphysema | 0.807 | 0.151 | 0.058 | 0.909 | 0.109 |
| Fibrosis | 0.751 | 0.079 | 0.055 | 0.545 | 0.100 |
| Hernia | 0.990 | 0.200 | 0.011 | 1.000 | 0.022 |
| Infiltration | 0.641 | 0.362 | 0.278 | 0.772 | 0.409 |
| Mass | 0.669 | 0.057 | 0.062 | 0.800 | 0.114 |
| Nodule | 0.518 | 0.050 | 0.040 | 0.385 | 0.073 |
| Pleural Thickening | 0.820 | 0.083 | 0.044 | 0.900 | 0.084 |
| Pneumonia | 0.805 | 0.025 | 0.010 | 1.000 | 0.020 |
| Pneumothorax | 0.721 | 0.153 | 0.156 | 0.500 | 0.238 |

The fixed 0.5 threshold is intentionally reported as a baseline. The results show substantial class imbalance and a precision/recall trade-off, so the project includes a separate development threshold-analysis script.

### Important interpretation

These numbers **must not be described as clinical accuracy or clinical validation**. The development subset is small and the class distribution is imbalanced. In particular, high recall for some findings occurs alongside low precision.

The project therefore emphasizes transparent per-label evaluation rather than a single headline accuracy number.

## Threshold analysis

Run:

```powershell
python scripts/threshold_analysis.py `
  --checkpoint models/checkpoints/resnet18_latest.pt `
  --data-root data/raw/NIH_ChestXray14_subset `
  --batch-size 16 `
  --num-workers 0
```

This searches thresholds from 0.05 to 0.95 and reports the threshold producing the highest F1 for each label on the development subset.

Output:

```text
outputs/threshold_analysis.json
outputs/threshold_analysis.csv
```

These are **development thresholds for engineering analysis only**, not clinically calibrated thresholds.

## Calibration diagnostics

Run:

```powershell
python scripts/calibration.py `
  --checkpoint models/checkpoints/resnet18_latest.pt `
  --data-root data/raw/NIH_ChestXray14_subset `
  --batch-size 16 `
  --num-workers 0
```

This reports per-label Brier score and a simple expected calibration error (ECE). These diagnostics are included to make probability-quality limitations explicit; they are not clinical calibration.

Outputs:

```text
outputs/calibration.json
outputs/calibration.csv
```

## Grad-CAM explainability

ChestXpert can generate a visual explanation for a selected finding.

```powershell
python scripts/explain.py `
  --checkpoint models/checkpoints/resnet18_latest.pt `
  --image path/to/xray.png `
  --finding Edema
```

The script prints the selected finding probability and saves a heatmap overlay to:

```text
outputs/gradcam_edema.png
```

The implementation targets the final ResNet convolutional block. Grad-CAM highlights image regions associated with a model output; it does **not** prove that a highlighted region is pathology.

## DICOM support

Research-oriented DICOM loading is implemented with pydicom.

Supported demo inputs:

- PNG
- JPEG
- DICOM (.dcm)

The loader applies pixel-value rescaling, percentile normalization, and MONOCHROME1 handling before converting the image to RGB.

This is not a full clinical DICOM presentation-state or PACS implementation.

## REST API

Start Flask:

```bash
python app.py
```

Endpoints:

```text
GET  /api/health
POST /api/predict
```

The prediction endpoint accepts a multipart image upload and returns structured finding probabilities together with a research/clinical-review disclaimer.

The API structure is intended as a starting point for downstream research integration; it is not a production PACS/RIS integration.

## Streamlit demo

Run:

```bash
streamlit run frontend/app.py
```

The demo provides:

1. PNG/JPEG/DICOM upload
2. Model probability table
3. Configurable flagging threshold
4. Selected-finding Grad-CAM visualization
5. Explicit research/clinical-review warnings

## Project structure

```text
ChestXpert/
├── app/
│   ├── api/                 # Flask API routes
│   ├── explainability/     # Grad-CAM and overlays
│   ├── inference/          # model, prediction and metrics
│   └── preprocessing/      # dataset, transforms and DICOM
├── configs/                 # training configuration
├── data/                    # dataset documentation only
├── frontend/                # Streamlit demo
├── models/                  # checkpoint documentation
├── notebooks/               # notebook documentation
├── scripts/
│   ├── evaluate.py
│   ├── explain.py
│   ├── predict.py
│   ├── prepare_hf_subset.py
│   ├── prepare_metadata.py
│   ├── threshold_analysis.py
│   └── train.py
├── tests/
├── Dockerfile
├── app.py
├── requirements.txt
└── README.md
```

## Testing

Run:

```bash
pytest -q
```

Tests cover dataset behavior, metrics, Grad-CAM, inference preprocessing, and DICOM loading.

## API / integration direction

The architecture is intentionally modular so that future work can connect:

- DICOM/PACS ingestion
- structured abnormality findings
- clinician review workflows
- calibration and threshold policies
- larger public medical-imaging datasets
- downstream healthcare applications

## Responsible-use statement

ChestXpert is a portfolio/research prototype. It has not been clinically validated, approved as a medical device, or evaluated for patient-care use. Model outputs can be wrong and should not be used as a substitute for qualified clinical judgment.

## License / data

Do not commit downloaded medical datasets or patient-identifiable information. Follow the license and terms of use of any dataset used for experiments.
