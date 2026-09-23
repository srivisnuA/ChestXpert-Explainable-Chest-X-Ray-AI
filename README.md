# ChestXpert — Explainable Chest X-Ray AI

ChestXpert is a research/educational prototype for chest X-ray abnormality classification and visual model explainability. It is designed as an end-to-end medical-imaging project covering preprocessing, PyTorch training/inference, evaluation, Grad-CAM explainability, REST API integration and a Streamlit demo.

> **Medical safety:** This project is not a medical device and must not be used to diagnose, triage or treat patients. Model outputs require qualified clinician review.

## Pipeline

Chest X-ray → preprocessing → PyTorch model → abnormality probabilities → per-label evaluation → Grad-CAM explanation → REST API / Streamlit demo.

## Current implementation

- NIH ChestX-ray14 multi-label dataset configuration
- ResNet-18 transfer-learning baseline
- Class-imbalance-aware BCE training objective
- Official train/test list separation
- Per-label AUROC, average precision, precision, recall and F1 evaluation
- Grad-CAM implementation targeting the final ResNet convolutional block
- PIL-image inference API shared by CLI and Streamlit
- Streamlit demo with selectable Grad-CAM overlays
- Flask health and prediction endpoints
- Automated unit tests for dataset, metrics, explainability and preprocessing logic
- Docker deployment scaffold

## Local setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the processed metadata:

```bash
python scripts/prepare_metadata.py
```

Train the baseline:

```bash
python scripts/train.py --config configs/training_config.yaml
```

Run evaluation:

```bash
python scripts/evaluate.py --checkpoint models/checkpoints/resnet18_latest.pt
```

Run the CLI predictor:

```bash
python scripts/predict.py --checkpoint models/checkpoints/resnet18_latest.pt --image path/to/xray.png
```

Run the Flask API:

```bash
python app.py
```

Run the Streamlit demo:

```bash
streamlit run frontend/app.py
```

The Streamlit interface accepts an X-ray image, displays model probabilities, shows findings crossing the selected threshold, and generates a Grad-CAM overlay for a selected finding.

## API

- `GET /api/health` — service/model status
- `POST /api/predict` — multipart image inference

The API returns model outputs with an explicit research/clinical-review disclaimer.

## Evaluation

The evaluation script writes:

- `outputs/test_metrics.json`
- `outputs/test_metrics.csv`

No clinical performance claim is made until the model is actually trained and evaluated on the held-out test split. Reported metrics must come from a reproducible run rather than being estimated or copied from a dataset paper.

## Dataset

The first experiment is configured for NIH ChestX-ray14. The dataset and any patient-identifiable data must be downloaded separately and must not be committed to this repository. See `data/README.md` for the expected local layout.

## Explainability

Grad-CAM highlights regions that contribute to a selected model output. In ChestXpert, the final ResNet convolutional block is used as the target layer. The visualization is an interpretability aid and does not establish that a highlighted region represents a real pathology.

## Testing

Run:

```bash
pytest -q
```

## Roadmap

- Run reproducible training and record actual held-out test metrics
- Add calibration analysis and threshold selection
- Expand DICOM/pydicom support
- Add structured finding export suitable for downstream research workflows
- Add deployment documentation

## Important

Do not interpret model probability scores as a diagnosis. This repository is intended for research and portfolio demonstration, not clinical use.
