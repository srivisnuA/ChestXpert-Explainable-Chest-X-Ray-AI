# ChestXpert — Explainable Chest X-Ray AI

ChestXpert is a research/educational prototype for chest X-ray abnormality classification and visual model explainability. It is designed as an end-to-end medical-imaging project covering preprocessing, PyTorch inference, evaluation, explainability and API integration.

> **Medical safety:** This project is not a medical device and must not be used to diagnose, triage or treat patients. Model outputs require qualified clinician review.

## Pipeline

Chest X-ray → preprocessing → PyTorch model → abnormality probabilities → per-label evaluation → Grad-CAM explanation → REST API → demo interface.

## Current implementation

- NIH ChestX-ray14 multi-label dataset configuration
- ResNet-18 transfer-learning baseline
- Class-imbalance-aware BCE training objective
- Official train/test list separation
- Per-label AUROC, average precision, precision, recall and F1 evaluation
- Grad-CAM implementation for visual explanation
- Automated unit tests for dataset, metrics and explainability logic
- Flask service scaffold
- Docker deployment scaffold

## Evaluation

After training, run:

```bash
python scripts/evaluate.py --checkpoint models/checkpoints/resnet18_latest.pt
```

The evaluation writes:

- `outputs/test_metrics.json`
- `outputs/test_metrics.csv`

No clinical performance claim is made until the model is actually trained and evaluated on the held-out test split.

## Planned next steps

- Run reproducible training and record actual test metrics
- Add inference service with image upload
- Add clinician-oriented demo interface
- Add calibration analysis and threshold selection
- Add deployment documentation

## Important

Do not interpret model probability scores as a diagnosis. Grad-CAM visualizations indicate model-associated regions and are not proof of clinical causality.
