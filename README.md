# ChestXpert — Explainable Chest X-Ray AI

ChestXpert is a research/educational prototype for chest X-ray abnormality classification and visual model explainability. It is designed as an end-to-end medical-imaging project covering preprocessing, PyTorch inference, evaluation, explainability and API integration.

> **Medical safety:** This project is not a medical device and must not be used to diagnose, triage or treat patients. Model outputs require qualified clinician review.

## Planned pipeline

Chest X-ray → preprocessing → PyTorch model → abnormality probabilities → evaluation → visual explanation → REST API → demo interface.

## Planned capabilities

- Chest X-ray image validation and preprocessing
- Transfer-learning-based PyTorch classifier
- Multi-label abnormality prediction
- Precision, recall, F1 and AUROC evaluation
- Class-imbalance analysis
- Grad-CAM-style visual explanations
- Confidence-aware inference
- Flask REST API
- Dockerized deployment
- Reproducible training/evaluation workflow

## Repository structure

```
ChestXpert/
├── app/
│   ├── api/
│   ├── inference/
│   ├── preprocessing/
│   └── explainability/
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
├── scripts/
├── tests/
├── frontend/
├── app.py
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## Project status

🚧 Phase 1 — repository scaffold.

Next phases will add the public dataset configuration, preprocessing pipeline, model training, held-out evaluation, explainability and API.

## Dataset and privacy

Research datasets will be downloaded separately and will **not** be committed to this repository. No patient-identifiable information should be added to the repository.

Dataset-specific licenses and usage conditions remain applicable to their respective sources.

## Disclaimer

This software is intended for research and educational demonstration only. It does not provide medical advice or a clinical diagnosis.
