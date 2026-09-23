"""Evaluation helpers for ChestXpert."""

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def collect_predictions(model, loader, device):
    model.eval()
    probabilities = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels.numpy())

    return np.concatenate(targets), np.concatenate(probabilities)


def per_label_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: Sequence[str],
    threshold: float = 0.5,
):
    rows = []

    for index, label in enumerate(labels):
        truth = y_true[:, index]
        probability = y_prob[:, index]
        prediction = (probability >= threshold).astype(int)

        try:
            auroc = float(roc_auc_score(truth, probability))
        except ValueError:
            auroc = None

        ap = float(average_precision_score(truth, probability))

        precision, recall, f1, _ = precision_recall_fscore_support(
            truth,
            prediction,
            average="binary",
            zero_division=0,
        )

        rows.append(
            {
                "label": label,
                "auroc": auroc,
                "average_precision": ap,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "threshold": threshold,
            }
        )

    return rows
