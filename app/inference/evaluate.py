"""Evaluation helpers for ChestXpert."""

from typing import Sequence

import numpy as np
import torch
import math
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

    if not targets:
        raise ValueError("Evaluation loader contains no samples")
    return np.concatenate(targets), np.concatenate(probabilities)


def validate_arrays(y_true, y_prob):
    truth, probability = np.asarray(y_true), np.asarray(y_prob)
    if truth.ndim != 2 or truth.shape != probability.shape or not all(truth.shape):
        raise ValueError("Targets and probabilities must have matching non-empty [samples, labels] shapes")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Targets must be binary")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    return truth, probability


def per_label_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: Sequence[str],
    threshold: float = 0.5,
):
    y_true, y_prob = validate_arrays(y_true, y_prob)
    if len(labels) != y_true.shape[1] or len(set(labels)) != len(labels):
        raise ValueError("Labels must be unique and match the output dimension")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    rows = []

    for index, label in enumerate(labels):
        truth = y_true[:, index]
        probability = y_prob[:, index]
        prediction = (probability >= threshold).astype(int)

        positives = int(truth.sum())
        negatives = len(truth) - positives
        auroc = float(roc_auc_score(truth, probability)) if positives and negatives else None
        ap = float(average_precision_score(truth, probability)) if positives else None

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
                "positive_count": positives,
                "negative_count": negatives,
            }
        )

    return rows
