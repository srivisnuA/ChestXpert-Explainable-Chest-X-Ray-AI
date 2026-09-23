"""Evaluation metrics for multi-label chest X-ray classification."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def multilabel_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    result = {}

    try:
        result["macro_auroc"] = float(
            roc_auc_score(y_true, y_prob, average="macro")
        )
    except ValueError:
        result["macro_auroc"] = None

    try:
        result["macro_average_precision"] = float(
            average_precision_score(y_true, y_prob, average="macro")
        )
    except ValueError:
        result["macro_average_precision"] = None

    return result
