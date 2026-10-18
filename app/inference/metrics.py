"""Macro metrics over eligible labels; undefined metrics are JSON null."""
import numpy as np
from app.inference.evaluate import per_label_metrics, validate_arrays


def multilabel_metrics(y_true, y_prob):
    truth, probability = validate_arrays(y_true, y_prob)
    rows = per_label_metrics(truth, probability, [str(i) for i in range(truth.shape[1])])
    auroc = [row["auroc"] for row in rows if row["auroc"] is not None]
    ap = [row["average_precision"] for row in rows if row["average_precision"] is not None]
    return {
        "macro_auroc": float(np.mean(auroc)) if auroc else None,
        "macro_average_precision": float(np.mean(ap)) if ap else None,
        "auroc_label_count": len(auroc),
        "average_precision_label_count": len(ap),
        "label_count": truth.shape[1],
    }
