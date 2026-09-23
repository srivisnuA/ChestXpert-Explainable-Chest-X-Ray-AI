import numpy as np

from app.inference.metrics import multilabel_metrics


def test_multilabel_metrics_returns_auroc():
    y_true = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    y_prob = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.7, 0.9],
        [0.1, 0.2],
    ])

    result = multilabel_metrics(y_true, y_prob)

    assert result["macro_auroc"] is not None
    assert 0.0 <= result["macro_auroc"] <= 1.0
