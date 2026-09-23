import pandas as pd

from app.preprocessing.dataset import LABELS, expand_labels


def test_expand_labels_creates_all_targets():
    df = pd.DataFrame({
        "Image Index": ["a.png"],
        "Finding Labels": ["Atelectasis|Effusion"],
    })

    result = expand_labels(df)

    assert len(LABELS) == 14
    assert result.loc[0, "Atelectasis"] == 1.0
    assert result.loc[0, "Effusion"] == 1.0
    assert result.loc[0, "Pneumonia"] == 0.0
