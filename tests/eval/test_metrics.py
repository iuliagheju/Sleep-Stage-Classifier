from __future__ import annotations

import numpy as np

from sleep_stager.eval.metrics import (
    MetricConfig,
    compute_classification_metrics,
    confusion_matrix_artifacts,
    per_subject_metrics,
)


def test_macro_f1_per_subject():
    y_true = np.array(["W", "N1", "N2", "W"])
    y_pred = np.array(["W", "N2", "N2", "N1"])
    subjects = np.array(["S1", "S1", "S2", "S2"])
    cfg = MetricConfig(labels=["W", "N1", "N2"])
    metrics = compute_classification_metrics(y_true, y_pred, cfg)
    assert "macro_f1" in metrics
    per_subject = per_subject_metrics(subjects, y_true, y_pred, cfg.labels)
    assert set(per_subject.keys()) == {"S1", "S2"}


def test_macro_f1_matches_expected_value():
    y_true = np.array(["W", "N1", "N2", "W"])
    y_pred = np.array(["W", "N2", "N2", "N1"])
    cfg = MetricConfig(labels=["W", "N1", "N2"])
    metrics = compute_classification_metrics(y_true, y_pred, cfg)
    assert np.isclose(metrics["macro_f1"], 4 / 9)


def test_confusion_matrix_counts():
    y_true = np.array(["W", "N1", "N2", "W"])
    y_pred = np.array(["W", "N2", "N2", "N1"])
    labels = ["W", "N1", "N2"]
    cm = confusion_matrix_artifacts(y_true, y_pred, labels)
    expected = np.array(
        [
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(cm["matrix"], expected)
