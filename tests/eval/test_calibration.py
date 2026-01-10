from __future__ import annotations

import numpy as np

from sleep_stager.eval.calibration import CalibrationConfig, brier_score, expected_calibration_error


def test_calibration_metrics():
    probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    y_true = np.array([0, 1])
    cfg = CalibrationConfig(num_bins=5)
    ece = expected_calibration_error(y_true, probs, cfg)
    brier = brier_score(y_true, probs, num_classes=2)
    assert 0 <= ece <= 1
    assert brier >= 0


def test_calibration_perfect_predictions_zero_error():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_true = np.array([0, 1])
    cfg = CalibrationConfig(num_bins=5)
    ece = expected_calibration_error(y_true, probs, cfg)
    brier = brier_score(y_true, probs, num_classes=2)
    assert np.isclose(ece, 0.0)
    assert np.isclose(brier, 0.0)
