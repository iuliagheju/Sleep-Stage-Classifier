from __future__ import annotations

import numpy as np

from sleep_stager.eval.baseline import compute_classical_baseline_metrics
from sleep_stager.models.classical import ClassicalConfig


def test_compute_classical_baseline_metrics():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(20, 4))
    y_train = np.array(["W", "N1", "N2", "N3", "REM"] * 4)
    X_test = rng.normal(size=(10, 4))
    y_test = np.array(["W", "N1", "N2", "N3", "REM"] * 2)
    metrics = compute_classical_baseline_metrics(
        X_train, y_train, X_test, y_test, ClassicalConfig(max_iter=50), ["W", "N1", "N2", "N3", "REM"]
    )
    assert "macro_f1" in metrics
    assert 0.0 <= float(metrics["macro_f1"]) <= 1.0
