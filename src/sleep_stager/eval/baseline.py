"""Baseline evaluation helpers."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from ..models import classical
from .metrics import MetricConfig, compute_classification_metrics, per_class_f1


def compute_classical_baseline_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: classical.ClassicalConfig,
    stages: Iterable[str],
) -> Dict[str, float | dict]:
    model = classical.train_model(X_train, y_train, config)
    preds, _ = classical.predict(model, X_test)
    metric_cfg = MetricConfig(labels=list(stages))
    metrics = compute_classification_metrics(y_test, preds, metric_cfg)
    metrics["per_class_f1"] = per_class_f1(y_test, preds, list(stages))
    return metrics
