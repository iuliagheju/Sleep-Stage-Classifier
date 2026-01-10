"""Metric computation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import json
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


@dataclass(slots=True)
class MetricConfig:
    labels: Iterable[str]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: MetricConfig,
) -> Dict[str, float]:
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=list(config.labels)),
        "micro_f1": f1_score(y_true, y_pred, average="micro", labels=list(config.labels)),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    return metrics


def confusion_matrix_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Iterable[str],
) -> Dict[str, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return {"matrix": cm, "labels": list(labels)}


def per_subject_metrics(
    subjects: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for subject in np.unique(subjects):
        mask = subjects == subject
        result[str(subject)] = {
            "macro_f1": f1_score(y_true[mask], y_pred[mask], average="macro", labels=list(labels)),
            "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
            "balanced_accuracy": balanced_accuracy_score(y_true[mask], y_pred[mask]),
        }
    return result


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, float]:
    scores = f1_score(y_true, y_pred, labels=labels, average=None)
    return {label: float(score) for label, score in zip(labels, scores)}


def dump_metrics(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
