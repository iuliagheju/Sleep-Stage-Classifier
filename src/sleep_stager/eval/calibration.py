"""Calibration metrics and visualization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import json
import matplotlib.pyplot as plt
import numpy as np


@dataclass(slots=True)
class CalibrationConfig:
    num_bins: int = 15


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, config: CalibrationConfig) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, config.num_bins + 1)
    ece = 0.0
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if not np.any(mask):
            continue
        accuracy = (y_true[mask] == predictions[mask]).mean()
        avg_conf = confidences[mask].mean()
        ece += np.abs(avg_conf - accuracy) * mask.mean()
    return float(ece)


def brier_score(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    one_hot = np.eye(num_classes)[y_true]
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def reliability_diagram(y_true: np.ndarray, probs: np.ndarray, config: CalibrationConfig, path: Path) -> None:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, config.num_bins + 1)
    accuracies = []
    mean_conf = []
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.any(mask):
            accuracies.append((y_true[mask] == predictions[mask]).mean())
            mean_conf.append(confidences[mask].mean())
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.bar(mean_conf, np.array(accuracies) - np.array(mean_conf), width=0.05, bottom=mean_conf)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def dump_calibration_metrics(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
