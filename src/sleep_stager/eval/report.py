"""Aggregate metrics across k-fold runs."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

METRIC_KEYS = ("macro_f1", "accuracy", "balanced_accuracy", "hmm_macro_f1")
CALIBRATION_KEYS = ("ece", "brier", "nll")
TEMPORAL_KEYS = ("implausible_transition_rate_raw", "implausible_transition_rate_hmm")


def aggregate_kfold_metrics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        raise ValueError("At least one run is required to aggregate metrics.")
    summary: Dict[str, Any] = {
        "num_folds": len(runs),
        "run_ids": [run.get("run_id") for run in runs],
        "folds": [
            {
                "run_id": run.get("run_id"),
                "fold_id": run.get("fold_id"),
                "model_name": run.get("model_name"),
            }
            for run in runs
        ],
        "metrics": _aggregate_section(runs, "metrics", METRIC_KEYS),
        "calibration": _aggregate_section(runs, "calibration", CALIBRATION_KEYS),
        "temporal": _aggregate_section(runs, "temporal", TEMPORAL_KEYS),
        "gates": _aggregate_gates(runs),
    }
    return summary


def _aggregate_section(
    runs: Iterable[Dict[str, Any]], section: str, keys: Iterable[str]
) -> Dict[str, Dict[str, float]]:
    aggregated: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = _collect_values(runs, section, key)
        if values:
            aggregated[key] = _summary_stats(values)
    return aggregated


def _collect_values(runs: Iterable[Dict[str, Any]], section: str, key: str) -> List[float]:
    values: List[float] = []
    for run in runs:
        section_data = run.get(section, {})
        if key not in section_data:
            continue
        try:
            values.append(float(section_data[key]))
        except (TypeError, ValueError):
            continue
    return values


def _summary_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n": int(arr.size),
    }


def _aggregate_gates(runs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    failed = []
    for run in runs:
        gates = run.get("gates", {})
        passed = bool(gates.get("passed", True))
        if not passed:
            failed.append(
                {
                    "run_id": run.get("run_id"),
                    "errors": gates.get("errors", []),
                }
            )
    return {"passed_all": not failed, "failed_folds": failed}
