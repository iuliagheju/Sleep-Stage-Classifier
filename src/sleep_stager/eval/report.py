"""Aggregate metrics across runs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import csv
import json
import numpy as np

METRIC_KEYS = ("macro_f1", "accuracy", "balanced_accuracy", "hmm_macro_f1")
CALIBRATION_KEYS = ("ece", "brier", "nll")
TEMPORAL_KEYS = ("implausible_transition_rate_raw", "implausible_transition_rate_hmm")
SUMMARY_METRICS = ("macro_f1", "accuracy", "balanced_accuracy")
COMPARISON_METRICS = ("macro_f1", "hmm_macro_f1")
COMPARISON_CALIBRATION = ("ece", "brier", "nll")
COMPARISON_TEMPORAL = ("implausible_transition_rate_raw", "implausible_transition_rate_hmm")
COMPARISON_EFFICIENCY = ("train_time_sec", "infer_epochs_per_sec", "param_count")


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


def aggregate_loso_metrics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        raise ValueError("At least one run is required to aggregate metrics.")
    summary: Dict[str, Any] = {
        "num_folds": len(runs),
        "run_ids": [run.get("run_id") for run in runs],
        "held_out_subject_ids": [run.get("held_out_subject_id") for run in runs],
        "folds": [
            {
                "run_id": run.get("run_id"),
                "held_out_subject_id": run.get("held_out_subject_id"),
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


def build_dataset_summary(runs: List[Dict[str, Any]], dataset_root: str) -> Dict[str, Any]:
    if not runs:
        raise ValueError("At least one run is required to build a summary.")
    run_rows = []
    for run in runs:
        metrics = run.get("metrics", {})
        run_rows.append(
            {
                "run_id": run.get("run_id"),
                "model_name": run.get("model_name"),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
                "git_commit": run.get("git_commit", "unknown"),
                "git_dirty": bool(run.get("git_dirty", False)),
                "config_hash": run.get("config_hash", ""),
                "spec_hash": run.get("spec_hash", ""),
            }
        )
    aggregate = {}
    for key in SUMMARY_METRICS:
        values = [row[key] for row in run_rows]
        aggregate[key] = _aggregate_values(values)
    summary = {
        "dataset_root": dataset_root,
        "generated_utc": datetime.utcnow().isoformat(),
        "run_count": len(run_rows),
        "run_ids": [row["run_id"] for row in run_rows],
        "model_names": sorted({row["model_name"] for row in run_rows if row["model_name"]}),
        "runs": run_rows,
        "aggregate": aggregate,
    }
    return summary


def write_dataset_summary(summary: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    csv_path = output_dir / "summary.csv"
    _write_summary_csv(summary.get("runs", []), csv_path)
    return summary_path


def _aggregate_values(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run_id",
        "model_name",
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "git_commit",
        "git_dirty",
        "config_hash",
        "spec_hash",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_comparison_report(runs: List[Dict[str, Any]], dataset_root: str) -> Dict[str, Any]:
    if not runs:
        raise ValueError("At least one run is required to build a comparison report.")
    class_order = _class_order_from_runs(runs)
    run_rows = []
    for run in runs:
        metrics = run.get("metrics", {})
        per_class = metrics.get("per_class_f1", {})
        calibration = run.get("calibration", {})
        temporal = run.get("temporal", {})
        row = {
            "run_id": run.get("run_id"),
            "model_name": run.get("model_name"),
            "metrics": {key: float(metrics.get(key, 0.0)) for key in COMPARISON_METRICS},
            "per_class_f1": {label: float(per_class.get(label, 0.0)) for label in class_order},
            "calibration": {key: float(calibration.get(key, 0.0)) for key in COMPARISON_CALIBRATION},
            "temporal": {key: float(temporal.get(key, 0.0)) for key in COMPARISON_TEMPORAL},
            "efficiency": {key: float(run.get(key, 0.0)) for key in COMPARISON_EFFICIENCY},
        }
        run_rows.append(row)
    return {
        "dataset_root": dataset_root,
        "generated_utc": datetime.utcnow().isoformat(),
        "run_count": len(run_rows),
        "class_order": class_order,
        "runs": run_rows,
    }


def write_comparison_report(report: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2))
    csv_path = output_dir / "comparison.csv"
    _write_comparison_csv(report, csv_path)
    return report_path


def _class_order_from_runs(runs: Iterable[Dict[str, Any]]) -> List[str]:
    for run in runs:
        cm = run.get("metrics", {}).get("confusion_matrix", {})
        order = cm.get("class_order")
        if order:
            return list(order)
    for run in runs:
        per_class = run.get("metrics", {}).get("per_class_f1", {})
        if per_class:
            return list(per_class.keys())
    return []


def _write_comparison_csv(report: Dict[str, Any], path: Path) -> None:
    class_order = report.get("class_order", [])
    fieldnames = [
        "run_id",
        "model_name",
        *COMPARISON_METRICS,
        *[f"f1_{label}" for label in class_order],
        *COMPARISON_CALIBRATION,
        *COMPARISON_TEMPORAL,
        *COMPARISON_EFFICIENCY,
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for run in report.get("runs", []):
            metrics = run.get("metrics", {})
            per_class = run.get("per_class_f1", {})
            calibration = run.get("calibration", {})
            temporal = run.get("temporal", {})
            efficiency = run.get("efficiency", {})
            row = {
                "run_id": run.get("run_id"),
                "model_name": run.get("model_name"),
            }
            row.update({key: metrics.get(key, 0.0) for key in COMPARISON_METRICS})
            row.update({f"f1_{label}": per_class.get(label, 0.0) for label in class_order})
            row.update({key: calibration.get(key, 0.0) for key in COMPARISON_CALIBRATION})
            row.update({key: temporal.get(key, 0.0) for key in COMPARISON_TEMPORAL})
            row.update({key: efficiency.get(key, 0.0) for key in COMPARISON_EFFICIENCY})
            writer.writerow(row)
