"""Evaluation gate checks for metrics and efficiency thresholds."""
from __future__ import annotations

from typing import Any, Dict, List


def evaluate_gates(
    metrics: Dict[str, Any],
    eval_spec: Dict[str, Any],
    model_limits: Dict[str, Any] | None = None,
    baseline_macro_f1: float | None = None,
) -> List[str]:
    errors: List[str] = []
    model_name = metrics.get("model_name", "")
    macro_f1 = float(metrics.get("metrics", {}).get("macro_f1", 0.0))
    hmm_macro_f1 = float(metrics.get("metrics", {}).get("hmm_macro_f1", macro_f1))

    gates = eval_spec.get("gates", {}) if eval_spec else {}
    _check_calibration(metrics, gates.get("calibration", {}), errors)
    _check_hmm(metrics, gates.get("hmm", {}), macro_f1, hmm_macro_f1, errors)
    _check_performance(metrics, gates.get("performance", {}), model_name, macro_f1, baseline_macro_f1, errors)
    _check_efficiency(metrics, eval_spec.get("efficiency_budgets", {}), errors)
    _check_param_caps(metrics, model_limits, model_name, errors)
    return errors


def _check_calibration(metrics: Dict[str, Any], cal_gates: Dict[str, Any], errors: List[str]) -> None:
    if not cal_gates:
        return
    calib = metrics.get("calibration", {})
    ece = float(calib.get("ece", 0.0))
    ece_before = float(calib.get("ece_before", 0.0))
    brier = float(calib.get("brier", 0.0))
    if "ece_after_max" in cal_gates and ece > float(cal_gates["ece_after_max"]):
        errors.append(f"calibration.ece {ece:.4f} > {cal_gates['ece_after_max']}")
    if "ece_before_max" in cal_gates and ece_before > float(cal_gates["ece_before_max"]):
        errors.append(f"calibration.ece_before {ece_before:.4f} > {cal_gates['ece_before_max']}")
    if "brier_after_max" in cal_gates and brier > float(cal_gates["brier_after_max"]):
        errors.append(f"calibration.brier {brier:.4f} > {cal_gates['brier_after_max']}")


def _check_hmm(
    metrics: Dict[str, Any],
    hmm_gates: Dict[str, Any],
    macro_f1: float,
    hmm_macro_f1: float,
    errors: List[str],
) -> None:
    if not hmm_gates:
        return
    drop = macro_f1 - hmm_macro_f1
    if "macro_f1_drop_max" in hmm_gates and drop > float(hmm_gates["macro_f1_drop_max"]):
        errors.append(f"hmm.macro_f1_drop {drop:.4f} > {hmm_gates['macro_f1_drop_max']}")
    temporal = metrics.get("temporal", {})
    raw_rate = float(temporal.get("implausible_transition_rate_raw", 0.0))
    hmm_rate = float(temporal.get("implausible_transition_rate_hmm", 0.0))
    if raw_rate > 0 and "implausible_transition_reduction_min" in hmm_gates:
        reduction = (raw_rate - hmm_rate) / raw_rate
        if reduction < float(hmm_gates["implausible_transition_reduction_min"]):
            errors.append(
                f"hmm.implausible_reduction {reduction:.4f} < {hmm_gates['implausible_transition_reduction_min']}"
            )


def _check_performance(
    metrics: Dict[str, Any],
    perf_gates: Dict[str, Any],
    model_name: str,
    macro_f1: float,
    baseline_macro_f1: float | None,
    errors: List[str],
) -> None:
    if not perf_gates:
        return
    if model_name == "classical":
        minimum = perf_gates.get("baseline_macro_f1_min")
        if minimum is not None and macro_f1 < float(minimum):
            errors.append(f"performance.baseline_macro_f1 {macro_f1:.4f} < {minimum}")
        return
    if model_name == "cnn":
        _check_model_thresholds(macro_f1, perf_gates.get("cnn", {}), baseline_macro_f1, "cnn", errors)
        return
    if model_name == "seq":
        _check_model_thresholds(
            macro_f1, perf_gates.get("cnn_bilstm", {}), baseline_macro_f1, "cnn_bilstm", errors
        )
        return
    if model_name == "attention":
        _check_model_thresholds(
            macro_f1, perf_gates.get("attention", {}), baseline_macro_f1, "attention", errors
        )


def _check_model_thresholds(
    macro_f1: float,
    gates: Dict[str, Any],
    baseline_macro_f1: float | None,
    name: str,
    errors: List[str],
) -> None:
    abs_min = gates.get("macro_f1_min_absolute")
    if abs_min is not None and macro_f1 < float(abs_min):
        errors.append(f"performance.{name}.macro_f1 {macro_f1:.4f} < {abs_min}")
    delta_min = gates.get("macro_f1_min_over_baseline")
    if baseline_macro_f1 is not None and delta_min is not None:
        if macro_f1 < baseline_macro_f1 + float(delta_min):
            errors.append(
                f"performance.{name}.macro_f1 {macro_f1:.4f} < baseline {baseline_macro_f1:.4f} + {delta_min}"
            )


def _check_efficiency(metrics: Dict[str, Any], budgets: Dict[str, Any], errors: List[str]) -> None:
    inference = budgets.get("inference", {}) if budgets else {}
    threshold = inference.get("cpu_epochs_per_sec_min")
    if threshold is None:
        return
    throughput = float(metrics.get("infer_epochs_per_sec", 0.0))
    if throughput < float(threshold):
        errors.append(f"efficiency.infer_epochs_per_sec {throughput:.2f} < {threshold}")


def _check_param_caps(
    metrics: Dict[str, Any],
    model_limits: Dict[str, Any] | None,
    model_name: str,
    errors: List[str],
) -> None:
    if not model_limits:
        return
    caps = model_limits.get("param_caps", {})
    cap = caps.get(model_name)
    if cap is None:
        return
    param_count = metrics.get("param_count")
    if param_count is None:
        return
    if int(param_count) > int(cap):
        errors.append(f"model.param_count {param_count} > {cap}")
