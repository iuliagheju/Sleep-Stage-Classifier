from __future__ import annotations

from sleep_stager.eval.gates import evaluate_gates


def test_gate_calibration_failure():
    metrics = {
        "model_name": "classical",
        "metrics": {"macro_f1": 0.9, "hmm_macro_f1": 0.9},
        "calibration": {"ece": 0.2, "ece_before": 0.2, "brier": 0.9},
        "temporal": {"implausible_transition_rate_raw": 0.0, "implausible_transition_rate_hmm": 0.0},
        "infer_epochs_per_sec": 1000,
        "param_count": 10,
    }
    eval_spec = {"gates": {"calibration": {"ece_after_max": 0.05, "ece_before_max": 0.1, "brier_after_max": 0.55}}}
    errors = evaluate_gates(metrics, eval_spec, model_limits=None, baseline_macro_f1=None)
    assert errors


def test_gate_param_cap():
    metrics = {
        "model_name": "cnn",
        "metrics": {"macro_f1": 0.9, "hmm_macro_f1": 0.9},
        "calibration": {"ece": 0.0, "ece_before": 0.0, "brier": 0.0},
        "temporal": {"implausible_transition_rate_raw": 0.0, "implausible_transition_rate_hmm": 0.0},
        "infer_epochs_per_sec": 1000,
        "param_count": 2_000_000,
    }
    model_limits = {"param_caps": {"cnn": 1_000_000}}
    errors = evaluate_gates(metrics, eval_spec={}, model_limits=model_limits, baseline_macro_f1=None)
    assert errors
