from __future__ import annotations

from sleep_stager.eval.gates import evaluate_gates


def _gates_spec(min_f1: float) -> dict:
    return {
        "version": 1,
        "datasets": {
            "sleep-telemetry": {
                "performance": {
                    "classical": {"macro_f1_min": min_f1},
                }
            }
        },
    }


def _base_metrics(macro_f1: float) -> dict:
    return {
        "model_name": "classical",
        "metrics": {"macro_f1": macro_f1, "hmm_macro_f1": macro_f1},
        "temporal": {"implausible_transition_rate_raw": 0.0, "implausible_transition_rate_hmm": 0.0},
        "infer_epochs_per_sec": 1000,
        "param_count": 10,
    }


def test_gate_fails_below_threshold():
    metrics = _base_metrics(macro_f1=0.6)
    gates_spec = _gates_spec(min_f1=0.7)
    errors = evaluate_gates(
        metrics,
        gates_spec,
        model_limits=None,
        baseline_macro_f1=None,
        dataset_tag="sleep-telemetry",
        efficiency_budgets=None,
    )
    assert errors


def test_gate_passes_above_threshold():
    metrics = _base_metrics(macro_f1=0.8)
    gates_spec = _gates_spec(min_f1=0.7)
    errors = evaluate_gates(
        metrics,
        gates_spec,
        model_limits=None,
        baseline_macro_f1=None,
        dataset_tag="sleep-telemetry",
        efficiency_budgets=None,
    )
    assert not errors
