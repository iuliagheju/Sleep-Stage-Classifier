from __future__ import annotations

import pytest

from sleep_stager.eval.report import aggregate_kfold_metrics


def test_aggregate_kfold_metrics_summary():
    runs = [
        {
            "run_id": "r1",
            "metrics": {
                "macro_f1": 0.5,
                "accuracy": 0.6,
                "balanced_accuracy": 0.55,
                "hmm_macro_f1": 0.52,
            },
            "calibration": {"ece": 0.1, "brier": 0.2, "nll": 0.3},
            "temporal": {
                "implausible_transition_rate_raw": 0.3,
                "implausible_transition_rate_hmm": 0.1,
            },
            "gates": {"passed": True, "errors": []},
        },
        {
            "run_id": "r2",
            "metrics": {
                "macro_f1": 0.7,
                "accuracy": 0.8,
                "balanced_accuracy": 0.75,
                "hmm_macro_f1": 0.71,
            },
            "calibration": {"ece": 0.05, "brier": 0.15, "nll": 0.25},
            "temporal": {
                "implausible_transition_rate_raw": 0.2,
                "implausible_transition_rate_hmm": 0.08,
            },
            "gates": {"passed": False, "errors": ["gate failed"]},
        },
    ]

    summary = aggregate_kfold_metrics(runs)
    assert summary["num_folds"] == 2
    assert summary["metrics"]["macro_f1"]["mean"] == pytest.approx(0.6)
    assert summary["metrics"]["macro_f1"]["n"] == 2
    assert summary["calibration"]["ece"]["mean"] == pytest.approx(0.075)
    assert summary["gates"]["passed_all"] is False
    assert summary["gates"]["failed_folds"][0]["run_id"] == "r2"
