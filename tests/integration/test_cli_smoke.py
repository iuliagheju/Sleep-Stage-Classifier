from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.smoke
def test_cli_smoke(smoke_run):
    latest = smoke_run["run_dir"]
    metrics = smoke_run["metrics"]

    # basic schema checks
    for key in [
        "run_id",
        "git_commit",
        "timestamp_utc",
        "dataset_name",
        "metrics",
        "calibration",
        "temporal",
        "held_out_subject_id",
        "gates",
        "environment",
        "features",
        "artifacts",
    ]:
        assert key in metrics
    features = metrics["features"]
    assert features["feature_dim"] > 0
    if features.get("feature_names"):
        assert len(features["feature_names"]) == features["feature_dim"]
    environment = metrics["environment"]
    assert "seeds" in environment
    assert "hardware" in environment
    gates = metrics["gates"]
    assert isinstance(gates["passed"], bool)

    # required artifact outputs
    artifacts_paths = metrics["artifacts"]
    required_files = [
        "per_subject_csv",
        "split_manifest_json",
        "confusion_matrix_png",
        "confusion_matrix_json",
        "confusion_matrix_hmm_png",
        "confusion_matrix_hmm_json",
        "reliability_diagram_png",
        "calibration_json",
        "predictions_raw_probs",
        "predictions_raw_pred_labels",
        "predictions_hmm_pred_labels",
        "hmm_transition_matrix_json",
        "model_checkpoint",
        "dataset_spec_yaml",
        "evaluation_spec_yaml",
    ]
    for key in required_files:
        path = latest / Path(artifacts_paths[key]).name
        assert path.exists(), f"Missing artifact {key}"

    # per-subject CSV contains expected columns
    per_subj = pd.read_csv(latest / Path(artifacts_paths["per_subject_csv"]).name)
    expected_cols = {
        "subject_id",
        "held_out_subject_id",
        "n_epochs",
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "ece",
        "brier",
        "nll",
        "implausible_transition_rate_raw",
        "implausible_transition_rate_hmm",
    }
    assert expected_cols.issubset(set(per_subj.columns))


def test_cli_help():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    cmd = [sys.executable, "-m", "sleep_stager.cli.main", "--help"]
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True)
    assert result.returncode == 0


@pytest.mark.smoke
def test_smoke_metrics_golden(smoke_run):
    repo_root = Path(__file__).resolve().parents[2]
    golden_path = repo_root / "tests" / "golden" / "metrics_smoke.json"
    expected = json.loads(golden_path.read_text())
    _assert_subset_close(smoke_run["metrics"], expected)


def test_cli_make_fixture(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    out_dir = tmp_path / "fixtures"
    cmd = [
        sys.executable,
        "-m",
        "sleep_stager.cli.main",
        "make-fixture",
        "--out",
        str(out_dir),
        "--num-subjects",
        "2",
        "--epochs-per-subject",
        "5",
        "--seed",
        "7",
    ]
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True)
    assert result.returncode == 0
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "manifest.csv").exists()


def test_cli_eval(smoke_run):
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    run_dir = smoke_run["run_dir"]
    cmd = [
        sys.executable,
        "-m",
        "sleep_stager.cli.main",
        "eval",
        "--run",
        str(run_dir),
    ]
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True)
    assert result.returncode == 0
    assert (run_dir / "metrics_eval.json").exists()


@pytest.mark.smoke
def test_cli_compare_runs(smoke_run):
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    runs_root = smoke_run["run_dir"].parent
    cmd = [
        sys.executable,
        "-m",
        "sleep_stager.cli.main",
        "compare-runs",
        "--runs-root",
        str(runs_root),
    ]
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True)
    assert result.returncode == 0
    report_dir = runs_root / "_comparison"
    report_path = report_dir / "comparison.json"
    csv_path = report_dir / "comparison.csv"
    assert report_path.exists()
    assert csv_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["run_count"] >= 1
    row = payload["runs"][0]
    assert "metrics" in row
    assert "per_class_f1" in row
    assert "calibration" in row
    assert "temporal" in row
    assert "efficiency" in row


@pytest.mark.smoke
def test_cli_train_loso_runs_all_subjects(tmp_path, fixture_dataset):
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["DATA_DIR"] = str(fixture_dataset)
    env["PYTHONPATH"] = str(repo_root)
    artifacts_root = tmp_path / "artifacts"
    cmd = [
        sys.executable,
        "-m",
        "sleep_stager.cli.main",
        "train-loso",
        "--config-path",
        str(repo_root / "configs"),
        "--config-name",
        "default",
        "--override",
        f"artifacts.root={artifacts_root}",
        "--override",
        "model.family=classical",
        "--override",
        "evaluation.enforce_gates=false",
    ]
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True)
    assert result.returncode == 0
    run_dirs = sorted([path for path in artifacts_root.iterdir() if path.is_dir()])
    assert len(run_dirs) == 3


def test_cli_fetch_data_validate(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    dest = tmp_path / "sleep-edfx-1.0.0" / "sleep-telemetry"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ST0001J0-PSG.edf").write_text("psg")
    (dest / "ST0001JP-Hypnogram.edf").write_text("hyp")
    cmd = [
        sys.executable,
        "-m",
        "sleep_stager.cli.main",
        "fetch-data",
        "--dest",
        str(dest.parent),
    ]
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True)
    assert result.returncode == 0


def _assert_subset_close(actual, expected, rel: float = 1e-5, abs: float = 1e-6) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, value in expected.items():
            assert key in actual, f"Missing key {key}"
            _assert_subset_close(actual[key], value, rel=rel, abs=abs)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_subset_close(actual_item, expected_item, rel=rel, abs=abs)
        return
    if isinstance(expected, (int, float)):
        assert actual == pytest.approx(expected, rel=rel, abs=abs)
        return
    assert actual == expected
