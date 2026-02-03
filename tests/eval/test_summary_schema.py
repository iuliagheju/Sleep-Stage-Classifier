from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sleep_stager.eval.report import build_dataset_summary
from sleep_stager.eval.schema import validate_summary_schema


def _load_summary_schema() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    schema_path = repo_root / "specs" / "summary_schema.yaml"
    return yaml.safe_load(schema_path.read_text())


def _sample_runs() -> list[dict]:
    return [
        {
            "run_id": "run_1",
            "model_name": "classical",
            "metrics": {
                "macro_f1": 0.55,
                "accuracy": 0.6,
                "balanced_accuracy": 0.58,
            },
            "git_commit": "a" * 40,
            "git_dirty": False,
            "config_hash": "b" * 64,
            "spec_hash": "c" * 64,
        },
        {
            "run_id": "run_2",
            "model_name": "cnn",
            "metrics": {
                "macro_f1": 0.65,
                "accuracy": 0.7,
                "balanced_accuracy": 0.68,
            },
            "git_commit": "d" * 40,
            "git_dirty": True,
            "config_hash": "e" * 64,
            "spec_hash": "f" * 64,
        },
    ]


def test_summary_schema_valid():
    summary = build_dataset_summary(_sample_runs(), dataset_root="artifacts/sleep-cassette")
    schema = _load_summary_schema()
    fields = schema.get("summary_json", {}).get("fields", {})
    errors = validate_summary_schema(summary, fields)
    assert not errors


def test_summary_schema_missing_field_errors():
    summary = build_dataset_summary(_sample_runs(), dataset_root="artifacts/sleep-cassette")
    summary.pop("model_names")
    schema = _load_summary_schema()
    fields = schema.get("summary_json", {}).get("fields", {})
    errors = validate_summary_schema(summary, fields)
    assert errors
