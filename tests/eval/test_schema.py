from __future__ import annotations

import json
from pathlib import Path

from sleep_stager.eval.schema import validate_metrics_schema


def test_schema_missing_fields():
    schema = {"run_id": "string", "metrics": {"macro_f1": "float"}}
    errors = validate_metrics_schema({"run_id": "abc"}, schema)
    assert errors


def test_schema_matches_smoke_metrics(smoke_run):
    repo_root = Path(__file__).resolve().parents[2]
    schema_path = repo_root / "specs" / "artifacts_schema.yaml"
    schema = json.loads(schema_path.read_text()) if schema_path.suffix == ".json" else None
    if schema is None:
        import yaml

        schema = yaml.safe_load(schema_path.read_text())
    fields = schema.get("metrics_json", {}).get("fields", {})
    errors = validate_metrics_schema(smoke_run["metrics"], fields)
    assert not errors
