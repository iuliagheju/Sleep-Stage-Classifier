from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_train_rejects_dataset_mismatch(tmp_path, fixture_dataset):
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["DATA_DIR"] = str(fixture_dataset)
    env["PYTHONPATH"] = str(repo_root)
    mismatched_root = tmp_path / "sleep-telemetry"
    cmd = [
        sys.executable,
        "-m",
        "sleep_stager.cli.main",
        "train",
        "--config-path",
        str(repo_root / "configs"),
        "--config-name",
        "default",
        "--override",
        f"artifacts.root={mismatched_root}",
        "--override",
        "model.family=classical",
        "--override",
        "evaluation.enforce_gates=false",
    ]
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True)
    assert result.returncode != 0
    assert b"dataset" in result.stderr.lower()
