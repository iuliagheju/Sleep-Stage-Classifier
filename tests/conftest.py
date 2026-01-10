from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixture_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path_factory.mktemp("fixtures")
    script = repo_root / "scripts" / "make_fixture_subset.py"
    subprocess.run([sys.executable, str(script), "--output-dir", str(out_dir)], check=True)
    return out_dir


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory: pytest.TempPathFactory, fixture_dataset: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_root = tmp_path_factory.mktemp("artifacts")
    env = os.environ.copy()
    env["DATA_DIR"] = str(fixture_dataset)
    env["PYTHONPATH"] = str(repo_root)
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
        f"artifacts.root={artifacts_root}",
        "--override",
        "model.family=classical",
    ]
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)
    run_dirs = sorted([p for p in artifacts_root.iterdir() if p.is_dir()])
    assert run_dirs, "Expected CLI to create artifact directory"
    latest = run_dirs[-1]
    metrics_path = latest / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    return {"run_dir": latest, "metrics": metrics}
