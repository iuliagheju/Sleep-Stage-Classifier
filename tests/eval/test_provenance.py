from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


def _has_git_commit(repo_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    if result.returncode != 0:
        return False
    return bool(re.fullmatch(r"[0-9a-f]{40}", result.stdout.strip()))


def test_metrics_provenance_keys(smoke_run):
    metrics_path = smoke_run["run_dir"] / "metrics.json"
    data = json.loads(metrics_path.read_text())
    for key in ("git_commit", "git_dirty", "config_hash", "spec_hash"):
        assert key in data
    assert isinstance(data["git_dirty"], bool)
    assert re.fullmatch(r"[0-9a-f]{64}", data["config_hash"])
    assert re.fullmatch(r"[0-9a-f]{64}", data["spec_hash"])

    repo_root = Path(__file__).resolve().parents[2]
    if _has_git_commit(repo_root):
        assert re.fullmatch(r"[0-9a-f]{40}", data["git_commit"])
