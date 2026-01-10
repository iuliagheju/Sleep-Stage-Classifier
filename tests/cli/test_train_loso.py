from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import sleep_stager.cli.main as main_cli


def test_train_loso_continues_on_gate_failure(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_train(*, config_path: str, config_name: str, override: list[str]) -> None:
        calls.append(list(override))
        if len(calls) == 1:
            raise ValueError("Evaluation gates failed:\nexample")

    cfg = OmegaConf.create(
        {
            "data": {"dir": "data/raw/sleep-casette"},
            "artifacts": {"root": str(tmp_path / "sleep-cassette" / "loso")},
            "evaluation": {"protocol": "loso"},
        }
    )

    monkeypatch.setattr(main_cli, "train_command", fake_train)
    monkeypatch.setattr(main_cli, "list_subject_ids", lambda _path: ["S1", "S2"])
    monkeypatch.setattr(main_cli, "_resolve_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(main_cli, "_hash_payload", lambda *_args, **_kwargs: "hash")
    monkeypatch.setattr(main_cli, "_existing_loso_run", lambda *_args, **_kwargs: False)

    main_cli.train_loso(
        config_path="configs",
        config_name="default",
        override=[],
        runs_root=Path(cfg.artifacts.root),
        max_subjects=None,
    )

    assert len(calls) == 2
