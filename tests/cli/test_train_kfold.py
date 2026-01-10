from __future__ import annotations

from pathlib import Path

import pytest

import sleep_stager.cli.main as main_cli


def test_train_kfold_invokes_all_folds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_train(*, config_path: str, config_name: str, override: list[str]) -> None:
        calls.append({"config_path": config_path, "config_name": config_name, "override": list(override)})

    monkeypatch.setattr(main_cli, "train_command", fake_train)
    runs_root = tmp_path / "runs"

    main_cli.train_kfold(
        config_path="configs",
        config_name="default",
        override=["model.family=classical"],
        runs_root=runs_root,
        max_folds=2,
        aggregate=False,
    )

    assert len(calls) == 2
    for idx, call in enumerate(calls):
        overrides = call["override"]
        assert f"evaluation.fold_index={idx}" in overrides
        assert any(item.startswith("evaluation.protocol=") for item in overrides)
        assert f"artifacts.root={runs_root}" in overrides


def test_train_kfold_rejects_too_many_folds() -> None:
    with pytest.raises(ValueError):
        main_cli.train_kfold(
            config_path="configs",
            config_name="default",
            override=[],
            runs_root=None,
            max_folds=999,
            aggregate=False,
        )
