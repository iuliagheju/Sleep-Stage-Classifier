from __future__ import annotations

import pytest

from sleep_stager.cli.train import _resolve_protocol, _select_kfold_fold


def _spec(headline: str = "holdout", secondary: str = "kfold_subject") -> dict:
    return {
        "headline_protocol": {"name": headline},
        "secondary_protocol": {"name": secondary, "k": 5},
    }


def test_resolve_protocol_defaults_to_headline():
    assert _resolve_protocol(_spec(), None) == "holdout"


def test_resolve_protocol_secondary_alias():
    assert _resolve_protocol(_spec(), "secondary") == "kfold_subject"


def test_resolve_protocol_headline_alias():
    assert _resolve_protocol(_spec(), "headline") == "holdout"


def test_resolve_protocol_invalid_raises():
    with pytest.raises(ValueError):
        _resolve_protocol(_spec(), "unknown")


def test_select_kfold_fold_bounds():
    splits = [
        {"train": ["S00"], "val": ["S01"], "test": ["S02"]},
        {"train": ["S02"], "val": ["S01"], "test": ["S00"]},
    ]
    assert _select_kfold_fold(splits, 1) == splits[1]
    with pytest.raises(ValueError):
        _select_kfold_fold(splits, 2)
