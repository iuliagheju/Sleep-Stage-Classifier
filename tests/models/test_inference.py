from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from sleep_stager.models.inference import collect_probs


class LinearModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes, bias=False)
        nn.init.constant_(self.linear.weight, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GuardModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes, bias=False)
        nn.init.constant_(self.linear.weight, 0.2)
        self.saw_training = None
        self.saw_grad_enabled = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.saw_training = self.training
        self.saw_grad_enabled = torch.is_grad_enabled()
        return self.linear(x)


def _inputs(num_samples: int = 5, features: int = 4) -> np.ndarray:
    data = np.arange(num_samples * features, dtype=np.float32).reshape(num_samples, features)
    return data / 10.0


def test_collect_probs_batched_matches_unbatched():
    model = LinearModel(in_features=4, num_classes=3)
    inputs = _inputs()
    probs_batched = collect_probs(model, inputs, batch_size=2)
    probs_full = collect_probs(model, inputs, batch_size=None)
    np.testing.assert_allclose(probs_batched, probs_full, rtol=1e-6, atol=1e-7)


def test_collect_probs_enforces_eval_and_no_grad():
    model = GuardModel(in_features=4, num_classes=3)
    inputs = _inputs()
    _ = collect_probs(model, inputs, batch_size=3)
    assert model.saw_training is False
    assert model.saw_grad_enabled is False


def test_collect_probs_valid_simplex():
    model = LinearModel(in_features=4, num_classes=3)
    inputs = _inputs()
    probs = collect_probs(model, inputs, batch_size=2)
    assert np.isfinite(probs).all()
    assert (probs >= 0).all()
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6, atol=1e-7)
