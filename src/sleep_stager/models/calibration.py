"""Calibration utilities (temperature scaling)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class TemperatureConfig:
    lr: float = 0.01
    max_steps: int = 200


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.clamp(self.temperature, min=1e-3)
        return logits / temperature


def fit_temperature(logits: np.ndarray, labels: np.ndarray, config: TemperatureConfig) -> TemperatureScaler:
    model = TemperatureScaler()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    inputs = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)

    def closure():  # type: ignore[override]
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        return loss

    for _ in range(config.max_steps):
        optimizer.step(closure)
    return model


def apply_temperature(model: TemperatureScaler, logits: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.tensor(logits, dtype=torch.float32)
        scaled = model(tensor)
        probs = torch.softmax(scaled, dim=-1)
        return probs.numpy()
