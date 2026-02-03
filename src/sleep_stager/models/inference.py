"""Inference helpers for batched probability collection."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def collect_probs(model: Any, inputs: np.ndarray, batch_size: int | None = 64) -> np.ndarray:
    model.eval()
    if inputs is None or len(inputs) == 0:
        return np.zeros((0, 0))
    if isinstance(inputs, np.ndarray):
        tensor = torch.tensor(inputs, dtype=torch.float32)
    else:
        tensor = inputs
    device = _model_device(model)
    tensor = tensor.to(device)
    with torch.no_grad():
        if batch_size is None or batch_size <= 0 or tensor.shape[0] <= batch_size:
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()
        batches = []
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start : start + batch_size]
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            batches.append(probs.cpu().numpy())
        return np.concatenate(batches, axis=0)


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
