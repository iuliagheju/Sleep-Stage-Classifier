"""Deterministic seeding helpers reused across the codebase."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedConfig:
    numpy_seed: int = 42
    torch_seed: int = 1337


def seed_everything(config: SeedConfig | None = None) -> SeedConfig:
    cfg = config or SeedConfig()
    random.seed(cfg.numpy_seed)
    np.random.seed(cfg.numpy_seed)
    torch.manual_seed(cfg.torch_seed)
    torch.cuda.manual_seed_all(cfg.torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(cfg.numpy_seed)
    return cfg
