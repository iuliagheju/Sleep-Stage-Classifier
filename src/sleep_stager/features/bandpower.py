"""Bandpower feature extraction using Welch PSD estimates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.signal import welch


@dataclass(slots=True)
class BandConfig:
    sample_rate: int
    bands: Tuple[Tuple[float, float], ...] = (
        (0.5, 4),  # delta
        (4, 8),  # theta
        (8, 12),  # alpha
        (12, 30),  # beta
    )


BandRatioNames: Tuple[str, ...] = ("delta_theta", "theta_alpha", "alpha_beta", "slow_fast")


def bandpower_feature_names(config: BandConfig) -> Tuple[str, ...]:
    return tuple(f"band_{low:g}_{high:g}" for low, high in config.bands)


def compute_bandpower(epoch: np.ndarray, config: BandConfig) -> np.ndarray:
    """Computes bandpower per channel for the configured bands.

    Args:
        epoch: Array shaped (channels, samples).
    Returns:
        Array shaped (channels, len(bands)).
    """
    freqs, psd = welch(epoch, fs=config.sample_rate, axis=-1, nperseg=256)
    features = []
    for low, high in config.bands:
        idx = np.logical_and(freqs >= low, freqs < high)
        features.append(psd[:, idx].mean(axis=-1))
    return np.stack(features, axis=-1)


def aggregate_subject_bandpower(epochs: np.ndarray, config: BandConfig) -> np.ndarray:
    feats = [compute_bandpower(epoch, config) for epoch in epochs]
    return np.stack(feats, axis=0)


def compute_bandpower_ratios(bandpower: np.ndarray) -> np.ndarray:
    if bandpower.shape[-1] < 4:
        raise ValueError("Bandpower ratios require at least 4 bands (delta/theta/alpha/beta)")
    delta = bandpower[:, 0]
    theta = bandpower[:, 1]
    alpha = bandpower[:, 2]
    beta = bandpower[:, 3]
    eps = 1e-9
    slow = delta + theta
    fast = alpha + beta
    ratios = np.stack(
        [
            delta / (theta + eps),
            theta / (alpha + eps),
            alpha / (beta + eps),
            slow / (fast + eps),
        ],
        axis=-1,
    )
    return ratios


def aggregate_subject_bandpower_ratios(bandpower_epochs: np.ndarray) -> np.ndarray:
    feats = [compute_bandpower_ratios(epoch) for epoch in bandpower_epochs]
    return np.stack(feats, axis=0)


def flatten_features(features: np.ndarray) -> np.ndarray:
    return features.reshape(features.shape[0], -1)
