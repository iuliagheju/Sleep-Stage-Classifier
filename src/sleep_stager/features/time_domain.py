"""Time-domain feature extraction utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np

TIME_FEATURE_NAMES: Tuple[str, ...] = (
    "mean",
    "std",
    "rms",
    "skew",
    "kurtosis",
    "zero_crossing_rate",
)


def compute_time_features(epoch: np.ndarray) -> np.ndarray:
    """Compute time-domain features for a single epoch.

    Args:
        epoch: Array shaped (channels, samples).
    Returns:
        Array shaped (channels, n_features) following TIME_FEATURE_NAMES.
    """
    mean = epoch.mean(axis=-1)
    std = epoch.std(axis=-1)
    rms = np.sqrt(np.mean(epoch**2, axis=-1))
    std_safe = np.where(std == 0, 1e-9, std)
    centered = epoch - mean[:, None]
    skew = np.mean((centered / std_safe[:, None]) ** 3, axis=-1)
    kurtosis = np.mean((centered / std_safe[:, None]) ** 4, axis=-1)
    if epoch.shape[-1] < 2:
        zcr = np.zeros(epoch.shape[0])
    else:
        sign = np.signbit(epoch)
        zcr = (np.diff(sign, axis=-1) != 0).mean(axis=-1)
    return np.stack([mean, std, rms, skew, kurtosis, zcr], axis=-1)


def aggregate_subject_time_features(epochs: np.ndarray) -> np.ndarray:
    feats = [compute_time_features(epoch) for epoch in epochs]
    return np.stack(feats, axis=0)
