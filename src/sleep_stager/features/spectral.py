"""Frequency-domain feature extraction utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import welch

SPECTRAL_FEATURE_NAMES: Tuple[str, ...] = ("spectral_entropy",)


def compute_spectral_entropy(epoch: np.ndarray, sample_rate: int, nperseg: int = 256) -> np.ndarray:
    """Compute spectral entropy for a single epoch.

    Args:
        epoch: Array shaped (channels, samples).
    Returns:
        Array shaped (channels, 1).
    """
    _, psd = welch(epoch, fs=sample_rate, axis=-1, nperseg=nperseg)
    psd = np.clip(psd, 1e-12, None)
    psd_norm = psd / psd.sum(axis=-1, keepdims=True)
    entropy = -np.sum(psd_norm * np.log(psd_norm), axis=-1)
    entropy /= np.log(psd_norm.shape[-1])
    return entropy[:, None]


def aggregate_subject_spectral_entropy(epochs: np.ndarray, sample_rate: int, nperseg: int = 256) -> np.ndarray:
    feats = [compute_spectral_entropy(epoch, sample_rate=sample_rate, nperseg=nperseg) for epoch in epochs]
    return np.stack(feats, axis=0)
