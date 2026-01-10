"""Feature assembly utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..data.ingest import SubjectRecord
from .bandpower import (
    BandConfig,
    BandRatioNames,
    aggregate_subject_bandpower,
    aggregate_subject_bandpower_ratios,
    bandpower_feature_names,
    flatten_features,
)
from .spectral import SPECTRAL_FEATURE_NAMES, aggregate_subject_spectral_entropy
from .time_domain import TIME_FEATURE_NAMES, aggregate_subject_time_features


@dataclass(slots=True)
class FeatureConfig:
    use_bandpower: bool = True
    use_time_domain: bool = False
    use_bandpower_ratios: bool = False
    use_spectral_entropy: bool = False


def build_feature_matrix(
    subjects: List[SubjectRecord], band_config: BandConfig, feature_config: FeatureConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels: List[str] = []
    subject_index: List[str] = []
    for subject in subjects:
        band_feats = None
        band_ratios = None
        time_feats = None
        spectral_feats = None
        if feature_config.use_bandpower or feature_config.use_bandpower_ratios:
            band_feats = aggregate_subject_bandpower(subject.signals, band_config)
        if feature_config.use_bandpower_ratios and band_feats is not None:
            band_ratios = aggregate_subject_bandpower_ratios(band_feats)
        if feature_config.use_time_domain:
            time_feats = aggregate_subject_time_features(subject.signals)
        if feature_config.use_spectral_entropy:
            spectral_feats = aggregate_subject_spectral_entropy(subject.signals, band_config.sample_rate)
        combined: List[np.ndarray] = []
        if band_feats is not None and feature_config.use_bandpower:
            combined.append(band_feats)
        if band_ratios is not None:
            combined.append(band_ratios)
        if time_feats is not None:
            combined.append(time_feats)
        if spectral_feats is not None:
            combined.append(spectral_feats)
        if combined:
            subject_feats = flatten_features(np.concatenate(combined, axis=-1))
        else:
            subject_feats = subject.signals.reshape(subject.signals.shape[0], -1)
        features.append(subject_feats)
        labels.extend(subject.labels.tolist())
        subject_index.extend([subject.subject_id] * subject_feats.shape[0])
    return (
        np.concatenate(features, axis=0),
        np.array(labels),
        np.array(subject_index),
    )


def build_feature_names(
    num_channels: int, band_config: BandConfig, feature_config: FeatureConfig
) -> List[str]:
    base_names: List[str] = []
    if feature_config.use_bandpower:
        base_names.extend(bandpower_feature_names(band_config))
    if feature_config.use_bandpower_ratios:
        base_names.extend(list(BandRatioNames))
    if feature_config.use_time_domain:
        base_names.extend(list(TIME_FEATURE_NAMES))
    if feature_config.use_spectral_entropy:
        base_names.extend(list(SPECTRAL_FEATURE_NAMES))
    if not base_names:
        return []
    names: List[str] = []
    for ch in range(num_channels):
        for base in base_names:
            names.append(f"ch{ch}_{base}")
    return names
