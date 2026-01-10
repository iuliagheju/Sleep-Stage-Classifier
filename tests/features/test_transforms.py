from __future__ import annotations

import numpy as np

from sleep_stager.data.ingest import SubjectRecord
from sleep_stager.features.bandpower import BandConfig, BandRatioNames
from sleep_stager.features.spectral import SPECTRAL_FEATURE_NAMES
from sleep_stager.features.time_domain import TIME_FEATURE_NAMES
from sleep_stager.features.transforms import FeatureConfig, build_feature_matrix


def test_feature_matrix_with_bandpower_and_time_domain():
    signals = np.random.randn(3, 2, 300).astype(np.float32)
    labels = np.array(["W", "N1", "N2"])
    subject = SubjectRecord(subject_id="S1", signals=signals, labels=labels, sample_rate=100)
    band_cfg = BandConfig(sample_rate=100)
    feature_cfg = FeatureConfig(use_bandpower=True, use_time_domain=True)
    X, y, subject_index = build_feature_matrix([subject], band_cfg, feature_cfg)
    expected = signals.shape[1] * (len(band_cfg.bands) + len(TIME_FEATURE_NAMES))
    assert X.shape == (signals.shape[0], expected)
    assert y.shape == (signals.shape[0],)
    assert subject_index.shape == (signals.shape[0],)


def test_feature_matrix_time_domain_only():
    signals = np.random.randn(4, 2, 300).astype(np.float32)
    labels = np.array(["W", "N1", "N2", "N3"])
    subject = SubjectRecord(subject_id="S1", signals=signals, labels=labels, sample_rate=100)
    band_cfg = BandConfig(sample_rate=100)
    feature_cfg = FeatureConfig(use_bandpower=False, use_time_domain=True)
    X, y, subject_index = build_feature_matrix([subject], band_cfg, feature_cfg)
    expected = signals.shape[1] * len(TIME_FEATURE_NAMES)
    assert X.shape == (signals.shape[0], expected)
    assert y.shape == (signals.shape[0],)
    assert subject_index.shape == (signals.shape[0],)


def test_feature_matrix_with_ratios_and_entropy():
    signals = np.random.randn(2, 2, 300).astype(np.float32)
    labels = np.array(["W", "N1"])
    subject = SubjectRecord(subject_id="S1", signals=signals, labels=labels, sample_rate=100)
    band_cfg = BandConfig(sample_rate=100)
    feature_cfg = FeatureConfig(
        use_bandpower=True,
        use_time_domain=False,
        use_bandpower_ratios=True,
        use_spectral_entropy=True,
    )
    X, _, _ = build_feature_matrix([subject], band_cfg, feature_cfg)
    expected = signals.shape[1] * (
        len(band_cfg.bands) + len(BandRatioNames) + len(SPECTRAL_FEATURE_NAMES)
    )
    assert X.shape == (signals.shape[0], expected)
