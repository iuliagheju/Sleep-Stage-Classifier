from __future__ import annotations

import numpy as np

from sleep_stager.features.time_domain import TIME_FEATURE_NAMES, compute_time_features


def test_time_domain_shape_and_finite():
    epoch = np.random.randn(2, 300).astype(np.float32)
    feats = compute_time_features(epoch)
    assert feats.shape == (2, len(TIME_FEATURE_NAMES))
    assert np.isfinite(feats).all()


def test_time_domain_zero_signal():
    epoch = np.zeros((2, 300), dtype=np.float32)
    feats = compute_time_features(epoch)
    np.testing.assert_allclose(feats, np.zeros((2, len(TIME_FEATURE_NAMES))))


def test_time_domain_deterministic():
    epoch = np.random.randn(2, 300).astype(np.float32)
    first = compute_time_features(epoch)
    second = compute_time_features(epoch)
    np.testing.assert_allclose(first, second)
