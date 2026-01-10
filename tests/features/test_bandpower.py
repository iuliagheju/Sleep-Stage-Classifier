from __future__ import annotations

import numpy as np

from sleep_stager.features.bandpower import BandConfig, compute_bandpower


def test_bandpower_shape():
    cfg = BandConfig(sample_rate=100)
    fake_epoch = np.random.randn(2, 3000)
    feats = compute_bandpower(fake_epoch, cfg)
    assert feats.shape == (2, len(cfg.bands))


def test_bandpower_detects_alpha_peak():
    cfg = BandConfig(sample_rate=100)
    duration = 30
    t = np.linspace(0, duration, cfg.sample_rate * duration, endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t).astype(np.float32)
    epoch = signal.reshape(1, -1)
    feats = compute_bandpower(epoch, cfg)
    alpha = feats[0, 2]
    others = np.delete(feats[0], 2)
    assert alpha > 100 * np.max(others)
