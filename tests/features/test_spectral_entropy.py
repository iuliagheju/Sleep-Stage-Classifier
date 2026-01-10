from __future__ import annotations

import numpy as np

from sleep_stager.features.spectral import compute_spectral_entropy


def test_spectral_entropy_sine_lower_than_noise():
    sample_rate = 100
    duration = 30
    t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    sine = np.sin(2 * np.pi * 10 * t).astype(np.float32)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(t.shape).astype(np.float32)
    entropy_sine = compute_spectral_entropy(sine.reshape(1, -1), sample_rate=sample_rate)[0, 0]
    entropy_noise = compute_spectral_entropy(noise.reshape(1, -1), sample_rate=sample_rate)[0, 0]
    assert entropy_sine < entropy_noise
    assert 0.0 <= entropy_sine <= 1.0
    assert 0.0 <= entropy_noise <= 1.0
