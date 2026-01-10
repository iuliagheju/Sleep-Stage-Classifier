from __future__ import annotations

import numpy as np

from sleep_stager.features.bandpower import BandConfig, compute_bandpower


def test_bandpower_deterministic_given_seed():
    rng = np.random.default_rng(123)
    epoch = rng.standard_normal((2, 3000))
    cfg = BandConfig(sample_rate=100)
    first = compute_bandpower(epoch, cfg)
    second = compute_bandpower(epoch, cfg)
    np.testing.assert_allclose(first, second)