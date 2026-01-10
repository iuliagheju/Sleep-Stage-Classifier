from __future__ import annotations

import numpy as np
import pytest

from sleep_stager.features.bandpower import BandRatioNames, compute_bandpower_ratios


def test_bandpower_ratios_values():
    bandpower = np.array([[2.0, 1.0, 0.5, 0.25]])
    ratios = compute_bandpower_ratios(bandpower)
    assert ratios.shape == (1, len(BandRatioNames))
    np.testing.assert_allclose(ratios[0, 0], 2.0)  # delta/theta
    np.testing.assert_allclose(ratios[0, 1], 2.0)  # theta/alpha
    np.testing.assert_allclose(ratios[0, 2], 2.0)  # alpha/beta
    np.testing.assert_allclose(ratios[0, 3], 4.0)  # (delta+theta)/(alpha+beta)


def test_bandpower_ratios_requires_four_bands():
    with pytest.raises(ValueError):
        compute_bandpower_ratios(np.ones((1, 3)))
