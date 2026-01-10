from __future__ import annotations

import numpy as np

from sleep_stager.models import cnn


def test_cnn_probability_simplex():
    signals = np.random.randn(6, 2, 128).astype(np.float32)
    labels = np.array(["W", "N1", "N2", "N3", "REM", "W"])
    label_to_idx = {label: idx for idx, label in enumerate(["W", "N1", "N2", "N3", "REM"])}
    model, _ = cnn.train_model(signals, labels, label_to_idx, cnn.CNNConfig(epochs=1, batch_size=2))
    _, probs = cnn.predict(model, signals)
    row_sums = probs.sum(axis=1)
    assert np.all(probs >= 0)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)
