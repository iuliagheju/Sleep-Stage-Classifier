from __future__ import annotations

import numpy as np

from sleep_stager.models.seq import ContextWindowDataset


def test_context_window_center_alignment():
    signals = np.stack([np.full((1, 4), idx, dtype=np.float32) for idx in range(5)], axis=0)
    labels = np.array(["W", "N1", "N2", "N3", "REM"])
    label_to_idx = {label: i for i, label in enumerate(labels)}
    dataset = ContextWindowDataset(signals, labels, label_to_idx, window_size=3)
    window, label = dataset[0]
    assert label == 0
    values = window[:, 0, 0].numpy().tolist()
    assert values == [0.0, 0.0, 1.0]
