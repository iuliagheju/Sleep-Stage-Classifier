from __future__ import annotations

import numpy as np
import torch

from sleep_stager.models.attention import AttentionNet


def test_attention_weights_shape():
    signals = np.random.randn(2, 2, 200).astype(np.float32)
    model = AttentionNet(channels=2, num_classes=5, window_size=3, embed_dim=8, hidden_size=8)
    window = np.stack([signals[0], signals[1], signals[0]], axis=0)[None, ...]
    logits, weights = model.forward_with_attention(torch.tensor(window, dtype=torch.float32))
    assert logits.shape == (1, 5)
    assert weights.shape == (1, 3)
    np.testing.assert_allclose(weights.sum(dim=1).detach().numpy(), np.ones(1), atol=1e-5)
