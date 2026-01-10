from __future__ import annotations

import numpy as np
import pytest

from sleep_stager.models import attention, classical, cnn, seq


def _toy_signals(num_samples: int = 20) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    signals = np.random.randn(num_samples, 2, 200).astype(np.float32)
    labels = np.array(["W", "N1", "N2", "N3", "REM"] * (num_samples // 5))
    label_to_idx = {label: idx for idx, label in enumerate(["W", "N1", "N2", "N3", "REM"])}
    return signals, labels, label_to_idx


@pytest.mark.parametrize("model_type", ["logreg", "svm", "rf"])
def test_classical_pipeline(model_type):
    X = np.random.randn(30, 4)
    y = np.array(["W", "N1", "N2"] * 10)
    cfg = classical.ClassicalConfig(model_type=model_type, max_iter=50, n_estimators=25)
    model = classical.train_model(X, y, cfg)
    preds, probs = classical.predict(model, X)
    assert preds.shape[0] == X.shape[0]
    assert probs.shape[1] == len(model.classes_)
    assert np.all(probs >= 0)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(probs.shape[0]), atol=1e-6)


def test_cnn_training_forward():
    signals, labels, label_to_idx = _toy_signals()
    model, history = cnn.train_model(signals, labels, label_to_idx, cnn.CNNConfig(epochs=1, batch_size=4))
    assert history
    preds, probs = cnn.predict(model, signals)
    assert probs.shape[0] == signals.shape[0]
    assert np.all(probs >= 0)


def test_seq_training_forward():
    signals, labels, label_to_idx = _toy_signals()
    model, history = seq.train_model(
        signals,
        labels,
        label_to_idx,
        seq.SeqConfig(epochs=1, batch_size=4, window_size=3, embed_dim=16, hidden_size=16),
    )
    assert history
    preds, probs = seq.predict(model, signals)
    assert probs.shape[0] == signals.shape[0]
    assert np.all(probs >= 0)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(probs.shape[0]), atol=1e-5)


def test_attention_training_forward():
    signals, labels, label_to_idx = _toy_signals()
    model, history = attention.train_model(
        signals,
        labels,
        label_to_idx,
        attention.AttentionConfig(epochs=1, batch_size=4, window_size=5, embed_dim=16, hidden_size=16),
    )
    assert history
    preds, probs = attention.predict(model, signals)
    assert probs.shape[0] == signals.shape[0]
