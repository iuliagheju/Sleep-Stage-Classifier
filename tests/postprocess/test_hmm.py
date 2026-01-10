from __future__ import annotations

import numpy as np

from sleep_stager.postprocess.hmm import HMMConfig, build_transition_matrix, smooth_sequence, viterbi


def test_hmm_prefers_self_transitions():
    probs = np.array(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.3, 0.7],
        ]
    )
    label_to_idx = {"A": 0, "B": 1}
    cfg = HMMConfig(states=["A", "B"], transition_bias=0.9)
    smoothed = smooth_sequence(probs, label_to_idx, cfg)
    assert len(smoothed) == probs.shape[0]


def test_hmm_transition_matrix_normalized():
    cfg = HMMConfig(states=["A", "B", "C"], transition_bias=0.7, min_prob=1e-6)
    mat = build_transition_matrix(cfg)
    row_sums = mat.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums))


def test_viterbi_length_matches_input():
    probs = np.array(
        [
            [0.6, 0.4],
            [0.2, 0.8],
            [0.5, 0.5],
        ]
    )
    log_probs = np.log(probs)
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    initial = np.log(np.array([0.5, 0.5]))
    path = viterbi(log_probs, transition, initial)
    assert path.shape[0] == probs.shape[0]


def test_hmm_outputs_labels_in_vocab():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    label_to_idx = {"W": 0, "N1": 1}
    cfg = HMMConfig(states=["W", "N1"], transition_bias=0.9)
    smoothed = smooth_sequence(probs, label_to_idx, cfg)
    assert set(smoothed).issubset(label_to_idx.keys())
