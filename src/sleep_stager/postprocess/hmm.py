"""HMM post-processing with configurable transition constraints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(slots=True)
class HMMConfig:
    states: Sequence[str]
    transition_bias: float = 0.8
    min_prob: float = 1e-9
    prior_weight: float | None = None


def build_transition_matrix(
    config: HMMConfig, empirical: np.ndarray | None = None, min_transition_prob: float | None = None
) -> np.ndarray:
    num_states = len(config.states)
    if empirical is not None:
        mat = np.array(empirical, dtype=np.float64)
        mat = np.clip(mat, min_transition_prob or config.min_prob, None)
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return mat / row_sums
    trans = np.full((num_states, num_states), (1 - config.transition_bias) / (num_states - 1))
    np.fill_diagonal(trans, config.transition_bias)
    trans = np.clip(trans, config.min_prob, 1.0)
    trans /= trans.sum(axis=1, keepdims=True)
    return trans


def viterbi(log_probs: np.ndarray, transition: np.ndarray, initial: np.ndarray) -> np.ndarray:
    num_states = transition.shape[0]
    num_steps = log_probs.shape[0]
    dp = np.zeros((num_steps, num_states))
    backpointer = np.zeros((num_steps, num_states), dtype=int)
    dp[0] = initial + log_probs[0]
    for t in range(1, num_steps):
        for j in range(num_states):
            probs = dp[t - 1] + np.log(transition[:, j])
            backpointer[t, j] = int(np.argmax(probs))
            dp[t, j] = np.max(probs) + log_probs[t, j]
    states = np.zeros(num_steps, dtype=int)
    states[-1] = int(np.argmax(dp[-1]))
    for t in range(num_steps - 2, -1, -1):
        states[t] = backpointer[t + 1, states[t + 1]]
    return states


def smooth_sequence(
    probs: np.ndarray,
    label_to_idx: Dict[str, int],
    config: HMMConfig,
    transition: np.ndarray | None = None,
    initial: np.ndarray | None = None,
) -> List[str]:
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    transition_matrix = transition if transition is not None else build_transition_matrix(config)
    initial_log = (
        initial
        if initial is not None
        else np.log(np.ones(len(label_to_idx)) / len(label_to_idx))
    )
    log_probs = np.log(np.clip(probs, config.min_prob, 1.0))
    best_path = viterbi(log_probs, transition_matrix, initial_log)
    return [idx_to_label[idx] for idx in best_path]
