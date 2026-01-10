"""Temporal consistency utilities: transition estimation and implausibility checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(slots=True)
class TransitionRule:
    source: str
    target: str


def count_transitions(labels: Sequence[str], states: Sequence[str]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(states)}
    counts = np.zeros((len(states), len(states)), dtype=np.float64)
    for a, b in zip(labels[:-1], labels[1:]):
        if a not in idx or b not in idx:
            continue
        counts[idx[a], idx[b]] += 1
    return counts


def estimate_transition_matrix(
    labels: Sequence[str],
    states: Sequence[str],
    prior_weight: float = 0.0,
    min_prob: float = 1e-6,
) -> np.ndarray:
    counts = count_transitions(labels, states)
    prior = np.ones_like(counts) * prior_weight
    counts = counts + prior
    counts = np.clip(counts, min_prob, None)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def implausible_transition_rate(labels: Sequence[str], rules: Iterable[TransitionRule]) -> float:
    if len(labels) < 2:
        return 0.0
    rule_set = {(r.source, r.target) for r in rules}
    total = 0
    implausible = 0
    for a, b in zip(labels[:-1], labels[1:]):
        total += 1
        if (a, b) in rule_set:
            implausible += 1
    return implausible / total if total else 0.0


def per_subject_implausible_rates(
    subjects: np.ndarray, labels: np.ndarray, rules: Iterable[TransitionRule]
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for subj in np.unique(subjects):
        mask = subjects == subj
        result[str(subj)] = implausible_transition_rate(labels[mask], rules)
    return result


def sequences_by_subject(subjects: np.ndarray, labels: np.ndarray) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for subj, lbl in zip(subjects, labels):
        grouped.setdefault(str(subj), []).append(lbl)
    return grouped


def stage_change_rate(labels: Sequence[str], epoch_length_sec: int = 30) -> float:
    if len(labels) < 2:
        return 0.0
    changes = sum(1 for a, b in zip(labels[:-1], labels[1:]) if a != b)
    hours = (len(labels) * epoch_length_sec) / 3600.0
    if hours == 0:
        return 0.0
    return changes / hours
