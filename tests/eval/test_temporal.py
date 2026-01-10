from __future__ import annotations

import numpy as np

from sleep_stager.eval.temporal import (
    TransitionRule,
    count_transitions,
    estimate_transition_matrix,
    implausible_transition_rate,
    stage_change_rate,
)


def test_transition_matrix_rows_normalized():
    labels = ["W", "N1", "N2", "N1", "N2", "N3", "REM", "W"]
    states = ["W", "N1", "N2", "N3", "REM"]
    mat = estimate_transition_matrix(labels, states, prior_weight=0.1, min_prob=1e-6)
    row_sums = mat.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums))


def test_implausible_transition_rate_counts_rules():
    labels = ["N3", "REM", "N2", "N3", "W", "W"]
    rules = [TransitionRule("N3", "REM"), TransitionRule("W", "N3")]
    rate = implausible_transition_rate(labels, rules)
    # transitions: N3->REM (implausible), REM->N2 (ok), N2->N3 (ok), N3->W (ok), W->W (ok)
    assert rate == 1 / 5


def test_stage_change_rate():
    labels = ["W", "W", "N1", "N1", "REM", "REM"]
    rate = stage_change_rate(labels, epoch_length_sec=30)
    # changes: W->N1, N1->REM => 2 changes over 3 minutes
    assert rate == 2 / (3 / 60)
