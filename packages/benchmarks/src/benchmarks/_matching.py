"""Hungarian matching, precision, recall, and F1 scoring."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from benchmarks._constants import (
    AMP_RATIO_BOUNDS,
    POS_TOLERANCE_SIGMA,
    WIDTH_RATIO_BOUNDS,
)
from benchmarks._types import Component
from scipy.optimize import linear_sum_assignment


def match_pairs(
    ref_comps: Sequence[Component],
    test_comps: Sequence[Component],
    pos_tol_sigma: float = 2.0,
) -> list[tuple[Component, Component]]:
    """Hungarian-match by position, return valid pairs.

    Parameters
    ----------
    ref_comps : Sequence[Component]
        Reference (e.g. GaussPy+) components.
    test_comps : Sequence[Component]
        Test (e.g. phspectra) components.
    pos_tol_sigma : float
        Maximum position offset in units of the reference stddev.

    Returns
    -------
    list[tuple[Component, Component]]
        Matched (ref, test) pairs.
    """
    if not ref_comps or not test_comps:
        return []
    n_ref, n_test = len(ref_comps), len(test_comps)
    cost = np.full((n_ref, n_test), 1e9)
    for i, rc in enumerate(ref_comps):
        for j, tc in enumerate(test_comps):
            cost[i, j] = abs(tc.mean - rc.mean)
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs: list[tuple[Component, Component]] = []
    for i, j in zip(row_ind, col_ind):
        rc, tc = ref_comps[i], test_comps[j]
        if abs(tc.mean - rc.mean) < pos_tol_sigma * rc.stddev:
            pairs.append((rc, tc))
    return pairs


def count_correct_matches(
    true_comps: Sequence[Component],
    guessed: Sequence[Component],
) -> int:
    """Count correctly matched components using Lindner et al. criteria.

    Parameters
    ----------
    true_comps : Sequence[Component]
        Ground-truth / reference components.
    guessed : Sequence[Component]
        Detected components.

    Returns
    -------
    int
        Number of correct matches.
    """
    if not true_comps or not guessed:
        return 0
    n_true, n_guess = len(true_comps), len(guessed)
    cost = np.full((n_true, n_guess), 1e9)
    for i, tc in enumerate(true_comps):
        for j, gc in enumerate(guessed):
            cost[i, j] = abs(gc.mean - tc.mean)
    row_ind, col_ind = linear_sum_assignment(cost)
    n_correct = 0
    for i, j in zip(row_ind, col_ind):
        tc, gc = true_comps[i], guessed[j]
        amp_ratio = gc.amplitude / tc.amplitude if tc.amplitude > 0 else 0.0
        pos_off = abs(gc.mean - tc.mean)
        w_ratio = gc.stddev / tc.stddev if tc.stddev > 0 else 0.0
        if (
            AMP_RATIO_BOUNDS[0] < amp_ratio < AMP_RATIO_BOUNDS[1]
            and pos_off < POS_TOLERANCE_SIGMA * tc.stddev
            and WIDTH_RATIO_BOUNDS[0] < w_ratio < WIDTH_RATIO_BOUNDS[1]
        ):
            n_correct += 1
    return n_correct


def f1_score(
    n_correct: int,
    n_true: int,
    n_guessed: int,
) -> tuple[float, float, float]:
    """Return (precision, recall, F1).

    Parameters
    ----------
    n_correct : int
        Number of correctly matched components.
    n_true : int
        Total number of ground-truth components.
    n_guessed : int
        Total number of detected components.

    Returns
    -------
    tuple[float, float, float]
        (precision, recall, F1).
    """
    precision = n_correct / n_guessed if n_guessed > 0 else (1.0 if n_true == 0 else 0.0)
    recall = n_correct / n_true if n_true > 0 else 1.0
    denom = precision + recall
    if denom == 0:
        return 0.0, 0.0, 0.0
    return precision, recall, 2.0 * precision * recall / denom
