"""Tests for benchmarks._matching."""

from __future__ import annotations

from benchmarks._matching import count_correct_matches, f1_score, match_pairs
from benchmarks._types import Component


def test_match_pairs_basic() -> None:
    """Matching should pair nearby components."""
    ref = [Component(5.0, 100.0, 5.0), Component(3.0, 200.0, 4.0)]
    test = [Component(4.8, 101.0, 4.8), Component(2.9, 201.0, 4.2)]
    pairs = match_pairs(ref, test)
    assert len(pairs) == 2
    # Pairs should be (ref[0], test[0]) and (ref[1], test[1])
    assert pairs[0][0].mean == 100.0
    assert pairs[0][1].mean == 101.0


def test_match_pairs_empty_ref() -> None:
    """Empty reference should return no pairs."""
    pairs = match_pairs([], [Component(5.0, 100.0, 5.0)])
    assert not pairs


def test_match_pairs_empty_test() -> None:
    """Empty test set should return no pairs."""
    pairs = match_pairs([Component(5.0, 100.0, 5.0)], [])
    assert not pairs


def test_match_pairs_too_far() -> None:
    """Components too far apart should not be matched."""
    ref = [Component(5.0, 100.0, 2.0)]
    test = [Component(5.0, 200.0, 2.0)]
    pairs = match_pairs(ref, test, pos_tol_sigma=2.0)
    assert not pairs


def test_count_correct_matches_perfect() -> None:
    """Perfect match should count all correct."""
    true = [Component(5.0, 100.0, 5.0)]
    guessed = [Component(5.0, 100.0, 5.0)]
    n = count_correct_matches(true, guessed)
    assert n == 1


def test_count_correct_matches_empty() -> None:
    """No components should return 0."""
    assert count_correct_matches([], []) == 0
    assert count_correct_matches([Component(5.0, 100.0, 5.0)], []) == 0
    assert count_correct_matches([], [Component(5.0, 100.0, 5.0)]) == 0


def test_count_correct_matches_wrong_width() -> None:
    """Width outside bounds should not count as correct."""
    true = [Component(5.0, 100.0, 5.0)]
    guessed = [Component(5.0, 100.0, 50.0)]  # width ratio = 10 > 2.5
    n = count_correct_matches(true, guessed)
    assert n == 0


def test_f1_score_perfect() -> None:
    """Perfect detection should give F1 = 1.0."""
    p, r, f1 = f1_score(3, 3, 3)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_f1_score_no_detections() -> None:
    """No detections should give F1 = 0."""
    p, r, f1 = f1_score(0, 3, 0)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_f1_score_no_truth() -> None:
    """No ground truth but no detections should be perfect."""
    p, r, f1 = f1_score(0, 0, 0)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_f1_score_partial() -> None:
    """Partial detection: 2 correct out of 3 true, 4 guessed."""
    p, r, f1 = f1_score(2, 3, 4)
    assert abs(p - 0.5) < 1e-10
    assert abs(r - 2 / 3) < 1e-10
    expected_f1 = 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3)
    assert abs(f1 - expected_f1) < 1e-10
