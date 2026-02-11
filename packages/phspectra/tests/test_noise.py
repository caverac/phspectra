"""Tests for phspectra.noise."""

from __future__ import annotations

import numpy as np
import pytest

from phspectra.noise import estimate_rms, estimate_rms_simple


def _make_gaussian(x: np.ndarray, amp: float, mu: float, sig: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def test_simple_matches_original() -> None:
    """estimate_rms_simple should match the original MAD / 0.6745 formula."""
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 0.5, size=424)
    result = estimate_rms_simple(signal)
    # Manual MAD computation
    med = np.median(signal)
    expected = float(np.median(np.abs(signal - med))) / 0.6745
    assert result == pytest.approx(expected, rel=1e-10)


def test_improved_ignores_signal_peak() -> None:
    """Improved RMS should be closer to true sigma when a strong signal is present."""
    rng = np.random.default_rng(42)
    true_sigma = 0.3
    noise = rng.normal(0, true_sigma, size=424)
    x = np.arange(424, dtype=np.float64)
    # Use a very strong, broad peak to clearly bias the simple estimator
    signal = noise + _make_gaussian(x, 10.0, 200.0, 30.0)

    rms_improved = estimate_rms(signal)
    rms_simple = estimate_rms_simple(signal)

    # Simple should be biased upward significantly; improved should be closer
    assert rms_simple > true_sigma * 1.2  # simple is biased high
    assert abs(rms_improved - true_sigma) < abs(rms_simple - true_sigma)


def test_pure_noise_both_agree() -> None:
    """On pure Gaussian noise both methods should agree within 15%."""
    rng = np.random.default_rng(123)
    signal = rng.normal(0, 1.0, size=1000)

    rms_improved = estimate_rms(signal)
    rms_simple = estimate_rms_simple(signal)

    assert rms_improved == pytest.approx(rms_simple, rel=0.15)


def test_empty_returns_zero() -> None:
    """Empty signal should return 0.0 for both estimators."""
    assert estimate_rms(np.array([])) == 0.0
    assert estimate_rms_simple(np.array([])) == 0.0


def test_all_positive_fallback() -> None:
    """All-positive signal should not crash and should return a reasonable value."""
    signal = np.ones(100) + np.random.default_rng(0).uniform(0, 0.1, size=100)
    result = estimate_rms(signal)
    assert result > 0
    assert np.isfinite(result)
