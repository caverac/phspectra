"""Tests for phspectra.noise."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from phspectra.noise import estimate_rms, estimate_rms_simple


def _make_gaussian(x: npt.NDArray[np.float64], amp: float, mu: float, sig: float) -> npt.NDArray[np.float64]:
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


def test_mad_sigma_zero_fallback() -> None:
    """When MAD of negative channels is zero, should fall back to simple estimator."""
    # All negative channels have the same value -> MAD = 0 -> mad_sigma = 0
    # Positive channels in short runs (<=2) won't be masked, but negative channels
    # that are all identical give MAD=0
    signal = np.zeros(100)
    # Make some short positive runs (length <= 2, so not masked)
    signal[10] = 0.1
    signal[20] = 0.1
    # All other channels are 0 (not positive), so neg_unmasked is all zeros -> MAD = 0
    result = estimate_rms(signal)
    expected = estimate_rms_simple(signal)
    assert result == pytest.approx(expected)


def test_few_surviving_after_clip_fallback() -> None:
    """When clipping leaves < 5 channels, should fall back to simple estimator.

    The signal has a long positive run (masked in step 1) flanked by negative
    channels whose spread is tiny (tight MAD → small ``mad_sigma``), but whose
    absolute value is large (``|val| >> clip_thresh``).  After step 3 clips
    every surviving channel, ``len(surviving) < 5`` triggers the fallback.
    """
    # Long positive run in the center → masked in step 1 (length > 2)
    signal = np.concatenate(
        [
            np.array([-0.50, -0.51, -0.49, -0.50, -0.52]),  # 5 neg channels (left)
            np.full(10, 10.0),  # positive run
            np.array([-0.50, -0.51, -0.49, -0.50, -0.52]),  # 5 neg channels (right)
        ]
    )
    # Step 1: positive run at 5-14 (length 10 > 2), mask with pad 2 → mask[3:17].
    # Unmasked: channels 0,1,2 and 17,18,19 (6 channels, all ~ -0.50).
    # Step 2: med_neg ≈ -0.50, MAD ≈ 0.01, mad_sigma ≈ 0.015.
    # Step 3: clip_thresh = 5 * 0.015 ≈ 0.074.  All |val| ≈ 0.50 > 0.074 → clipped.
    # Surviving: 0 channels < 5 → falls back to estimate_rms_simple.
    result = estimate_rms(signal)
    expected = estimate_rms_simple(signal)
    assert result == pytest.approx(expected)
