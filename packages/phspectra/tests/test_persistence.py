"""Smoke tests for persistent homology peak detection."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from phspectra.decompose import fit_gaussians
from phspectra.persistence import find_peaks_by_persistence


def _make_gaussian(x: npt.NDArray[np.float64], amp: float, mu: float, sig: float) -> npt.NDArray[np.float64]:
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def test_single_peak() -> None:
    """A single Gaussian should produce exactly one persistent peak."""
    x = np.linspace(0, 100, 500)
    signal = _make_gaussian(x, amp=5.0, mu=50.0, sig=3.0)
    peaks = find_peaks_by_persistence(signal)
    assert len(peaks) == 1
    assert peaks[0].persistence > 4.0


def test_two_peaks() -> None:
    """Two well-separated Gaussians should produce two significant peaks."""
    x = np.linspace(0, 100, 500)
    signal = _make_gaussian(x, 5.0, 30.0, 3.0) + _make_gaussian(x, 3.0, 70.0, 4.0)
    peaks = find_peaks_by_persistence(signal, min_persistence=1.0)
    assert len(peaks) == 2


def test_empty_signal() -> None:
    """Empty signal should return no peaks."""
    peaks = find_peaks_by_persistence(np.array([]))
    assert not peaks


def test_plateau_same_component() -> None:
    """Consecutive equal values should not create spurious peaks.

    When a neighbor is already in the same component as the current index,
    the union-find ``idx_root == neighbor_root`` branch is taken (line 113).
    """
    # A flat plateau followed by a single peak
    signal = np.concatenate([np.full(20, 1.0), np.full(5, 5.0), np.full(20, 1.0)])
    peaks = find_peaks_by_persistence(signal, min_persistence=0.5)
    # Should find exactly 1 peak (the plateau at 5.0)
    assert len(peaks) == 1


def test_fit_gaussians_recovers_params() -> None:
    """fit_gaussians should approximately recover ground-truth parameters."""
    x = np.linspace(0, 100, 500)
    signal = _make_gaussian(x, 5.0, 50.0, 5.0)
    components = fit_gaussians(signal)
    assert len(components) == 1
    c = components[0]
    # Amplitude and mean should be close
    assert abs(c.amplitude - 5.0) < 0.5
    peak_idx = int(50.0 / 100.0 * 500)
    assert abs(c.mean - peak_idx) < 5
