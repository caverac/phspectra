"""Tests for the _gaussfit C extension (bounded Levenberg-Marquardt)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

_gaussfit = pytest.importorskip("phspectra._gaussfit")
bounded_lm_fit = _gaussfit.bounded_lm_fit
find_peaks = _gaussfit.find_peaks


def _gaussian(x: npt.NDArray[np.float64], a: float, mu: float, sig: float) -> npt.NDArray[np.float64]:
    """Sum a single Gaussian component."""
    return a * np.exp(-0.5 * ((x - mu) / sig) ** 2)


class TestSingleGaussian:
    """Single-Gaussian parameter recovery."""

    def test_recovers_known_params(self) -> None:
        """Recover amplitude, mean, and stddev from a noiseless Gaussian."""
        x = np.arange(200, dtype=np.float64)
        y = _gaussian(x, 5.0, 100.0, 8.0)
        p0 = np.array([4.0, 95.0, 6.0], dtype=np.float64)
        lower = np.array([0.0, 0.0, 0.3], dtype=np.float64)
        upper = np.array([np.inf, 200.0, 100.0], dtype=np.float64)

        popt = bounded_lm_fit(x, y, p0, lower, upper, 10000)

        assert abs(popt[0] - 5.0) < 0.01
        assert abs(popt[1] - 100.0) < 0.01
        assert abs(popt[2] - 8.0) < 0.01


class TestTwoGaussians:
    """Two-Gaussian recovery."""

    def test_recovers_two_components(self) -> None:
        """Recover parameters of two well-separated Gaussians."""
        x = np.arange(300, dtype=np.float64)
        y = _gaussian(x, 5.0, 80.0, 6.0) + _gaussian(x, 3.0, 200.0, 10.0)
        p0 = np.array([4.0, 75.0, 5.0, 2.5, 195.0, 8.0], dtype=np.float64)
        lower = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.3], dtype=np.float64)
        upper = np.array([np.inf, 300.0, 150.0, np.inf, 300.0, 150.0], dtype=np.float64)

        popt = bounded_lm_fit(x, y, p0, lower, upper, 10000)

        # First component
        assert abs(popt[0] - 5.0) < 0.05
        assert abs(popt[1] - 80.0) < 0.1
        assert abs(popt[2] - 6.0) < 0.1
        # Second component
        assert abs(popt[3] - 3.0) < 0.05
        assert abs(popt[4] - 200.0) < 0.1
        assert abs(popt[5] - 10.0) < 0.1


class TestBoundsRespected:
    """Output parameters must respect bounds."""

    def test_output_within_bounds(self) -> None:
        """Fitted parameters stay within the specified bounds."""
        x = np.arange(100, dtype=np.float64)
        y = _gaussian(x, 5.0, 50.0, 5.0)
        p0 = np.array([3.0, 50.0, 3.0], dtype=np.float64)
        lower = np.array([1.0, 30.0, 2.0], dtype=np.float64)
        upper = np.array([10.0, 70.0, 20.0], dtype=np.float64)

        popt = bounded_lm_fit(x, y, p0, lower, upper, 10000)

        for j in range(3):
            assert popt[j] >= lower[j] - 1e-10
            assert popt[j] <= upper[j] + 1e-10


class TestMaxfevTermination:
    """Should raise RuntimeError when maxfev is too low to converge."""

    def test_raises_on_maxfev(self) -> None:
        """RuntimeError raised when max function evaluations exceeded."""
        x = np.arange(200, dtype=np.float64)
        y = _gaussian(x, 5.0, 100.0, 8.0)
        p0 = np.array([0.1, 50.0, 20.0], dtype=np.float64)
        lower = np.array([0.0, 0.0, 0.3], dtype=np.float64)
        upper = np.array([np.inf, 200.0, 100.0], dtype=np.float64)

        with pytest.raises(RuntimeError, match="maxfev"):
            bounded_lm_fit(x, y, p0, lower, upper, 2)


class TestNoisySignal:
    """Convergence on noisy signals."""

    def test_noisy_recovery(self) -> None:
        """Recover approximate parameters from a noisy Gaussian."""
        rng = np.random.default_rng(42)
        x = np.arange(200, dtype=np.float64)
        y = _gaussian(x, 5.0, 100.0, 8.0) + rng.normal(0, 0.3, size=200)
        p0 = np.array([4.0, 95.0, 6.0], dtype=np.float64)
        lower = np.array([0.0, 0.0, 0.3], dtype=np.float64)
        upper = np.array([np.inf, 200.0, 100.0], dtype=np.float64)

        popt = bounded_lm_fit(x, y, p0, lower, upper, 10000)

        assert abs(popt[0] - 5.0) < 0.5
        assert abs(popt[1] - 100.0) < 1.0
        assert abs(popt[2] - 8.0) < 1.0


class TestFindPeaksSinglePeak:
    """Single-peak recovery via C persistence detection."""

    def test_single_peak(self) -> None:
        """Single spike produces exactly one peak with correct attributes."""
        signal = np.zeros(100, dtype=np.float64)
        signal[50] = 5.0
        peaks = find_peaks(signal, 0.0)
        assert len(peaks) == 1
        idx, birth, death, persistence, saddle = peaks[0]
        assert idx == 50
        assert birth == 5.0
        assert death == 0.0
        assert persistence == 5.0
        assert saddle == -1


class TestFindPeaksTwoPeaks:
    """Two-peak ordering by persistence."""

    def test_ordered_by_persistence(self) -> None:
        """Two spikes are returned sorted by descending persistence."""
        signal = np.zeros(100, dtype=np.float64)
        signal[30] = 5.0
        signal[70] = 3.0
        peaks = find_peaks(signal, 0.0)
        assert len(peaks) == 2
        assert peaks[0][3] >= peaks[1][3]
        assert peaks[0][0] == 30
        assert peaks[1][0] == 70


class TestFindPeaksEmpty:
    """Empty signal returns empty list."""

    def test_empty(self) -> None:
        """Zero-length signal produces no peaks."""
        peaks = find_peaks(np.array([], dtype=np.float64), 0.0)
        assert peaks == []


class TestFindPeaksFiltering:
    """min_persistence filtering."""

    def test_filters_low_persistence(self) -> None:
        """Only peaks above min_persistence threshold are returned."""
        signal = np.zeros(100, dtype=np.float64)
        signal[30] = 5.0
        signal[70] = 3.0
        peaks = find_peaks(signal, 4.0)
        assert len(peaks) == 1
        assert peaks[0][0] == 30
