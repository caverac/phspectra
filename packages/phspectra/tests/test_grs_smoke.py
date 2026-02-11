"""Smoke tests: run phspectra on real GRS spectra from the GaussPy+ test field.

Requires network access (first run) and astropy.  Run with:

    make test-python-smoke

or:

    cd packages/phspectra && uv run pytest -m slow
"""

from __future__ import annotations

import numpy as np
import pytest

from phspectra.decompose import estimate_rms, fit_gaussians
from phspectra.persistence import find_peaks_by_persistence

# Spatial positions to test: (y, x) in the cube.
# The first entry is the one used in the GaussPy+ tutorial.
# "bright" have clear emission (SNR>5); "marginal" have weak/narrow features;
# "any" may be noise-only.
BRIGHT_POSITIONS = [
    (31, 40),
]
MARGINAL_POSITIONS = [
    (5, 5),
    (50, 18),
]
ALL_POSITIONS = (
    BRIGHT_POSITIONS
    + MARGINAL_POSITIONS
    + [
        (10, 10),
        (20, 15),
    ]
)


@pytest.mark.slow
class TestGRSSmoke:
    """Sanity checks on real GRS spectra."""

    def test_cube_shape(self, grs_cube: np.ndarray) -> None:
        """The cube should be 3-D with a reasonable number of channels."""
        assert grs_cube.ndim == 3
        n_channels = grs_cube.shape[0]
        assert 200 < n_channels < 1000, f"Unexpected channel count: {n_channels}"

    @pytest.mark.parametrize("y,x", ALL_POSITIONS)
    def test_find_peaks(self, grs_cube: np.ndarray, y: int, x: int) -> None:
        """Persistence peak finder should run without error on any position."""
        spectrum = grs_cube[:, y, x]
        spectrum = np.nan_to_num(spectrum, nan=0.0)

        rms = estimate_rms(spectrum)
        peaks = find_peaks_by_persistence(spectrum, min_persistence=5.0 * rms)

        for pk in peaks:
            assert pk.persistence > 0
            assert pk.birth > pk.death

    @pytest.mark.parametrize("y,x", BRIGHT_POSITIONS)
    def test_find_peaks_bright(self, grs_cube: np.ndarray, y: int, x: int) -> None:
        """Bright positions should have at least one peak above the noise."""
        spectrum = grs_cube[:, y, x]
        spectrum = np.nan_to_num(spectrum, nan=0.0)

        rms = estimate_rms(spectrum)
        peaks = find_peaks_by_persistence(spectrum, min_persistence=5.0 * rms)

        assert len(peaks) >= 1, f"No peaks found at (y={y}, x={x})"

    @pytest.mark.parametrize("y,x", BRIGHT_POSITIONS)
    def test_fit_gaussians(self, grs_cube: np.ndarray, y: int, x: int) -> None:
        """Gaussian fitting should produce components with positive amplitude and width."""
        spectrum = grs_cube[:, y, x]
        spectrum = np.nan_to_num(spectrum, nan=0.0)

        components = fit_gaussians(spectrum)

        assert len(components) >= 1, f"No components fitted at (y={y}, x={x})"
        for comp in components:
            assert comp.amplitude > 0, f"Non-positive amplitude: {comp}"
            assert comp.stddev > 0, f"Non-positive stddev: {comp}"

    @pytest.mark.parametrize("y,x", MARGINAL_POSITIONS)
    def test_fit_gaussians_marginal(self, grs_cube: np.ndarray, y: int, x: int) -> None:
        """Marginal positions may return 0 components after quality validation."""
        spectrum = grs_cube[:, y, x]
        spectrum = np.nan_to_num(spectrum, nan=0.0)

        components = fit_gaussians(spectrum)

        # Marginal detections may be filtered by quality control
        for comp in components:
            assert comp.amplitude > 0, f"Non-positive amplitude: {comp}"
            assert comp.stddev > 0, f"Non-positive stddev: {comp}"
