"""Smoke tests: run phspectra on real GRS spectra from the GaussPy+ test field.

Requires network access (first run) and astropy.  Run with:

    make test-python-smoke

or:

    cd packages/phspectra && uv run pytest -m slow
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pytest

from phspectra.decompose import fit_gaussians
from phspectra.persistence import find_peaks_by_persistence

FITS_URL = (
    "https://github.com/mriener/gausspyplus/raw/master/gausspyplus/data/grs-test_field.fits"
)

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
FITS_PATH = CACHE_DIR / "grs-test_field.fits"

# Spatial positions to test: (y, x) in the cube.
# The first entry is the one used in the GaussPy+ tutorial.
# "bright" positions are known to have real emission; "any" may be noise-only.
BRIGHT_POSITIONS = [
    (31, 40),
    (5, 5),
    (50, 18),
]
ALL_POSITIONS = BRIGHT_POSITIONS + [
    (10, 10),
    (20, 15),
]


@pytest.fixture(scope="session")
def fits_cube() -> np.ndarray:
    """Download the GRS test-field FITS cube (cached) and return the data array."""
    from astropy.io import fits  # type: ignore[import-untyped]

    if not FITS_PATH.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urlretrieve(FITS_URL, FITS_PATH)  # noqa: S310

    with fits.open(FITS_PATH) as hdul:
        data: np.ndarray = np.array(hdul[0].data, dtype=np.float64)
    return data


@pytest.mark.slow
class TestGRSSmoke:
    """Sanity checks on real GRS spectra."""

    def test_cube_shape(self, fits_cube: np.ndarray) -> None:
        """The cube should be 3-D with a reasonable number of channels."""
        assert fits_cube.ndim == 3
        n_channels = fits_cube.shape[0]
        assert 200 < n_channels < 1000, f"Unexpected channel count: {n_channels}"

    @pytest.mark.parametrize("y,x", ALL_POSITIONS)
    def test_find_peaks(self, fits_cube: np.ndarray, y: int, x: int) -> None:
        """Persistence peak finder should run without error on any position."""
        from phspectra.decompose import estimate_rms

        spectrum = fits_cube[:, y, x]
        spectrum = np.nan_to_num(spectrum, nan=0.0)

        rms = estimate_rms(spectrum)
        peaks = find_peaks_by_persistence(spectrum, min_persistence=5.0 * rms)

        for pk in peaks:
            assert pk.persistence > 0
            assert pk.birth > pk.death

    @pytest.mark.parametrize("y,x", BRIGHT_POSITIONS)
    def test_find_peaks_bright(self, fits_cube: np.ndarray, y: int, x: int) -> None:
        """Bright positions should have at least one peak above the noise."""
        from phspectra.decompose import estimate_rms

        spectrum = fits_cube[:, y, x]
        spectrum = np.nan_to_num(spectrum, nan=0.0)

        rms = estimate_rms(spectrum)
        peaks = find_peaks_by_persistence(spectrum, min_persistence=5.0 * rms)

        assert len(peaks) >= 1, f"No peaks found at (y={y}, x={x})"

    @pytest.mark.parametrize("y,x", BRIGHT_POSITIONS)
    def test_fit_gaussians(self, fits_cube: np.ndarray, y: int, x: int) -> None:
        """Gaussian fitting should produce components with positive amplitude and width."""
        spectrum = fits_cube[:, y, x]
        spectrum = np.nan_to_num(spectrum, nan=0.0)

        # Use default beta=5.0 (noise-aware threshold)
        components = fit_gaussians(spectrum)

        assert len(components) >= 1, f"No components fitted at (y={y}, x={x})"
        for comp in components:
            assert comp.amplitude > 0, f"Non-positive amplitude: {comp}"
            assert comp.stddev > 0, f"Non-positive stddev: {comp}"
