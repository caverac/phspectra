"""Shared fixtures for phspectra tests."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pytest
from astropy.io import fits  # type: ignore[import-untyped]

FITS_URL = "https://github.com/mriener/gausspyplus/raw/master/gausspyplus/data/grs-test_field.fits"

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
FITS_PATH = CACHE_DIR / "grs-test_field.fits"


@pytest.fixture(scope="session")
def grs_cube() -> np.ndarray:
    """Download the GRS test-field FITS cube (cached) and return the data array."""
    if not FITS_PATH.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urlretrieve(FITS_URL, FITS_PATH)  # noqa: S310

    with fits.open(FITS_PATH) as hdul:
        data: np.ndarray = np.array(hdul[0].data, dtype=np.float64)  # pylint: disable=no-member
    return data
