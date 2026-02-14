"""Tests for benchmarks._data."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from benchmarks._data import (
    ensure_catalog,
    ensure_fits,
    fits_bounds,
    match_catalog_pixels,
    select_spectra,
)


def _make_fits(path: Path, ctype1: str = "GLON-CAR") -> fits.Header:
    """Write a minimal FITS cube and return its header."""
    nx, ny, nz = 4, 3, 10
    data = np.ones((nz, ny, nx), dtype=np.float64)
    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = nz
    hdr["CTYPE1"] = ctype1
    hdr["CTYPE2"] = "GLAT-CAR" if "GLON" in ctype1 else "DEC--TAN"
    hdr["CTYPE3"] = "VELO-LSR"
    hdr["CRPIX1"] = 1.0
    hdr["CRPIX2"] = 1.0
    hdr["CRPIX3"] = 1.0
    hdr["CDELT1"] = 0.01
    hdr["CDELT2"] = 0.01
    hdr["CDELT3"] = 1000.0
    hdr["CRVAL1"] = 30.0 if "GLON" in ctype1 else 180.0
    hdr["CRVAL2"] = 0.0
    hdr["CRVAL3"] = 0.0
    fits.PrimaryHDU(data=data, header=hdr).writeto(str(path), overwrite=True)
    return hdr


def test_ensure_fits_cached(tmp_path: Path) -> None:
    """ensure_fits should return header+data from an existing FITS."""
    fits_path = tmp_path / "test.fits"
    _make_fits(fits_path)
    header, data = ensure_fits(url="http://unused", path=str(fits_path))
    assert data.shape == (10, 3, 4)
    assert "NAXIS1" in header


def test_ensure_fits_download(tmp_path: Path) -> None:
    """ensure_fits should call urlretrieve when cache missing."""
    fits_path = tmp_path / "sub" / "download.fits"

    def fake_download(_url: str, dest: str) -> None:
        _make_fits(Path(dest))

    with patch("benchmarks._data.urlretrieve", side_effect=fake_download):
        _, data = ensure_fits(url="http://example.com/f.fits", path=str(fits_path))
    assert data.shape == (10, 3, 4)


def test_fits_bounds_glon(tmp_path: Path) -> None:
    """fits_bounds should return galactic bounds for GLON WCS."""
    fits_path = tmp_path / "glon.fits"
    hdr = _make_fits(fits_path, ctype1="GLON-CAR")
    glon_min, glon_max, glat_min, glat_max = fits_bounds(hdr)
    assert glon_min < glon_max
    assert glat_min < glat_max


def test_fits_bounds_icrs(tmp_path: Path) -> None:
    """fits_bounds should handle ICRS WCS via coordinate transform."""
    fits_path = tmp_path / "icrs.fits"
    hdr = _make_fits(fits_path, ctype1="RA---TAN")
    glon_min, glon_max, _, _ = fits_bounds(hdr)
    assert glon_min < glon_max


def test_ensure_catalog_cached(tmp_path: Path) -> None:
    """ensure_catalog should read from cached votable."""
    cat = Table({"GLON": [30.0], "GLAT": [0.0], "TB": [1.0]})
    cat_path = tmp_path / "catalog.votable"
    cat.write(str(cat_path), format="votable", overwrite=True)
    result = ensure_catalog(29.0, 31.0, -1.0, 1.0, path=str(cat_path))
    assert len(result) == 1


def test_ensure_catalog_download(tmp_path: Path) -> None:
    """ensure_catalog should download when not cached."""
    cat = Table({"GLON": [30.0], "GLAT": [0.0], "TB": [1.0]})
    buf = io.BytesIO()
    cat.write(buf, format="votable")
    votable_bytes = buf.getvalue()

    cat_path = tmp_path / "sub" / "catalog.votable"
    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = votable_bytes

    with patch("benchmarks._data.urlopen", return_value=mock_resp):
        result = ensure_catalog(29.0, 31.0, -1.0, 1.0, path=str(cat_path))
    assert len(result) == 1


def test_match_catalog_pixels(tmp_path: Path) -> None:
    """match_catalog_pixels should map catalog rows to pixel coords."""
    fits_path = tmp_path / "match.fits"
    hdr = _make_fits(fits_path, ctype1="GLON-CAR")
    wcs = WCS(hdr, naxis=2)
    # Pixel (0,0) -> world coords
    lon, lat = wcs.pixel_to_world_values(0, 0)
    cat = Table({"GLON": [float(lon)], "GLAT": [float(lat)]})
    counts = match_catalog_pixels(cat, hdr)
    assert (0, 0) in counts
    assert counts[(0, 0)] == 1


def test_match_catalog_pixels_icrs(tmp_path: Path) -> None:
    """match_catalog_pixels should handle RA/DEC headers via ICRS conversion."""
    fits_path = tmp_path / "icrs_match.fits"
    hdr = _make_fits(fits_path, ctype1="RA---TAN")
    wcs = WCS(hdr, naxis=2)
    ra, dec = wcs.pixel_to_world_values(0, 0)
    sky = SkyCoord(ra=float(ra), dec=float(dec), unit="deg", frame="icrs")
    cat = Table({"GLON": [sky.galactic.l.deg], "GLAT": [sky.galactic.b.deg]})
    counts = match_catalog_pixels(cat, hdr)
    assert (0, 0) in counts


def test_select_spectra(tmp_path: Path) -> None:
    """select_spectra should select spectra with catalog components."""
    fits_path = tmp_path / "sel.fits"
    hdr = _make_fits(fits_path, ctype1="GLON-CAR")
    nx, ny, nz = 4, 3, 10
    cube = np.ones((nz, ny, nx), dtype=np.float64)
    wcs = WCS(hdr, naxis=2)
    lon, lat = wcs.pixel_to_world_values(1, 1)
    cat = Table({"GLON": [float(lon)], "GLAT": [float(lat)]})
    selected, signals = select_spectra(cube, hdr, cat, n_spectra=5, seed=42)
    assert len(selected) == 1
    assert signals.shape == (1, nz)
