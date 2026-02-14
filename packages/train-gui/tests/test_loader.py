"""Tests for train_gui._loader."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from train_gui._loader import ComparisonData, PixelData, _find_fits, load_comparison_data


def test_load_with_npz(comparison_dir: Path) -> None:
    """Load from spectra.npz when it exists."""
    data = load_comparison_data(comparison_dir)
    assert isinstance(data, ComparisonData)
    assert len(data) == 2
    assert data.pixels == [(10, 20), (30, 40)]


def test_pixel_data_structure(comparison_dir: Path) -> None:
    """PixelData carries correct signal and components."""
    data = load_comparison_data(comparison_dir)
    pd = data.pixel_data[0]
    assert isinstance(pd, PixelData)
    assert pd.pixel == (10, 20)
    assert len(pd.signal) == 50
    assert len(pd.ph_components) == 1
    assert pd.ph_components[0]["source"] == "phspectra"
    assert len(pd.gp_components) == 1
    assert pd.gp_components[0]["source"] == "gausspyplus"


def test_load_falls_back_to_fits(comparison_dir: Path) -> None:
    """When spectra.npz is missing, fall back to FITS cube."""
    (comparison_dir / "spectra.npz").unlink()
    # Create a minimal FITS cube.
    n_channels, ny, nx = 50, 50, 50
    cube = np.zeros((n_channels, ny, nx), dtype=np.float64)
    cube[:, 20, 10] = 1.0  # pixel (10, 20)
    cube[:, 40, 30] = 2.0  # pixel (30, 40)
    from astropy.io import fits

    hdu = fits.PrimaryHDU(data=cube)
    hdu.writeto(comparison_dir / "grs-test-field.fits")

    data = load_comparison_data(comparison_dir)
    assert len(data) == 2
    np.testing.assert_array_equal(data.pixel_data[0].signal, 1.0)
    np.testing.assert_array_equal(data.pixel_data[1].signal, 2.0)


def test_find_fits_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FileNotFoundError when no FITS cube candidate exists."""
    # Ensure the hardcoded /tmp/phspectra path doesn't interfere.
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(FileNotFoundError, match="Cannot find FITS cube"):
        _find_fits(tmp_path / "nonexistent")


def test_comparison_data_len() -> None:
    """ComparisonData __len__ returns pixel_data count."""
    cd = ComparisonData(pixels=[], pixel_data=[])
    assert len(cd) == 0
