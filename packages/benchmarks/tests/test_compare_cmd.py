"""Tests for benchmarks.commands.compare."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from benchmarks.cli import main
from click.testing import CliRunner


def _make_header_and_cube() -> tuple[fits.Header, np.ndarray]:
    """Return a minimal header and cube for testing."""
    nx, ny, nz = 4, 3, 50
    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = nz
    hdr["CTYPE1"] = "GLON-CAR"
    hdr["CTYPE2"] = "GLAT-CAR"
    hdr["CRPIX1"] = 1.0
    hdr["CRPIX2"] = 1.0
    hdr["CDELT1"] = 0.01
    hdr["CDELT2"] = 0.01
    hdr["CRVAL1"] = 30.0
    hdr["CRVAL2"] = 0.0
    cube = np.random.default_rng(0).normal(0, 0.1, (nz, ny, nx))
    return hdr, cube


def _make_catalog(header: fits.Header) -> Table:
    """Create a catalog with entries matching pixel (1,1)."""
    wcs = WCS(header, naxis=2)
    lon, lat = wcs.pixel_to_world_values(1, 1)
    return Table({"GLON": [float(lon)], "GLAT": [float(lat)]})


def _gp_results_for(n_spectra: int) -> dict:
    """Build a fake GaussPy+ results dict."""
    return {
        "amplitudes_fit": [[1.0]] * n_spectra,
        "means_fit": [[25.0]] * n_spectra,
        "stddevs_fit": [[3.0]] * n_spectra,
        "times": [0.5] * n_spectra,
        "total_time_s": 0.5 * n_spectra,
        "mean_n_components": 1.0,
    }


def test_compare_cli(tmp_path: Path) -> None:
    """CLI should run end-to-end with all deps mocked."""
    header, cube = _make_header_and_cube()
    catalog = _make_catalog(header)
    output_dir = tmp_path / "output"

    with (
        patch("benchmarks.commands.compare.ensure_fits", return_value=(header, cube)),
        patch("benchmarks.commands.compare.fits_bounds", return_value=(29.0, 31.0, -1.0, 1.0)),
        patch("benchmarks.commands.compare.ensure_catalog", return_value=catalog),
        patch("benchmarks.commands.compare.build_image"),
        patch("benchmarks.commands.compare.run_gausspyplus") as mock_gp,
    ):
        mock_gp.return_value = _gp_results_for(1)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["compare", "--n-spectra", "1", "--output-dir", str(output_dir), "--seed", "42"],
        )
    assert result.exit_code == 0, result.output
    assert (output_dir / "spectra.npz").exists()
    assert (output_dir / "phspectra_results.json").exists()


def test_compare_extra_pixels(tmp_path: Path) -> None:
    """CLI should handle --extra-pixels: empty pair, duplicate, OOB."""
    header, cube = _make_header_and_cube()
    catalog = _make_catalog(header)
    output_dir = tmp_path / "output"

    with (
        patch("benchmarks.commands.compare.ensure_fits", return_value=(header, cube)),
        patch("benchmarks.commands.compare.fits_bounds", return_value=(29.0, 31.0, -1.0, 1.0)),
        patch("benchmarks.commands.compare.ensure_catalog", return_value=catalog),
        patch("benchmarks.commands.compare.build_image"),
        patch("benchmarks.commands.compare.run_gausspyplus") as mock_gp,
    ):
        # (2,2) in bounds; (99,99) OOB; (1,1) duplicate; trailing ; = empty pair
        mock_gp.return_value = _gp_results_for(2)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "compare", "--n-spectra", "1", "--output-dir", str(output_dir),
                "--seed", "42", "--extra-pixels", "2,2;99,99;1,1;",
            ],
        )
    assert result.exit_code == 0, result.output


def test_compare_empty_catalog(tmp_path: Path) -> None:
    """CLI should exit when catalog is empty."""
    header, cube = _make_header_and_cube()
    empty_cat = Table({"GLON": np.array([], dtype=float), "GLAT": np.array([], dtype=float)})
    output_dir = tmp_path / "output"

    with (
        patch("benchmarks.commands.compare.ensure_fits", return_value=(header, cube)),
        patch("benchmarks.commands.compare.fits_bounds", return_value=(29.0, 31.0, -1.0, 1.0)),
        patch("benchmarks.commands.compare.ensure_catalog", return_value=empty_cat),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["compare", "--n-spectra", "1", "--output-dir", str(output_dir)],
        )
    assert result.exit_code != 0


def test_compare_progress_line(tmp_path: Path) -> None:
    """CLI should print progress at every 100 spectra."""
    header, cube = _make_header_and_cube()
    catalog = _make_catalog(header)
    output_dir = tmp_path / "output"

    # Mock select_spectra to return 100 spectra
    pixels_100 = [(i % 4, i % 3) for i in range(100)]
    signals_100 = np.random.default_rng(0).normal(0, 0.1, (100, 50))

    with (
        patch("benchmarks.commands.compare.ensure_fits", return_value=(header, cube)),
        patch("benchmarks.commands.compare.fits_bounds", return_value=(29.0, 31.0, -1.0, 1.0)),
        patch("benchmarks.commands.compare.ensure_catalog", return_value=catalog),
        patch("benchmarks.commands.compare.select_spectra", return_value=(pixels_100, signals_100)),
        patch("benchmarks.commands.compare.fit_gaussians", return_value=[]),
        patch("benchmarks.commands.compare.build_image"),
        patch("benchmarks.commands.compare.run_gausspyplus") as mock_gp,
    ):
        mock_gp.return_value = _gp_results_for(100)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["compare", "--n-spectra", "100", "--output-dir", str(output_dir), "--seed", "42"],
        )
    assert result.exit_code == 0, result.output
    assert "100/100" in result.output


def test_compare_linalg_error(tmp_path: Path) -> None:
    """CLI should handle LinAlgError/ValueError from fit_gaussians."""
    from numpy.linalg import LinAlgError

    header, cube = _make_header_and_cube()
    catalog = _make_catalog(header)
    output_dir = tmp_path / "output"

    with (
        patch("benchmarks.commands.compare.ensure_fits", return_value=(header, cube)),
        patch("benchmarks.commands.compare.fits_bounds", return_value=(29.0, 31.0, -1.0, 1.0)),
        patch("benchmarks.commands.compare.ensure_catalog", return_value=catalog),
        patch("benchmarks.commands.compare.fit_gaussians", side_effect=LinAlgError("test")),
        patch("benchmarks.commands.compare.build_image"),
        patch("benchmarks.commands.compare.run_gausspyplus") as mock_gp,
    ):
        mock_gp.return_value = _gp_results_for(1)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["compare", "--n-spectra", "1", "--output-dir", str(output_dir), "--seed", "42"],
        )
    assert result.exit_code == 0, result.output
