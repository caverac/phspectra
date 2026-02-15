"""Tests for benchmarks.commands.pre_compute (was compare)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from benchmarks.cli import main
from click.testing import CliRunner
from numpy.linalg import LinAlgError


def _make_header_and_cube() -> tuple[fits.Header, npt.NDArray[np.float64]]:
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


def _gp_results_for(n_spectra: int) -> dict[str, object]:
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
    output_dir = tmp_path / "output"

    with (
        patch("benchmarks.commands.pre_compute.ensure_fits", return_value=(header, cube)),
        patch("benchmarks.commands.pre_compute.build_image"),
        patch("benchmarks.commands.pre_compute.run_gausspyplus") as mock_gp,
    ):
        mock_gp.return_value = _gp_results_for(1)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["pre-compute", "--n-spectra", "1", "--output-dir", str(output_dir), "--seed", "42"],
        )
    assert result.exit_code == 0, result.output
    assert (output_dir / "spectra.npz").exists()
    assert (output_dir / "pre-compute.db").exists()


def test_compare_extra_pixels(tmp_path: Path) -> None:
    """CLI should handle --extra-pixels: empty pair, duplicate, OOB."""
    header, cube = _make_header_and_cube()
    output_dir = tmp_path / "output"

    with (
        patch("benchmarks.commands.pre_compute.ensure_fits", return_value=(header, cube)),
        patch("benchmarks.commands.pre_compute.build_image"),
        patch("benchmarks.commands.pre_compute.run_gausspyplus") as mock_gp,
    ):
        # 1 sampled + (2,2) in bounds + (99,99) OOB skipped + (0,0) in bounds
        # + second (2,2) duplicate skipped = 3 total
        mock_gp.return_value = _gp_results_for(3)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "pre-compute",
                "--n-spectra",
                "1",
                "--output-dir",
                str(output_dir),
                "--seed",
                "42",
                "--extra-pixels",
                "2,2;99,99;0,0;2,2;",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "already selected" in result.output


def test_compare_progress_line(tmp_path: Path) -> None:
    """CLI should print progress at every 100 spectra."""
    # Need >= 100 pixels to trigger the progress line at (i+1) % 100 == 0
    nx, ny, nz = 10, 10, 50
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
    n = nx * ny  # 100

    output_dir = tmp_path / "output"
    with (
        patch("benchmarks.commands.pre_compute.ensure_fits", return_value=(hdr, cube)),
        patch("benchmarks.commands.pre_compute.fit_gaussians", return_value=[]),
        patch("benchmarks.commands.pre_compute.build_image"),
        patch("benchmarks.commands.pre_compute.run_gausspyplus") as mock_gp,
    ):
        mock_gp.return_value = _gp_results_for(n)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["pre-compute", "--output-dir", str(output_dir)],
        )
    assert result.exit_code == 0, result.output
    # Progress line at the 100th spectrum
    assert "100/100" in result.output


def test_compare_all_pixels(tmp_path: Path) -> None:
    """CLI should use all pixels when --n-spectra is omitted."""
    header, cube = _make_header_and_cube()
    output_dir = tmp_path / "output"
    # Cube is 4x3=12 pixels
    n = 4 * 3

    with (
        patch("benchmarks.commands.pre_compute.ensure_fits", return_value=(header, cube)),
        patch("benchmarks.commands.pre_compute.build_image"),
        patch("benchmarks.commands.pre_compute.run_gausspyplus") as mock_gp,
    ):
        mock_gp.return_value = _gp_results_for(n)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["pre-compute", "--output-dir", str(output_dir)],
        )
    assert result.exit_code == 0, result.output
    assert f"{n}/{n}" in result.output


def test_compare_linalg_error(tmp_path: Path) -> None:
    """CLI should handle LinAlgError/ValueError from fit_gaussians."""
    header, cube = _make_header_and_cube()
    output_dir = tmp_path / "output"

    with (
        patch("benchmarks.commands.pre_compute.ensure_fits", return_value=(header, cube)),
        patch("benchmarks.commands.pre_compute.fit_gaussians", side_effect=LinAlgError("test")),
        patch("benchmarks.commands.pre_compute.build_image"),
        patch("benchmarks.commands.pre_compute.run_gausspyplus") as mock_gp,
    ):
        mock_gp.return_value = _gp_results_for(1)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["pre-compute", "--n-spectra", "1", "--output-dir", str(output_dir), "--seed", "42"],
        )
    assert result.exit_code == 0, result.output
