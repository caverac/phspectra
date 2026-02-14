"""Tests for benchmarks.commands.inspect_pixel."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from benchmarks.cli import main
from click.testing import CliRunner


def _setup_inspect_dir(tmp_path: Path) -> Path:
    """Set up a data dir with fake FITS and JSON results."""
    data_dir = tmp_path / "compare-docker"
    data_dir.mkdir()

    # Create fake FITS cube at cache root
    nx, ny, nz = 4, 3, 50
    data = np.ones((nz, ny, nx), dtype=np.float64)
    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = nz
    fits.PrimaryHDU(data=data, header=hdr).writeto(
        str(tmp_path / "grs-test-field.fits"), overwrite=True
    )

    # phspectra results
    ph = {
        "pixels": [[1, 1], [2, 2]],
        "amplitudes_fit": [[1.0], [2.0]],
        "means_fit": [[25.0], [25.0]],
        "stddevs_fit": [[3.0], [4.0]],
    }
    (data_dir / "phspectra_results.json").write_text(json.dumps(ph))

    # GP+ results
    gp = {
        "amplitudes_fit": [[1.1], [1.9]],
        "means_fit": [[25.5], [24.5]],
        "stddevs_fit": [[3.1], [4.1]],
    }
    (data_dir / "results.json").write_text(json.dumps(gp))

    return data_dir


def test_inspect_cli(tmp_path: Path) -> None:
    """CLI should run for a valid pixel."""
    data_dir = _setup_inspect_dir(tmp_path)

    def fake_ensure_fits(url: str = "", path: str = "") -> tuple:
        with fits.open(str(tmp_path / "grs-test-field.fits")) as hdul:
            hdr = hdul[0].header.copy()
            dat = np.asarray(hdul[0].data, dtype=np.float64)
        return hdr, dat

    with (
        patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake_ensure_fits),
        patch("benchmarks.commands.inspect_pixel.plt.show"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["inspect", "1", "1", "--data-dir", str(data_dir), "--betas", "3.8", "--mf-snr-mins", "4.0"],
        )
    assert result.exit_code == 0, result.output


def test_inspect_missing_results_json(tmp_path: Path) -> None:
    """CLI should fail when results.json is missing."""
    data_dir = tmp_path / "compare-docker"
    data_dir.mkdir()
    (data_dir / "phspectra_results.json").write_text("{}")

    def fake_ensure_fits(url: str = "", path: str = "") -> tuple:
        return fits.Header(), np.zeros((10, 3, 4))

    with patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake_ensure_fits):
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", "0", "0", "--data-dir", str(data_dir)])
    assert result.exit_code != 0


def test_inspect_missing_phspectra_results(tmp_path: Path) -> None:
    """CLI should fail when phspectra_results.json is missing."""
    data_dir = tmp_path / "compare-docker"
    data_dir.mkdir()
    (data_dir / "results.json").write_text("{}")

    def fake_ensure_fits(url: str = "", path: str = "") -> tuple:
        return fits.Header(), np.zeros((10, 3, 4))

    with patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake_ensure_fits):
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", "0", "0", "--data-dir", str(data_dir)])
    assert result.exit_code != 0


def test_inspect_pixel_not_found_no_nearby(tmp_path: Path) -> None:
    """CLI should fail when pixel is not in selection and none are nearby."""
    data_dir = _setup_inspect_dir(tmp_path)

    def fake_ensure_fits(url: str = "", path: str = "") -> tuple:
        with fits.open(str(tmp_path / "grs-test-field.fits")) as hdul:
            hdr = hdul[0].header.copy()
            dat = np.asarray(hdul[0].data, dtype=np.float64)
        return hdr, dat

    with patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake_ensure_fits):
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", "99", "99", "--data-dir", str(data_dir)])
    assert result.exit_code != 0


def test_inspect_pixel_not_found_nearby(tmp_path: Path) -> None:
    """CLI should suggest nearby pixels when the requested pixel is close."""
    data_dir = _setup_inspect_dir(tmp_path)

    def fake_ensure_fits(url: str = "", path: str = "") -> tuple:
        with fits.open(str(tmp_path / "grs-test-field.fits")) as hdul:
            hdr = hdul[0].header.copy()
            dat = np.asarray(hdul[0].data, dtype=np.float64)
        return hdr, dat

    # Pixel (0,0) is within 5 of (1,1) and (2,2)
    with patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake_ensure_fits):
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", "0", "0", "--data-dir", str(data_dir)])
    assert result.exit_code != 0


def test_inspect_single_beta_single_mf(tmp_path: Path) -> None:
    """CLI 1x1 axes reshape branch: single beta, single mf_snr_min."""
    data_dir = _setup_inspect_dir(tmp_path)

    def fake_ensure_fits(url: str = "", path: str = "") -> tuple:
        with fits.open(str(tmp_path / "grs-test-field.fits")) as hdul:
            hdr = hdul[0].header.copy()
            dat = np.asarray(hdul[0].data, dtype=np.float64)
        return hdr, dat

    with (
        patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake_ensure_fits),
        patch("benchmarks.commands.inspect_pixel.plt.show"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["inspect", "1", "1", "--data-dir", str(data_dir),
             "--betas", "3.8", "--mf-snr-mins", "4.0"],
        )
    assert result.exit_code == 0, result.output


def test_inspect_multi_beta_single_mf(tmp_path: Path) -> None:
    """CLI Nx1 axes reshape branch: multiple betas, single mf_snr_min."""
    data_dir = _setup_inspect_dir(tmp_path)

    def fake_ensure_fits(url: str = "", path: str = "") -> tuple:
        with fits.open(str(tmp_path / "grs-test-field.fits")) as hdul:
            hdr = hdul[0].header.copy()
            dat = np.asarray(hdul[0].data, dtype=np.float64)
        return hdr, dat

    with (
        patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake_ensure_fits),
        patch("benchmarks.commands.inspect_pixel.plt.show"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["inspect", "1", "1", "--data-dir", str(data_dir),
             "--betas", "3.8,4.0", "--mf-snr-mins", "4.0"],
        )
    assert result.exit_code == 0, result.output


def test_inspect_single_beta_multi_mf(tmp_path: Path) -> None:
    """CLI 1xN axes reshape branch: single beta, multiple mf_snr_mins."""
    data_dir = _setup_inspect_dir(tmp_path)

    def fake_ensure_fits(url: str = "", path: str = "") -> tuple:
        with fits.open(str(tmp_path / "grs-test-field.fits")) as hdul:
            hdr = hdul[0].header.copy()
            dat = np.asarray(hdul[0].data, dtype=np.float64)
        return hdr, dat

    with (
        patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake_ensure_fits),
        patch("benchmarks.commands.inspect_pixel.plt.show"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["inspect", "1", "1", "--data-dir", str(data_dir),
             "--betas", "3.8", "--mf-snr-mins", "4.0,5.0"],
        )
    assert result.exit_code == 0, result.output


def test_inspect_no_components(tmp_path: Path) -> None:
    """CLI should handle zero components (full zoom range)."""
    data_dir = tmp_path / "compare-docker"
    data_dir.mkdir()

    nx, ny, nz = 4, 3, 50
    data = np.zeros((nz, ny, nx), dtype=np.float64)
    hdr = fits.Header()
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = nz
    fits.PrimaryHDU(data=data, header=hdr).writeto(
        str(tmp_path / "grs-test-field.fits"), overwrite=True
    )
    ph = {"pixels": [[1, 1]], "amplitudes_fit": [[]], "means_fit": [[]], "stddevs_fit": [[]]}
    (data_dir / "phspectra_results.json").write_text(json.dumps(ph))
    gp = {"amplitudes_fit": [[]], "means_fit": [[]], "stddevs_fit": [[]]}
    (data_dir / "results.json").write_text(json.dumps(gp))

    def fake_ensure_fits(url: str = "", path: str = "") -> tuple:
        with fits.open(str(tmp_path / "grs-test-field.fits")) as hdul:
            h = hdul[0].header.copy()
            d = np.asarray(hdul[0].data, dtype=np.float64)
        return h, d

    with (
        patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake_ensure_fits),
        patch("benchmarks.commands.inspect_pixel.plt.show"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["inspect", "1", "1", "--data-dir", str(data_dir),
             "--betas", "3.8", "--mf-snr-mins", "4.0"],
        )
    assert result.exit_code == 0, result.output
