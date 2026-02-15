"""Tests for benchmarks.commands.inspect_pixel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from benchmarks._database import (
    create_db,
    insert_components,
    insert_gausspyplus_run,
    insert_phspectra_run,
    insert_pixels,
)
from benchmarks.cli import main
from click.testing import CliRunner


def _make_fake_ensure_fits(fits_path: Path) -> Callable[..., Any]:
    """Create a fake ensure_fits that loads from a test FITS file."""

    def _inner(**_kwargs: str) -> tuple[fits.Header, npt.NDArray[np.float64]]:
        header = fits.getheader(str(fits_path))
        data = np.asarray(fits.getdata(str(fits_path)), dtype=np.float64)
        return header, data

    return _inner


def _fake_ensure_fits_empty(**_kwargs: str) -> tuple[fits.Header, npt.NDArray[np.float64]]:
    """Return a minimal empty header and cube."""
    return fits.Header(), np.zeros((10, 3, 4))


def _setup_inspect_dir(tmp_path: Path) -> Path:
    """Set up a data dir with fake FITS and SQLite database."""
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
        str(tmp_path / "grs-test-field.fits"),
        overwrite=True,
    )

    # SQLite database
    db_path = str(data_dir / "pre-compute.db")
    conn = create_db(db_path)

    ph_run_id = insert_phspectra_run(conn, beta=3.8, n_spectra=2, total_time_s=0.03)
    insert_components(conn, "phspectra_components", ph_run_id, 1, 1, [(1.0, 25.0, 3.0)])
    insert_components(conn, "phspectra_components", ph_run_id, 2, 2, [(2.0, 25.0, 4.0)])
    insert_pixels(
        conn,
        "phspectra_pixels",
        ph_run_id,
        [(1, 1, 1, 0.1, 0.01), (2, 2, 1, 0.12, 0.02)],
    )

    gp_run_id = insert_gausspyplus_run(
        conn,
        alpha1=2.89,
        alpha2=6.65,
        phase="two",
        n_spectra=2,
        total_time_s=1.1,
    )
    insert_components(conn, "gausspyplus_components", gp_run_id, 1, 1, [(1.1, 25.5, 3.1)])
    insert_components(conn, "gausspyplus_components", gp_run_id, 2, 2, [(1.9, 24.5, 4.1)])
    insert_pixels(
        conn,
        "gausspyplus_pixels",
        gp_run_id,
        [(1, 1, 1, 0.11, 0.5), (2, 2, 1, 0.13, 0.6)],
    )

    conn.commit()
    conn.close()

    return data_dir


def test_inspect_cli(tmp_path: Path) -> None:
    """CLI should run for a valid pixel."""
    data_dir = _setup_inspect_dir(tmp_path)
    fake = _make_fake_ensure_fits(tmp_path / "grs-test-field.fits")

    with (
        patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake),
        patch("benchmarks.commands.inspect_pixel.plt.show"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["inspect", "1", "1", "--data-dir", str(data_dir), "--betas", "3.8", "--mf-snr-mins", "4.0"],
        )
    assert result.exit_code == 0, result.output


def test_inspect_missing_db(tmp_path: Path) -> None:
    """CLI should fail when pre-compute.db is missing."""
    data_dir = tmp_path / "compare-docker"
    data_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(main, ["inspect", "0", "0", "--data-dir", str(data_dir)])
    assert result.exit_code != 0


def test_inspect_pixel_not_found_no_nearby(tmp_path: Path) -> None:
    """CLI should fail when pixel is not in selection and none are nearby."""
    data_dir = _setup_inspect_dir(tmp_path)
    fake = _make_fake_ensure_fits(tmp_path / "grs-test-field.fits")

    with patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake):
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", "99", "99", "--data-dir", str(data_dir)])
    assert result.exit_code != 0


def test_inspect_pixel_not_found_nearby(tmp_path: Path) -> None:
    """CLI should suggest nearby pixels when the requested pixel is close."""
    data_dir = _setup_inspect_dir(tmp_path)
    fake = _make_fake_ensure_fits(tmp_path / "grs-test-field.fits")

    # Pixel (0,0) is within 5 of (1,1) and (2,2)
    with patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake):
        runner = CliRunner()
        result = runner.invoke(main, ["inspect", "0", "0", "--data-dir", str(data_dir)])
    assert result.exit_code != 0


def test_inspect_single_beta_single_mf(tmp_path: Path) -> None:
    """CLI 1x1 axes reshape branch: single beta, single mf_snr_min."""
    data_dir = _setup_inspect_dir(tmp_path)
    fake = _make_fake_ensure_fits(tmp_path / "grs-test-field.fits")

    with (
        patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake),
        patch("benchmarks.commands.inspect_pixel.plt.show"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["inspect", "1", "1", "--data-dir", str(data_dir), "--betas", "3.8", "--mf-snr-mins", "4.0"],
        )
    assert result.exit_code == 0, result.output


def test_inspect_multi_beta_single_mf(tmp_path: Path) -> None:
    """CLI Nx1 axes reshape branch: multiple betas, single mf_snr_min."""
    data_dir = _setup_inspect_dir(tmp_path)
    fake = _make_fake_ensure_fits(tmp_path / "grs-test-field.fits")

    with (
        patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake),
        patch("benchmarks.commands.inspect_pixel.plt.show"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["inspect", "1", "1", "--data-dir", str(data_dir), "--betas", "3.8,4.0", "--mf-snr-mins", "4.0"],
        )
    assert result.exit_code == 0, result.output


def test_inspect_single_beta_multi_mf(tmp_path: Path) -> None:
    """CLI 1xN axes reshape branch: single beta, multiple mf_snr_mins."""
    data_dir = _setup_inspect_dir(tmp_path)
    fake = _make_fake_ensure_fits(tmp_path / "grs-test-field.fits")

    with (
        patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake),
        patch("benchmarks.commands.inspect_pixel.plt.show"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["inspect", "1", "1", "--data-dir", str(data_dir), "--betas", "3.8", "--mf-snr-mins", "4.0,5.0"],
        )
    assert result.exit_code == 0, result.output


def test_inspect_multi_beta_multi_mf(tmp_path: Path) -> None:
    """CLI NxM axes branch: multiple betas and multiple mf_snr_mins."""
    data_dir = _setup_inspect_dir(tmp_path)
    fake = _make_fake_ensure_fits(tmp_path / "grs-test-field.fits")

    with (
        patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake),
        patch("benchmarks.commands.inspect_pixel.plt.show"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["inspect", "1", "1", "--data-dir", str(data_dir), "--betas", "3.8,4.0", "--mf-snr-mins", "4.0,5.0"],
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
        str(tmp_path / "grs-test-field.fits"),
        overwrite=True,
    )

    # SQLite with empty components
    db_path = str(data_dir / "pre-compute.db")
    conn = create_db(db_path)
    ph_run_id = insert_phspectra_run(conn, beta=3.8, n_spectra=1, total_time_s=0.01)
    insert_pixels(conn, "phspectra_pixels", ph_run_id, [(1, 1, 0, 0.1, 0.01)])

    gp_run_id = insert_gausspyplus_run(
        conn,
        alpha1=2.89,
        alpha2=6.65,
        phase="two",
        n_spectra=1,
        total_time_s=0.5,
    )
    insert_pixels(conn, "gausspyplus_pixels", gp_run_id, [(1, 1, 0, 0.11, 0.5)])

    conn.commit()
    conn.close()

    fake = _make_fake_ensure_fits(tmp_path / "grs-test-field.fits")

    with (
        patch("benchmarks.commands.inspect_pixel.ensure_fits", side_effect=fake),
        patch("benchmarks.commands.inspect_pixel.plt.show"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["inspect", "1", "1", "--data-dir", str(data_dir), "--betas", "3.8", "--mf-snr-mins", "4.0"],
        )
    assert result.exit_code == 0, result.output
