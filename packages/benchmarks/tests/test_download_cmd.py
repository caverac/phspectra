"""Tests for benchmarks.commands.download."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.table import Table
from benchmarks.cli import main
from click.testing import CliRunner


def _fake_ensure_fits(_url: str, _path: str) -> tuple[fits.Header, npt.NDArray[np.float64]]:
    """Return a minimal header and cube."""
    hdr = fits.Header()
    hdr["NAXIS1"] = 4
    hdr["NAXIS2"] = 3
    hdr["CTYPE1"] = "GLON-CAR"
    data = np.ones((10, 3, 4))
    return hdr, data


def _fake_fits_bounds(_header: fits.Header) -> tuple[float, float, float, float]:
    """Return fixed galactic bounds."""
    return (29.0, 31.0, -1.0, 1.0)


def test_download_cli(tmp_path: Path) -> None:
    """CLI should call ensure_fits, fits_bounds, ensure_catalog."""
    cat = Table({"GLON": [30.0], "GLAT": [0.0]})

    with (
        patch("benchmarks.commands.download.ensure_fits", side_effect=_fake_ensure_fits),
        patch("benchmarks.commands.download.fits_bounds", side_effect=_fake_fits_bounds),
        patch("benchmarks.commands.download.ensure_catalog", return_value=cat),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output


def test_download_force(tmp_path: Path) -> None:
    """CLI --force should remove cached files before downloading."""
    # Create fake cached files
    (tmp_path / "grs-test-field.fits").write_text("fake")
    (tmp_path / "gausspy-catalog.votable").write_text("fake")

    cat = Table({"GLON": [30.0], "GLAT": [0.0]})

    with (
        patch("benchmarks.commands.download.ensure_fits", side_effect=_fake_ensure_fits),
        patch("benchmarks.commands.download.fits_bounds", side_effect=_fake_fits_bounds),
        patch("benchmarks.commands.download.ensure_catalog", return_value=cat),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--cache-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
