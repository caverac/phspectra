"""Tests for benchmarks.commands.download."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from benchmarks.cli import main
from click.testing import CliRunner


def _fake_ensure_fits(url: str, path: str) -> tuple:
    """Return a minimal header and cube."""
    hdr = fits.Header()
    hdr["NAXIS1"] = 4
    hdr["NAXIS2"] = 3
    hdr["CTYPE1"] = "GLON-CAR"
    data = np.ones((10, 3, 4))
    return hdr, data


def _fake_fits_bounds(header: fits.Header) -> tuple:
    return (29.0, 31.0, -1.0, 1.0)


def test_download_cli(tmp_path: Path) -> None:
    """CLI should call ensure_fits, fits_bounds, ensure_catalog."""
    from astropy.table import Table

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
    from astropy.table import Table

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
