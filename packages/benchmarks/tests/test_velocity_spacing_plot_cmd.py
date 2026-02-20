"""Tests for the ``benchmarks velocity-spacing-plot`` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from benchmarks.cli import main
from benchmarks.commands.survey_map import DecompositionData
from benchmarks.commands.velocity_spacing_plot import (
    _build_figure,
    _compute_spacings,
    _read_first_header,
    _survey_from_grs_filename,
)
from click.testing import CliRunner
from matplotlib import pyplot as plt

# pylint: disable=redefined-outer-name

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grs_header(
    naxis1: int = 3,
    naxis2: int = 2,
    naxis3: int = 10,
    crpix1: float = 1.0,
    crpix2: float = 1.0,
) -> fits.Header:
    """Build a minimal GRS-like FITS header."""
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = naxis1
    header["NAXIS2"] = naxis2
    header["NAXIS3"] = naxis3
    header["CTYPE1"] = "GLON-CAR"
    header["CTYPE2"] = "GLAT-CAR"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRPIX1"] = crpix1
    header["CRPIX2"] = crpix2
    header["CDELT1"] = -0.00611
    header["CDELT2"] = 0.00611
    header["CRVAL3"] = -50000.0
    header["CRPIX3"] = 1.0
    header["CDELT3"] = 500.0
    return header


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def grs_tiles_dir(tmp_path: Path) -> Path:
    """Create 2 small GRS FITS tiles."""
    for name in ("grs-26-cube.fits", "grs-28-cube.fits"):
        h = _make_grs_header()
        data = np.zeros((10, 2, 3), dtype=np.float32)
        hdu = fits.PrimaryHDU(data=data, header=h)
        hdu.writeto(str(tmp_path / name), overwrite=True)
    return tmp_path


@pytest.fixture()
def decomposition_parquet(tmp_path: Path) -> Path:
    """Write a Parquet file with multi-component spectra for spacing tests."""
    table = pa.table(
        {
            "x": pa.array([0, 1, 2, 0, 1, 2], type=pa.int32()),
            "y": pa.array([0, 0, 0, 1, 1, 1], type=pa.int32()),
            "n_components": pa.array([2, 3, 1, 2, 5, 2], type=pa.int32()),
            "component_amplitudes": pa.array(
                [[2.0, 3.0], [1.0, 2.0, 1.5], [0.5], [3.0, 2.0], [1.0, 1.5, 2.0, 1.0, 1.0], [4.0, 3.0]],
                type=pa.list_(pa.float64()),
            ),
            "component_means": pa.array(
                [[1.0, 5.0], [0.0, 3.0, 7.0], [5.0], [2.0, 8.0], [0.0, 2.0, 4.0, 6.0, 8.0], [1.0, 9.0]],
                type=pa.list_(pa.float64()),
            ),
            "component_stddevs": pa.array(
                [[1.0, 1.0], [1.0, 1.0, 1.0], [1.0], [1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0]],
                type=pa.list_(pa.float64()),
            ),
        }
    )
    out = tmp_path / "parquet"
    out.mkdir()
    pq.write_table(table, out / "part-0.parquet")
    return out


# ---------------------------------------------------------------------------
# TestSurveyFromGrsFilename
# ---------------------------------------------------------------------------


class TestSurveyFromGrsFilename:
    """Tests for _survey_from_grs_filename."""

    def test_strips_cube_suffix(self) -> None:
        """Should remove -cube suffix."""
        assert _survey_from_grs_filename("grs-15-cube.fits") == "grs-15"

    def test_lowercases(self) -> None:
        """Should lowercase the stem."""
        assert _survey_from_grs_filename("GRS-15.fits") == "grs-15"

    def test_no_cube_suffix(self) -> None:
        """Should work without -cube suffix."""
        assert _survey_from_grs_filename("grs-26.fits") == "grs-26"


# ---------------------------------------------------------------------------
# TestReadFirstHeader
# ---------------------------------------------------------------------------


class TestReadFirstHeader:
    """Tests for _read_first_header."""

    def test_returns_header(self, grs_tiles_dir: Path) -> None:
        """Should return a FITS header from the first matching tile."""
        header = _read_first_header(str(grs_tiles_dir))
        assert "NAXIS3" in header

    def test_raises_when_no_tiles(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when no matching tiles exist."""
        with pytest.raises(FileNotFoundError, match="No matching GRS tiles"):
            _read_first_header(str(tmp_path))

    def test_ignores_non_matching_tiles(self, tmp_path: Path) -> None:
        """Should ignore tiles not in the TILES list."""
        h = _make_grs_header()
        data = np.zeros((10, 2, 3), dtype=np.float32)
        hdu = fits.PrimaryHDU(data=data, header=h)
        hdu.writeto(str(tmp_path / "grs-99-cube.fits"), overwrite=True)
        with pytest.raises(FileNotFoundError):
            _read_first_header(str(tmp_path))


# ---------------------------------------------------------------------------
# TestComputeSpacings
# ---------------------------------------------------------------------------


class TestComputeSpacings:
    """Tests for _compute_spacings."""

    def test_basic_two_component(self) -> None:
        """Two well-separated components should produce one spacing."""
        velocity = np.arange(10, dtype=np.float64) * 1000.0  # 0-9 km/s
        data = DecompositionData(
            x=np.array([0], dtype=np.int32),
            y=np.array([0], dtype=np.int32),
            n_components=np.array([2], dtype=np.int32),
            component_amplitudes=[[2.0, 3.0]],
            component_means=[[2.0, 7.0]],
            component_stddevs=[[1.0, 1.0]],
        )
        spacings, ncomp_tags = _compute_spacings(data, velocity)
        assert len(spacings) == 1
        assert spacings[0] == pytest.approx(5.0)
        assert ncomp_tags[0] == 2

    def test_filters_low_amplitude(self) -> None:
        """Components below the amplitude threshold should be excluded."""
        velocity = np.arange(10, dtype=np.float64) * 1000.0
        data = DecompositionData(
            x=np.array([0], dtype=np.int32),
            y=np.array([0], dtype=np.int32),
            n_components=np.array([2], dtype=np.int32),
            component_amplitudes=[[0.1, 3.0]],  # first below 3*NOISE_SIGMA
            component_means=[[2.0, 7.0]],
            component_stddevs=[[1.0, 1.0]],
        )
        spacings, ncomp_tags = _compute_spacings(data, velocity)
        assert len(spacings) == 0
        assert len(ncomp_tags) == 0

    def test_single_component_ignored(self) -> None:
        """Single-component spectra produce no spacings."""
        velocity = np.arange(10, dtype=np.float64) * 1000.0
        data = DecompositionData(
            x=np.array([0], dtype=np.int32),
            y=np.array([0], dtype=np.int32),
            n_components=np.array([1], dtype=np.int32),
            component_amplitudes=[[5.0]],
            component_means=[[5.0]],
            component_stddevs=[[1.0]],
        )
        spacings, ncomp_tags = _compute_spacings(data, velocity)  # pylint: disable=unused-variable
        assert len(spacings) == 0

    def test_three_components_two_spacings(self) -> None:
        """Three components should produce two adjacent spacings."""
        velocity = np.arange(20, dtype=np.float64) * 1000.0
        data = DecompositionData(
            x=np.array([0], dtype=np.int32),
            y=np.array([0], dtype=np.int32),
            n_components=np.array([3], dtype=np.int32),
            component_amplitudes=[[2.0, 3.0, 2.5]],
            component_means=[[2.0, 8.0, 14.0]],
            component_stddevs=[[1.0, 1.0, 1.0]],
        )
        spacings, ncomp_tags = _compute_spacings(data, velocity)
        assert len(spacings) == 2
        assert spacings[0] == pytest.approx(6.0)
        assert spacings[1] == pytest.approx(6.0)
        assert all(t == 3 for t in ncomp_tags)

    def test_multiple_spectra(self) -> None:
        """Spacings accumulate across multiple spectra."""
        velocity = np.arange(10, dtype=np.float64) * 1000.0
        data = DecompositionData(
            x=np.array([0, 1], dtype=np.int32),
            y=np.array([0, 0], dtype=np.int32),
            n_components=np.array([2, 2], dtype=np.int32),
            component_amplitudes=[[2.0, 3.0], [2.0, 3.0]],
            component_means=[[1.0, 5.0], [2.0, 8.0]],
            component_stddevs=[[1.0, 1.0], [1.0, 1.0]],
        )
        spacings, ncomp_tags = _compute_spacings(data, velocity)  # pylint: disable=unused-variable
        assert len(spacings) == 2


# ---------------------------------------------------------------------------
# TestBuildFigure
# ---------------------------------------------------------------------------


class TestBuildFigure:
    """Tests for _build_figure."""

    @pytest.mark.usefixtures("docs_img_dir")
    def test_returns_figure(self) -> None:
        """Should return a matplotlib figure."""
        spacings = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
        ncomp_tags = np.array([2, 2, 3, 4, 5], dtype=np.int32)
        fig = _build_figure(spacings, ncomp_tags)
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    @pytest.mark.usefixtures("docs_img_dir")
    def test_handles_all_same_ncomp(self) -> None:
        """Should handle data where all spectra have the same N_comp."""
        spacings = np.array([1.0, 2.0, 3.0])
        ncomp_tags = np.array([2, 2, 2], dtype=np.int32)
        fig = _build_figure(spacings, ncomp_tags)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.usefixtures("docs_img_dir")
    def test_handles_high_ncomp(self) -> None:
        """Should render the N>=5 group."""
        spacings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ncomp_tags = np.array([5, 6, 7, 5, 6], dtype=np.int32)
        fig = _build_figure(spacings, ncomp_tags)
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# TestVelocitySpacingPlotCli
# ---------------------------------------------------------------------------


class TestVelocitySpacingPlotCli:
    """Tests for the velocity-spacing-plot CLI command."""

    def test_help(self) -> None:
        """Show help text including --input-dir option."""
        result = CliRunner().invoke(main, ["velocity-spacing-plot", "--help"])
        assert result.exit_code == 0
        assert "--input-dir" in result.output

    def test_input_dir_required(self) -> None:
        """Fail when --input-dir is not provided."""
        result = CliRunner().invoke(main, ["velocity-spacing-plot"])
        assert result.exit_code != 0

    def test_no_tiles_aborts(self, tmp_path: Path) -> None:
        """Exit with error when no matching FITS tiles found."""
        result = CliRunner().invoke(main, ["velocity-spacing-plot", "--input-dir", str(tmp_path)])
        assert result.exit_code != 0

    def test_no_decomposition_data_aborts(self, grs_tiles_dir: Path) -> None:
        """Exit gracefully when tiles exist but decomposition data is empty."""

        def mock_download(_survey: str, _environment: str, _cache_dir: str, _force: bool) -> list[str]:
            return []

        with patch(
            "benchmarks.commands.velocity_spacing_plot._download_decompositions",
            side_effect=mock_download,
        ):
            result = CliRunner().invoke(
                main,
                [
                    "velocity-spacing-plot",
                    "--input-dir",
                    str(grs_tiles_dir),
                    "--cache-dir",
                    str(grs_tiles_dir),
                ],
            )

        assert result.exit_code == 0
        assert "No decomposition data found" in result.output

    @pytest.mark.usefixtures("docs_img_dir")
    def test_end_to_end(self, grs_tiles_dir: Path, decomposition_parquet: Path) -> None:
        """Run the full CLI command with mocked decomposition downloads."""
        parquet_path = str(decomposition_parquet / "part-0.parquet")

        def mock_download(_survey: str, _environment: str, _cache_dir: str, _force: bool) -> list[str]:
            return [parquet_path]

        with patch(
            "benchmarks.commands.velocity_spacing_plot._download_decompositions",
            side_effect=mock_download,
        ):
            result = CliRunner().invoke(
                main,
                [
                    "velocity-spacing-plot",
                    "--input-dir",
                    str(grs_tiles_dir),
                    "--cache-dir",
                    str(grs_tiles_dir),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Done." in result.output
