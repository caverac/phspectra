"""Tests for the ``benchmarks correlation-plot`` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from benchmarks.cli import main
from benchmarks.commands.correlation_plot import (
    _autocorrelation_fft,
    _build_correlation_figure,
    _build_scalar_grids,
    _CorrelationResult,
    _jackknife_autocorrelation,
)
from benchmarks.commands.survey_map import DecompositionData
from click.testing import CliRunner
from matplotlib import pyplot as plt

# pylint: disable=redefined-outer-name

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def decomposition_data() -> DecompositionData:
    """A 20x15 decomposition dataset with known values."""
    xs, ys, ncs = [], [], []
    all_amps, all_means, all_stddevs = [], [], []
    for yi in range(15):
        for xi in range(20):
            xs.append(xi)
            ys.append(yi)
            nc = 1 + (xi % 2)
            ncs.append(nc)
            all_amps.append([1.0 + xi * 0.5] * nc)
            all_means.append([50.0 + yi * 10.0] * nc)
            all_stddevs.append([3.0] * nc)
    return DecompositionData(
        x=np.array(xs, dtype=np.int32),
        y=np.array(ys, dtype=np.int32),
        n_components=np.array(ncs, dtype=np.int32),
        component_amplitudes=all_amps,
        component_means=all_means,
        component_stddevs=all_stddevs,
    )


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


@pytest.fixture()
def grs_tiles_dir(tmp_path: Path) -> Path:
    """Create 2 small GRS FITS tiles with different CRPIX1."""
    h_a = _make_grs_header(naxis1=3, naxis2=2, naxis3=10, crpix1=1.0, crpix2=1.0)
    data_a = np.zeros((10, 2, 3), dtype=np.float32)
    hdu_a = fits.PrimaryHDU(data=data_a, header=h_a)
    hdu_a.writeto(str(tmp_path / "grs-26-cube.fits"), overwrite=True)

    h_b = _make_grs_header(naxis1=3, naxis2=2, naxis3=10, crpix1=-2.0, crpix2=1.0)
    data_b = np.zeros((10, 2, 3), dtype=np.float32)
    hdu_b = fits.PrimaryHDU(data=data_b, header=h_b)
    hdu_b.writeto(str(tmp_path / "grs-28-cube.fits"), overwrite=True)

    return tmp_path


@pytest.fixture()
def decomposition_parquet(tmp_path: Path) -> Path:
    """Write a small Parquet file for a tile (3x2 grid, 6 pixels)."""
    table = pa.table(
        {
            "x": pa.array([0, 1, 2, 0, 1, 2], type=pa.int32()),
            "y": pa.array([0, 0, 0, 1, 1, 1], type=pa.int32()),
            "n_components": pa.array([1, 1, 1, 1, 1, 1], type=pa.int32()),
            "component_amplitudes": pa.array(
                [[1.0], [2.0], [3.0], [1.5], [2.5], [3.5]],
                type=pa.list_(pa.float64()),
            ),
            "component_means": pa.array(
                [[5.0], [5.0], [5.0], [5.0], [5.0], [5.0]],
                type=pa.list_(pa.float64()),
            ),
            "component_stddevs": pa.array(
                [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
                type=pa.list_(pa.float64()),
            ),
        }
    )
    out = tmp_path / "parquet"
    out.mkdir()
    pq.write_table(table, out / "part-0.parquet")
    return out


# ---------------------------------------------------------------------------
# TestBuildScalarGrids
# ---------------------------------------------------------------------------


class TestBuildScalarGrids:
    """Tests for _build_scalar_grids."""

    def test_returns_four_fields(self, decomposition_data: DecompositionData) -> None:
        """Return grids for all four field keys."""
        grids = _build_scalar_grids(decomposition_data, 20, 15)
        assert set(grids.keys()) == {"ncomp", "intensity", "velocity", "dispersion"}

    def test_grid_shapes(self, decomposition_data: DecompositionData) -> None:
        """All grids should have shape (ny, nx) = (15, 20)."""
        grids = _build_scalar_grids(decomposition_data, 20, 15)
        for grid, mask in grids.values():
            assert grid.shape == (15, 20)
            assert mask.shape == (15, 20)

    def test_ncomp_values(self, decomposition_data: DecompositionData) -> None:
        """N_comp grid should match the input n_components."""
        grids = _build_scalar_grids(decomposition_data, 20, 15)
        grid, mask = grids["ncomp"]
        assert mask.all()
        assert grid[0, 0] == 1.0
        assert grid[0, 1] == 2.0

    def test_all_pixels_valid(self, decomposition_data: DecompositionData) -> None:
        """All pixels should be valid (no NaN sentinel)."""
        grids = _build_scalar_grids(decomposition_data, 20, 15)
        for _grid, mask in grids.values():
            assert mask.all()

    def test_empty_components_skipped(self) -> None:
        """Pixels with n_components <= 0 or empty lists are skipped."""
        data = DecompositionData(
            x=np.array([0, 1], dtype=np.int32),
            y=np.array([0, 0], dtype=np.int32),
            n_components=np.array([0, 1], dtype=np.int32),
            component_amplitudes=[[], [2.0]],
            component_means=[[], [50.0]],
            component_stddevs=[[], [3.0]],
        )
        grids = _build_scalar_grids(data, 2, 1)
        _, mask = grids["intensity"]
        assert not mask[0, 0]
        assert mask[0, 1]


# ---------------------------------------------------------------------------
# TestAutocorrelationFft
# ---------------------------------------------------------------------------


class TestAutocorrelationFft:
    """Tests for _autocorrelation_fft."""

    def test_xi_zero_is_one(self) -> None:
        """Normalised autocorrelation at lag 0 should be 1."""
        rng = np.random.default_rng(42)
        field = rng.normal(5.0, 1.0, (20, 30))
        mask = np.ones_like(field, dtype=bool)
        _radii, xi = _autocorrelation_fft(field, mask)
        assert xi[0] == pytest.approx(1.0)

    def test_decreasing_trend_for_smooth_field(self) -> None:
        """A smooth Gaussian blob should have xi that decreases overall."""
        yy, xx = np.mgrid[:64, :64]
        field = np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 8**2))
        mask = np.ones_like(field, dtype=bool)
        _, xi = _autocorrelation_fft(field, mask)
        # xi at lag 10 should be less than xi at lag 1
        assert xi[10] < xi[1]

    def test_empty_mask_returns_zeros(self) -> None:
        """An all-False mask should return zero correlation."""
        field = np.ones((5, 5))
        mask = np.zeros_like(field, dtype=bool)
        _radii, xi = _autocorrelation_fft(field, mask)
        assert xi[0] == 0.0


# ---------------------------------------------------------------------------
# TestJackknifeAutocorrelation
# ---------------------------------------------------------------------------


class TestJackknifeAutocorrelation:
    """Tests for _jackknife_autocorrelation."""

    def test_returns_three_arrays(self) -> None:
        """Return radii, xi, and xi_err arrays of equal length."""
        rng = np.random.default_rng(42)
        field = rng.normal(5.0, 1.0, (32, 32))
        mask = np.ones_like(field, dtype=bool)
        radii, xi, xi_err = _jackknife_autocorrelation(field, mask, n_blocks_y=2, n_blocks_x=2)
        assert len(radii) == len(xi) == len(xi_err)

    def test_errors_are_non_negative(self) -> None:
        """Jackknife errors should be >= 0 everywhere."""
        rng = np.random.default_rng(42)
        field = rng.normal(5.0, 1.0, (32, 32))
        mask = np.ones_like(field, dtype=bool)
        _, _, xi_err = _jackknife_autocorrelation(field, mask, n_blocks_y=2, n_blocks_x=2)
        assert (xi_err >= 0).all()


# ---------------------------------------------------------------------------
# TestBuildCorrelationFigure
# ---------------------------------------------------------------------------


class TestBuildCorrelationFigure:
    """Tests for _build_correlation_figure."""

    @pytest.mark.usefixtures("docs_img_dir")
    def test_returns_figure_with_four_panels(self) -> None:
        """Build a figure with 4 axes (2x2 grid)."""
        results: dict[str, _CorrelationResult] = {}
        radii = np.arange(50, dtype=np.float64)
        xi = np.exp(-radii / 5.0)
        xi_err = xi * 0.1
        for key in ["ncomp", "intensity", "velocity", "dispersion"]:
            results[key] = (radii, xi, xi_err)

        fig = _build_correlation_figure(results, 0.367, 15.0)
        assert fig is not None
        assert len(fig.axes) == 4
        plt.close(fig)


# ---------------------------------------------------------------------------
# TestCorrelationPlotCli
# ---------------------------------------------------------------------------


class TestCorrelationPlotCli:
    """Tests for the correlation-plot CLI command."""

    def test_help(self) -> None:
        """Show help text including --input-dir option."""
        result = CliRunner().invoke(main, ["correlation-plot", "--help"])
        assert result.exit_code == 0
        assert "--input-dir" in result.output

    def test_input_dir_required(self) -> None:
        """Fail when --input-dir is not provided."""
        result = CliRunner().invoke(main, ["correlation-plot"])
        assert result.exit_code != 0

    def test_no_tiles_aborts(self, tmp_path: Path) -> None:
        """Exit gracefully when the input directory has no FITS tiles."""
        result = CliRunner().invoke(main, ["correlation-plot", "--input-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No FITS tiles" in result.output

    def test_no_decomposition_data_aborts(self, grs_tiles_dir: Path) -> None:
        """Exit gracefully when tiles exist but decomposition data is empty."""

        def mock_download(_survey: str, _environment: str, _cache_dir: str, _force: bool) -> list[str]:
            return []

        with patch(
            "benchmarks.commands.grs_map_plot._download_decompositions",
            side_effect=mock_download,
        ):
            result = CliRunner().invoke(
                main,
                [
                    "correlation-plot",
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
            "benchmarks.commands.grs_map_plot._download_decompositions",
            side_effect=mock_download,
        ):
            result = CliRunner().invoke(
                main,
                [
                    "correlation-plot",
                    "--input-dir",
                    str(grs_tiles_dir),
                    "--cache-dir",
                    str(grs_tiles_dir),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Done." in result.output
