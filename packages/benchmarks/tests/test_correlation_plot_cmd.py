"""Tests for the ``benchmarks correlation-plot`` command."""

from __future__ import annotations

from pathlib import Path

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
from click.testing import CliRunner
from matplotlib import pyplot as plt

# pylint: disable=redefined-outer-name

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def decomposition_table() -> pa.Table:
    """A 20x15 decomposition table with known values."""
    rows = []
    for yi in range(15):
        for xi in range(20):
            rows.append(
                {
                    "x": xi,
                    "y": yi,
                    "n_components": 1 + (xi % 2),
                    "component_amplitudes": [1.0 + xi * 0.5] * (1 + (xi % 2)),
                    "component_means": [50.0 + yi * 10.0] * (1 + (xi % 2)),
                    "component_stddevs": [3.0] * (1 + (xi % 2)),
                }
            )
    return pa.table(
        {
            "x": pa.array([r["x"] for r in rows], type=pa.int32()),
            "y": pa.array([r["y"] for r in rows], type=pa.int32()),
            "n_components": pa.array([r["n_components"] for r in rows], type=pa.int32()),
            "component_amplitudes": pa.array([r["component_amplitudes"] for r in rows], type=pa.list_(pa.float64())),
            "component_means": pa.array([r["component_means"] for r in rows], type=pa.list_(pa.float64())),
            "component_stddevs": pa.array([r["component_stddevs"] for r in rows], type=pa.list_(pa.float64())),
        }
    )


@pytest.fixture()
def cached_parquet(tmp_path: Path, decomposition_table: pa.Table) -> Path:
    """Write decomposition parquet to a cache dir layout."""
    out = tmp_path / "decompositions" / "grs-26"
    out.mkdir(parents=True)
    pq.write_table(decomposition_table, out / "part-0.parquet")
    return tmp_path


@pytest.fixture()
def fits_file(tmp_path: Path) -> Path:
    """Create a minimal FITS file with CDELT2 header."""
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 20
    header["NAXIS2"] = 15
    header["NAXIS3"] = 10
    header["CDELT2"] = 0.00611
    data = np.zeros((10, 15, 20), dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data, header=header)
    path = tmp_path / "test.fits"
    hdu.writeto(str(path), overwrite=True)
    return path


# ---------------------------------------------------------------------------
# TestBuildScalarGrids
# ---------------------------------------------------------------------------


class TestBuildScalarGrids:
    """Tests for _build_scalar_grids."""

    def test_returns_four_fields(self, decomposition_table: pa.Table) -> None:
        """Return grids for all four field keys."""
        grids = _build_scalar_grids(decomposition_table)
        assert set(grids.keys()) == {"ncomp", "intensity", "velocity", "dispersion"}

    def test_grid_shapes(self, decomposition_table: pa.Table) -> None:
        """All grids should have shape (ny, nx) = (15, 20)."""
        grids = _build_scalar_grids(decomposition_table)
        for grid, mask in grids.values():
            assert grid.shape == (15, 20)
            assert mask.shape == (15, 20)

    def test_ncomp_values(self, decomposition_table: pa.Table) -> None:
        """N_comp grid should match the input n_components."""
        grids = _build_scalar_grids(decomposition_table)
        grid, mask = grids["ncomp"]
        assert mask.all()
        assert grid[0, 0] == 1.0
        assert grid[0, 1] == 2.0

    def test_all_pixels_valid(self, decomposition_table: pa.Table) -> None:
        """All pixels should be valid (no NaN sentinel)."""
        grids = _build_scalar_grids(decomposition_table)
        for _grid, mask in grids.values():
            assert mask.all()


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
        """Show help text including --fits-file option."""
        result = CliRunner().invoke(main, ["correlation-plot", "--help"])
        assert result.exit_code == 0
        assert "--fits-file" in result.output

    def test_fits_file_required(self) -> None:
        """Fail when --fits-file is not provided."""
        result = CliRunner().invoke(main, ["correlation-plot"])
        assert result.exit_code != 0

    @pytest.mark.usefixtures("docs_img_dir")
    def test_end_to_end(self, cached_parquet: Path, fits_file: Path) -> None:
        """Run the full CLI command with cached parquet data."""
        result = CliRunner().invoke(
            main,
            [
                "correlation-plot",
                "--survey",
                "grs-26",
                "--fits-file",
                str(fits_file),
                "--cache-dir",
                str(cached_parquet),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Done." in result.output
