"""Tests for the ``benchmarks grs-map-plot`` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from benchmarks.cli import main
from benchmarks.commands.grs_map_plot import (
    GlobalGrid,
    _build_grs_figure,
    _compute_global_grid,
    _load_global_data,
    _read_tile_infos,
)
from benchmarks.commands.survey_map import DecompositionData
from click.testing import CliRunner
from matplotlib import pyplot as plt

# pylint: disable=redefined-outer-name

# ---------------------------------------------------------------------------
# Fixtures
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


@pytest.fixture()
def grs_tiles_dir(tmp_path: Path) -> Path:
    """Create 2 small GRS FITS tiles with different CRPIX1 (adjacent in lon)."""
    # Tile A: 3x2 pixels, CRPIX1=1 (pixels 0..2 in global lon)
    h_a = _make_grs_header(naxis1=3, naxis2=2, naxis3=10, crpix1=1.0, crpix2=1.0)
    data_a = np.zeros((10, 2, 3), dtype=np.float32)
    hdu_a = fits.PrimaryHDU(data=data_a, header=h_a)
    hdu_a.writeto(str(tmp_path / "grs-26-cube.fits"), overwrite=True)

    # Tile B: 3x2 pixels, CRPIX1=-2 (adjacent, pixels 3..5 in global lon)
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
# TestReadTileInfos
# ---------------------------------------------------------------------------


class TestReadTileInfos:
    """Tests for _read_tile_infos."""

    def test_correct_count(self, grs_tiles_dir: Path) -> None:
        """Return one TileInfo per FITS cube found."""
        tiles = _read_tile_infos(str(grs_tiles_dir))
        assert len(tiles) == 2

    def test_survey_names(self, grs_tiles_dir: Path) -> None:
        """Extract survey name from the FITS filename."""
        tiles = _read_tile_infos(str(grs_tiles_dir))
        names = {t.survey for t in tiles}
        assert names == {"grs-26", "grs-28"}

    def test_header_values(self, grs_tiles_dir: Path) -> None:
        """Parse NAXIS and CRPIX values from the FITS header."""
        tiles = _read_tile_infos(str(grs_tiles_dir))
        tile_a = next(t for t in tiles if t.survey == "grs-26")
        assert tile_a.naxis1 == 3
        assert tile_a.naxis2 == 2
        assert tile_a.naxis3 == 10
        assert tile_a.crpix1 == 1.0

    def test_empty_dir(self, tmp_path: Path) -> None:
        """Return an empty list when no FITS files exist."""
        tiles = _read_tile_infos(str(tmp_path))
        assert not tiles


# ---------------------------------------------------------------------------
# TestComputeGlobalGrid
# ---------------------------------------------------------------------------


class TestComputeGlobalGrid:
    """Tests for _compute_global_grid."""

    def test_dimensions(self, grs_tiles_dir: Path) -> None:
        """Compute the combined grid dimensions from adjacent tiles."""
        tiles = _read_tile_infos(str(grs_tiles_dir))
        grid = _compute_global_grid(tiles)
        # Tile A: CRPIX1=1, NAXIS1=3 -> pix 0 at offset 0, pix 2 at offset 2
        # Tile B: CRPIX1=-2, NAXIS1=3 -> pix 0 at offset 3, pix 2 at offset 5
        assert grid.nx == 6
        assert grid.ny == 2

    def test_offsets_non_negative(self, grs_tiles_dir: Path) -> None:
        """All tile offsets should be zero or positive."""
        tiles = _read_tile_infos(str(grs_tiles_dir))
        grid = _compute_global_grid(tiles)
        for x_off, y_off in grid.offsets.values():
            assert x_off >= 0
            assert y_off >= 0

    def test_naxis3_uses_max(self, grs_tiles_dir: Path) -> None:
        """Use the maximum spectral axis length across all tiles."""
        tiles = _read_tile_infos(str(grs_tiles_dir))
        grid = _compute_global_grid(tiles)
        assert grid.naxis3 == max(t.naxis3 for t in tiles)

    def test_header_crpix_consistent(self, grs_tiles_dir: Path) -> None:
        """Global header NAXIS values match the computed grid."""
        tiles = _read_tile_infos(str(grs_tiles_dir))
        grid = _compute_global_grid(tiles)
        assert grid.header["NAXIS1"] == grid.nx
        assert grid.header["NAXIS2"] == grid.ny
        assert grid.header["NAXIS3"] == grid.naxis3


# ---------------------------------------------------------------------------
# TestLoadGlobalData
# ---------------------------------------------------------------------------


class TestLoadGlobalData:
    """Tests for _load_global_data."""

    def test_coordinates_remapped(self, grs_tiles_dir: Path, decomposition_parquet: Path) -> None:
        """Coordinates from tile B should be shifted by its offset."""
        tiles = _read_tile_infos(str(grs_tiles_dir))
        grid = _compute_global_grid(tiles)

        parquet_path = str(decomposition_parquet / "part-0.parquet")

        def mock_download(_survey: str, _environment: str, _cache_dir: str, _force: bool) -> list[str]:
            return [parquet_path]

        with patch(
            "benchmarks.commands.grs_map_plot._download_decompositions",
            side_effect=mock_download,
        ):
            data = _load_global_data(tiles, grid, "development", "/tmp", False)

        # Each tile contributes 6 pixels -> 12 total
        assert len(data.x) == 12
        # At least some remapped x values should be > 2 (offset from tile B)
        assert data.x.max() > 2

    def test_empty_tile_skipped(self, grs_tiles_dir: Path) -> None:
        """Tiles with no decomposition data are skipped."""
        tiles = _read_tile_infos(str(grs_tiles_dir))
        grid = _compute_global_grid(tiles)

        def mock_download(_survey: str, _environment: str, _cache_dir: str, _force: bool) -> list[str]:
            return []

        with patch(
            "benchmarks.commands.grs_map_plot._download_decompositions",
            side_effect=mock_download,
        ):
            data = _load_global_data(tiles, grid, "development", "/tmp", False)

        assert len(data.x) == 0


# ---------------------------------------------------------------------------
# TestBuildGrsFigure
# ---------------------------------------------------------------------------


class TestBuildGrsFigure:
    """Tests for _build_grs_figure."""

    @pytest.mark.usefixtures("docs_img_dir")
    def test_returns_figure_with_axes(self) -> None:
        """Build a figure with at least one axis (bivariate panel)."""
        header = _make_grs_header(naxis1=6, naxis2=2)
        grid = GlobalGrid(
            nx=6,
            ny=2,
            naxis3=10,
            offsets={"grs-26": (0, 0), "grs-28": (3, 0)},
            header=header,
        )
        data = DecompositionData(
            x=np.array([0, 1, 3, 4], dtype=np.int32),
            y=np.array([0, 0, 1, 1], dtype=np.int32),
            n_components=np.array([1, 1, 1, 1], dtype=np.int32),
            component_amplitudes=[[1.0], [2.0], [1.5], [2.5]],
            component_means=[[5.0], [5.0], [5.0], [5.0]],
            component_stddevs=[[1.0], [1.0], [1.0], [1.0]],
        )
        velocity = np.linspace(-50000, 161500, 10, dtype=np.float64)

        fig = _build_grs_figure(data, velocity, grid)
        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)


# ---------------------------------------------------------------------------
# TestGrsMapPlotCli
# ---------------------------------------------------------------------------


class TestGrsMapPlotCli:
    """Tests for the grs-map-plot CLI command."""

    def test_help(self) -> None:
        """Show help text including the --input-dir option."""
        result = CliRunner().invoke(main, ["grs-map-plot", "--help"])
        assert result.exit_code == 0
        assert "--input-dir" in result.output

    def test_input_dir_required(self) -> None:
        """Fail when --input-dir is not provided."""
        result = CliRunner().invoke(main, ["grs-map-plot"])
        assert result.exit_code != 0

    def test_no_tiles_aborts(self, tmp_path: Path) -> None:
        """Exit gracefully when the input directory has no FITS tiles."""
        result = CliRunner().invoke(main, ["grs-map-plot", "--input-dir", str(tmp_path)])
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
                    "grs-map-plot",
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
                    "grs-map-plot",
                    "--input-dir",
                    str(grs_tiles_dir),
                    "--cache-dir",
                    str(grs_tiles_dir),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Done." in result.output
