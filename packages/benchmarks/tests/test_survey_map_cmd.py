"""Tests for benchmarks.commands.survey_map utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from benchmarks.commands.survey_map import (
    DecompositionData,
    _download_decompositions,
    _galactic_extent,
    _load_parquet_data,
    _panel_bivariate,
    _panel_complexity,
    _panel_dominant_velocity,
    _panel_velocity_rgb,
    _table_to_arrays,
    _velocity_axis,
)
from matplotlib import pyplot as plt

# pylint: disable=redefined-outer-name

MOCK_BOTO3 = "benchmarks.commands.survey_map.boto3"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def decomposition_parquet(tmp_path: Path) -> Path:
    """Write a small Parquet file with 4 pixels (2x2 grid)."""
    table = pa.table(
        {
            "x": pa.array([0, 1, 0, 1], type=pa.int32()),
            "y": pa.array([0, 0, 1, 1], type=pa.int32()),
            "n_components": pa.array([1, 2, 1, 0], type=pa.int32()),
            "component_amplitudes": pa.array(
                [[3.0], [2.0, 1.5], [4.0], []],
                type=pa.list_(pa.float64()),
            ),
            "component_means": pa.array(
                [[100.0], [50.0, 300.0], [200.0], []],
                type=pa.list_(pa.float64()),
            ),
            "component_stddevs": pa.array(
                [[5.0], [3.0, 4.0], [6.0], []],
                type=pa.list_(pa.float64()),
            ),
        }
    )
    path = tmp_path / "part-0.parquet"
    pq.write_table(table, path)
    return tmp_path


@pytest.fixture()
def fits_header() -> fits.Header:
    """Minimal FITS header with velocity axis and spatial WCS keywords."""
    header = fits.Header()
    # Velocity axis
    header["NAXIS3"] = 424
    header["CRVAL3"] = -50000.0  # m/s
    header["CRPIX3"] = 1.0
    header["CDELT3"] = 500.0  # m/s per channel
    # Spatial WCS (galactic CAR projection)
    header["CTYPE1"] = "GLON-CAR"
    header["CTYPE2"] = "GLAT-CAR"
    header["CRVAL1"] = 10.0  # deg
    header["CRPIX1"] = 1.0
    header["CDELT1"] = -0.05  # deg/px
    header["CRVAL2"] = 0.0  # deg
    header["CRPIX2"] = 1.0
    header["CDELT2"] = 0.05  # deg/px
    return header


@pytest.fixture()
def sample_data() -> DecompositionData:
    """Small 2x2 DecompositionData for panel tests."""
    return DecompositionData(
        x=np.array([0, 1, 0, 1], dtype=np.int32),
        y=np.array([0, 0, 1, 1], dtype=np.int32),
        n_components=np.array([1, 2, 1, 0], dtype=np.int32),
        component_amplitudes=[[3.0], [2.0, 1.5], [4.0], []],
        component_means=[[100.0], [50.0, 300.0], [200.0], []],
        component_stddevs=[[5.0], [3.0, 4.0], [6.0], []],
    )


@pytest.fixture()
def sample_velocity(fits_header: fits.Header) -> npt.NDArray[np.float64]:
    """Velocity axis from the test header."""
    return _velocity_axis(fits_header)


@pytest.fixture()
def sample_extent(fits_header: fits.Header) -> tuple[float, float, float, float]:
    """Galactic extent for the 2x2 test grid."""
    return _galactic_extent(fits_header, 2, 2)


# ---------------------------------------------------------------------------
# TestVelocityAxis
# ---------------------------------------------------------------------------


class TestVelocityAxis:
    """Tests for _velocity_axis."""

    def test_shape(self, fits_header: fits.Header) -> None:
        """Velocity axis length matches NAXIS3 from the header."""
        vel = _velocity_axis(fits_header)
        assert vel.shape == (424,)

    def test_values(self, fits_header: fits.Header) -> None:
        """Velocity values at specific channels match manual WCS calculation."""
        vel = _velocity_axis(fits_header)
        # channel 0: CRVAL3 + (0 - (CRPIX3 - 1)) * CDELT3 = -50000 + 0 = -50000
        assert vel[0] == pytest.approx(-50000.0)
        # channel 1: -50000 + 1 * 500 = -49500
        assert vel[1] == pytest.approx(-49500.0)
        # channel 423: -50000 + 423 * 500 = 161500
        assert vel[423] == pytest.approx(161500.0)


# ---------------------------------------------------------------------------
# TestGalacticExtent
# ---------------------------------------------------------------------------


class TestGalacticExtent:
    """Tests for _galactic_extent."""

    def test_returns_four_floats(self, fits_header: fits.Header) -> None:
        """Extent tuple contains exactly four float values."""
        extent = _galactic_extent(fits_header, 2, 2)
        assert len(extent) == 4
        assert all(isinstance(v, float) for v in extent)

    def test_lon_range(self, fits_header: fits.Header) -> None:
        """Longitude minimum is strictly less than longitude maximum."""
        lon_min, lon_max, _, _ = _galactic_extent(fits_header, 2, 2)
        assert lon_min < lon_max

    def test_lat_range(self, fits_header: fits.Header) -> None:
        """Latitude minimum is strictly less than latitude maximum."""
        _, _, lat_min, lat_max = _galactic_extent(fits_header, 2, 2)
        assert lat_min < lat_max

    def test_values_consistent_with_header(self, fits_header: fits.Header) -> None:
        """Extent spans match pixel count times CDELT from the header."""
        extent = _galactic_extent(fits_header, 2, 2)
        lon_min, lon_max, lat_min, lat_max = extent
        # CDELT1=-0.05, 2 pixels: span ~ 0.05 deg
        assert lon_max - lon_min == pytest.approx(0.05, abs=1e-10)
        # CDELT2=+0.05, 2 pixels: span ~ 0.05 deg
        assert lat_max - lat_min == pytest.approx(0.05, abs=1e-10)


# ---------------------------------------------------------------------------
# TestLoadParquetData
# ---------------------------------------------------------------------------


class TestLoadParquetData:
    """Tests for _load_parquet_data."""

    def test_single_file(self, decomposition_parquet: Path) -> None:
        """Loading a single Parquet file returns all its rows."""
        paths = [str(decomposition_parquet / "part-0.parquet")]
        table = _load_parquet_data(paths)
        assert len(table) == 4

    def test_concatenation(self, decomposition_parquet: Path) -> None:
        """Loading multiple Parquet files concatenates their rows."""
        # Write a second file
        table2 = pa.table(
            {
                "x": pa.array([2], type=pa.int32()),
                "y": pa.array([2], type=pa.int32()),
                "n_components": pa.array([1], type=pa.int32()),
                "component_amplitudes": pa.array([[1.0]], type=pa.list_(pa.float64())),
                "component_means": pa.array([[150.0]], type=pa.list_(pa.float64())),
                "component_stddevs": pa.array([[2.0]], type=pa.list_(pa.float64())),
            }
        )
        path2 = decomposition_parquet / "part-1.parquet"
        pq.write_table(table2, path2)

        paths = [
            str(decomposition_parquet / "part-0.parquet"),
            str(path2),
        ]
        table = _load_parquet_data(paths)
        assert len(table) == 5


# ---------------------------------------------------------------------------
# TestTableToArrays
# ---------------------------------------------------------------------------


class TestTableToArrays:
    """Tests for _table_to_arrays."""

    def test_scalar_columns_are_numpy(self, decomposition_parquet: Path) -> None:
        """Scalar columns are converted to NumPy arrays."""
        table = pq.read_table(decomposition_parquet / "part-0.parquet")
        data = _table_to_arrays(table)
        assert isinstance(data.x, np.ndarray)
        assert isinstance(data.y, np.ndarray)
        assert isinstance(data.n_components, np.ndarray)

    def test_list_columns_are_lists(self, decomposition_parquet: Path) -> None:
        """Variable-length columns are converted to nested Python lists."""
        table = pq.read_table(decomposition_parquet / "part-0.parquet")
        data = _table_to_arrays(table)
        assert isinstance(data.component_amplitudes, list)
        assert isinstance(data.component_amplitudes[0], list)
        assert isinstance(data.component_means, list)
        assert isinstance(data.component_stddevs, list)

    def test_values_match(self, decomposition_parquet: Path) -> None:
        """Converted values match the original Parquet data."""
        table = pq.read_table(decomposition_parquet / "part-0.parquet")
        data = _table_to_arrays(table)
        np.testing.assert_array_equal(data.x, [0, 1, 0, 1])
        np.testing.assert_array_equal(data.n_components, [1, 2, 1, 0])
        assert data.component_amplitudes[1] == [2.0, 1.5]


# ---------------------------------------------------------------------------
# TestDownloadDecompositions
# ---------------------------------------------------------------------------


class TestDownloadDecompositions:
    """Tests for _download_decompositions."""

    def _mock_paginator(self, keys: list[str]) -> MagicMock:
        """Build a mock paginator returning the given keys."""
        page = {"Contents": [{"Key": k} for k in keys]}
        paginator = MagicMock()
        paginator.paginate.return_value = [page]
        return paginator

    def test_cache_miss(self, tmp_path: Path) -> None:
        """File is downloaded when not present in the local cache."""
        with patch(MOCK_BOTO3) as mock_b3:
            client = MagicMock()
            mock_b3.client.return_value = client
            client.get_paginator.return_value = self._mock_paginator(["decompositions/survey=test/part-0.parquet"])
            paths = _download_decompositions("test", "development", str(tmp_path), False)

        assert len(paths) == 1
        client.download_file.assert_called_once()

    def test_cache_hit(self, tmp_path: Path) -> None:
        """Download is skipped when the file already exists locally."""
        local_dir = tmp_path / "decompositions" / "test"
        local_dir.mkdir(parents=True)
        (local_dir / "part-0.parquet").write_text("cached")

        with patch(MOCK_BOTO3) as mock_b3:
            client = MagicMock()
            mock_b3.client.return_value = client
            client.get_paginator.return_value = self._mock_paginator(["decompositions/survey=test/part-0.parquet"])
            paths = _download_decompositions("test", "development", str(tmp_path), False)

        assert len(paths) == 1
        client.download_file.assert_not_called()

    def test_force(self, tmp_path: Path) -> None:
        """Force flag triggers re-download even when cached locally."""
        local_dir = tmp_path / "decompositions" / "test"
        local_dir.mkdir(parents=True)
        (local_dir / "part-0.parquet").write_text("cached")

        with patch(MOCK_BOTO3) as mock_b3:
            client = MagicMock()
            mock_b3.client.return_value = client
            client.get_paginator.return_value = self._mock_paginator(["decompositions/survey=test/part-0.parquet"])
            paths = _download_decompositions("test", "development", str(tmp_path), True)

        assert len(paths) == 1
        client.download_file.assert_called_once()

    def test_pagination(self, tmp_path: Path) -> None:
        """Multiple S3 pages are iterated and all files are downloaded."""
        page1 = {"Contents": [{"Key": "decompositions/survey=test/part-0.parquet"}]}
        page2 = {"Contents": [{"Key": "decompositions/survey=test/part-1.parquet"}]}

        with patch(MOCK_BOTO3) as mock_b3:
            client = MagicMock()
            mock_b3.client.return_value = client
            paginator = MagicMock()
            paginator.paginate.return_value = [page1, page2]
            client.get_paginator.return_value = paginator

            paths = _download_decompositions("test", "development", str(tmp_path), False)

        assert len(paths) == 2
        assert client.download_file.call_count == 2

    def test_empty_prefix(self, tmp_path: Path) -> None:
        """Empty S3 listing returns an empty list of paths."""
        with patch(MOCK_BOTO3) as mock_b3:
            client = MagicMock()
            mock_b3.client.return_value = client
            paginator = MagicMock()
            paginator.paginate.return_value = [{}]
            client.get_paginator.return_value = paginator

            paths = _download_decompositions("test", "development", str(tmp_path), False)

        assert not paths

    def test_non_parquet_files_skipped(self, tmp_path: Path) -> None:
        """Non-parquet files in the S3 listing are filtered out."""
        with patch(MOCK_BOTO3) as mock_b3:
            client = MagicMock()
            mock_b3.client.return_value = client
            client.get_paginator.return_value = self._mock_paginator(
                [
                    "decompositions/survey=test/part-0.parquet",
                    "decompositions/survey=test/_SUCCESS",
                ]
            )
            paths = _download_decompositions("test", "development", str(tmp_path), False)

        assert len(paths) == 1


# ---------------------------------------------------------------------------
# TestPanels
# ---------------------------------------------------------------------------


class TestPanels:
    """Tests that each panel renders without error."""

    def test_velocity_rgb(
        self,
        sample_data: DecompositionData,
        sample_velocity: npt.NDArray[np.float64],
        sample_extent: tuple[float, float, float, float],
    ) -> None:
        """Velocity RGB panel renders without error."""
        _, ax = plt.subplots()
        _panel_velocity_rgb(ax, sample_data, sample_velocity, 2, 2, sample_extent)
        plt.close()

    def test_complexity(
        self,
        sample_data: DecompositionData,
        sample_extent: tuple[float, float, float, float],
    ) -> None:
        """Complexity panel renders without error."""
        fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 0.04]})
        _panel_complexity(ax, sample_data, 2, 2, sample_extent, cax)
        plt.close(fig)

    def test_dominant_velocity(
        self,
        sample_data: DecompositionData,
        sample_velocity: npt.NDArray[np.float64],
        sample_extent: tuple[float, float, float, float],
    ) -> None:
        """Dominant velocity panel renders without error."""
        fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 0.04]})
        _panel_dominant_velocity(ax, sample_data, sample_velocity, 2, 2, sample_extent, cax)
        plt.close(fig)

    def test_bivariate(
        self,
        sample_data: DecompositionData,
        sample_velocity: npt.NDArray[np.float64],
        sample_extent: tuple[float, float, float, float],
    ) -> None:
        """Bivariate panel renders without error."""
        _, ax = plt.subplots()
        _panel_bivariate(ax, sample_data, sample_velocity, 2, 2, sample_extent)
        plt.close()
