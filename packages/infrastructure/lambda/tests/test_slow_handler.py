"""Tests for the slow worker Lambda handler."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

# -- helpers -----------------------------------------------------------------

SLOW_MSG: dict[str, Any] = {
    "chunk_key": "chunks/run/chunk-0000000.npz",
    "output_key": "decompositions/survey=grs/chunk-0000000.parquet",
    "survey": "grs",
    "run_id": "run-001",
    "params": {"beta": 5.0},
    "failed_indices": [1],
}


@dataclass(frozen=True, slots=True)
class FakeGaussianComponent:
    """Mimics ``GaussianComponent`` without importing phspectra."""

    amplitude: float
    mean: float
    stddev: float


def _make_chunk(n_spectra: int, n_channels: int = 64) -> dict[str, npt.NDArray[Any]]:
    """Build a dict that mimics ``np.load(...)`` for a ``.npz`` chunk."""
    return {
        "spectra": np.random.default_rng(42).standard_normal((n_spectra, n_channels)),
        "x": np.arange(n_spectra, dtype=np.int32),
        "y": np.arange(n_spectra, dtype=np.int32),
    }


def _make_sentinel_parquet(path: str, n_spectra: int = 3, sentinel_idx: int = 1) -> None:
    """Write a parquet file with a sentinel row (n_components = -1) at ``sentinel_idx``."""
    rows = []
    for i in range(n_spectra):
        if i == sentinel_idx:
            rows.append(
                {
                    "x": i,
                    "y": i,
                    "beta": 5.0,
                    "rms": 0.1,
                    "min_persistence": 0.5,
                    "n_components": -1,
                    "component_amplitudes": [],
                    "component_means": [],
                    "component_stddevs": [],
                }
            )
        else:
            rows.append(
                {
                    "x": i,
                    "y": i,
                    "beta": 5.0,
                    "rms": 0.1,
                    "min_persistence": 0.5,
                    "n_components": 1,
                    "component_amplitudes": [1.0],
                    "component_means": [10.0],
                    "component_stddevs": [2.0],
                }
            )

    table = pa.table(
        {
            "x": pa.array([r["x"] for r in rows], type=pa.int32()),
            "y": pa.array([r["y"] for r in rows], type=pa.int32()),
            "beta": pa.array([r["beta"] for r in rows], type=pa.float64()),
            "rms": pa.array([r["rms"] for r in rows], type=pa.float64()),
            "min_persistence": pa.array([r["min_persistence"] for r in rows], type=pa.float64()),
            "n_components": pa.array([r["n_components"] for r in rows], type=pa.int32()),
            "component_amplitudes": pa.array(
                [r["component_amplitudes"] for r in rows],
                type=pa.list_(pa.float64()),
            ),
            "component_means": pa.array(
                [r["component_means"] for r in rows],
                type=pa.list_(pa.float64()),
            ),
            "component_stddevs": pa.array(
                [r["component_stddevs"] for r in rows],
                type=pa.list_(pa.float64()),
            ),
        }
    )
    pq.write_table(table, path, compression="snappy")


def _make_fake_download(chunk: dict[str, npt.NDArray[Any]], parquet_path: str) -> Callable[[str, str, str], None]:
    """Return a side_effect for ``s3.download_file`` that writes local files."""

    def _download(_bucket: str, key: str, local: str) -> None:
        if key.endswith(".npz"):
            np.savez(local, **chunk)  # type: ignore[arg-type]
        else:
            shutil.copy(parquet_path, local)

    return _download


# -- tests -------------------------------------------------------------------


def test_patches_sentinel_rows(slow_worker: Any, sqs_event: Any, lambda_context: MagicMock, tmp_path: Any) -> None:
    """Sentinel row (n_components=-1) is replaced with real fit results."""
    event = sqs_event(SLOW_MSG)
    chunk = _make_chunk(3)
    parquet_path = str(tmp_path / "input.parquet")
    _make_sentinel_parquet(parquet_path, n_spectra=3, sentinel_idx=1)

    refit_components = [
        FakeGaussianComponent(amplitude=2.0, mean=15.0, stddev=3.0),
        FakeGaussianComponent(amplitude=0.5, mean=25.0, stddev=1.5),
    ]
    uploaded: dict[str, str] = {}

    def _fake_upload(local: str, _bucket: str, key: str) -> None:
        uploaded[key] = str(tmp_path / "uploaded.parquet")
        shutil.copy(local, uploaded[key])

    with (
        patch.object(slow_worker.s3, "download_file", side_effect=_make_fake_download(chunk, parquet_path)),
        patch.object(slow_worker.s3, "upload_file", side_effect=_fake_upload),
        patch.object(slow_worker, "fit_gaussians", return_value=refit_components),
    ):
        slow_worker.handler(event, lambda_context)

    result = pq.read_table(uploaded[SLOW_MSG["output_key"]])
    n_comp = result.column("n_components").to_pylist()
    assert n_comp[1] == 2
    assert n_comp[1] >= 0

    amps = result.column("component_amplitudes").to_pylist()
    assert amps[1] == [2.0, 0.5]


def test_preserves_completed_rows(slow_worker: Any, sqs_event: Any, lambda_context: MagicMock, tmp_path: Any) -> None:
    """Rows that are not sentinel remain unchanged after patching."""
    event = sqs_event(SLOW_MSG)
    chunk = _make_chunk(3)
    parquet_path = str(tmp_path / "input.parquet")
    _make_sentinel_parquet(parquet_path, n_spectra=3, sentinel_idx=1)

    refit_components = [FakeGaussianComponent(amplitude=2.0, mean=15.0, stddev=3.0)]
    uploaded: dict[str, str] = {}

    def _fake_upload(local: str, _bucket: str, key: str) -> None:
        uploaded[key] = str(tmp_path / "uploaded.parquet")
        shutil.copy(local, uploaded[key])

    with (
        patch.object(slow_worker.s3, "download_file", side_effect=_make_fake_download(chunk, parquet_path)),
        patch.object(slow_worker.s3, "upload_file", side_effect=_fake_upload),
        patch.object(slow_worker, "fit_gaussians", return_value=refit_components),
    ):
        slow_worker.handler(event, lambda_context)

    result = pq.read_table(uploaded[SLOW_MSG["output_key"]])

    # Non-sentinel rows at index 0 and 2 should be unchanged
    n_comp = result.column("n_components").to_pylist()
    assert n_comp[0] == 1
    assert n_comp[2] == 1

    amps = result.column("component_amplitudes").to_pylist()
    assert amps[0] == [1.0]
    assert amps[2] == [1.0]


def test_overwrites_same_s3_key(slow_worker: Any, sqs_event: Any, lambda_context: MagicMock, tmp_path: Any) -> None:
    """``s3.upload_file`` is called with the same ``output_key``."""
    event = sqs_event(SLOW_MSG)
    chunk = _make_chunk(3)
    parquet_path = str(tmp_path / "input.parquet")
    _make_sentinel_parquet(parquet_path, n_spectra=3, sentinel_idx=1)

    with (
        patch.object(slow_worker.s3, "download_file", side_effect=_make_fake_download(chunk, parquet_path)),
        patch.object(slow_worker.s3, "upload_file") as mock_upload,
        patch.object(slow_worker, "fit_gaussians", return_value=[]),
    ):
        slow_worker.handler(event, lambda_context)

    mock_upload.assert_called_once()
    uploaded_key = mock_upload.call_args[0][2]
    assert uploaded_key == SLOW_MSG["output_key"]
