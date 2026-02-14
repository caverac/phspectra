"""Tests for the worker Lambda handler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

# -- helpers -----------------------------------------------------------------

MSG_BASE = {"chunk_key": "chunks/run/chunk-0000000.npz", "survey": "grs", "beta": 5.0, "run_id": "run-001"}


@dataclass(frozen=True, slots=True)
class FakeGaussianComponent:
    """Mimics ``GaussianComponent`` without importing phspectra."""

    amplitude: float
    mean: float
    stddev: float


def _make_chunk(n_spectra: int, n_channels: int = 64) -> dict[str, npt.NDArray[np.float64]]:
    """Build a dict that mimics ``np.load(...)`` for a ``.npz`` chunk."""
    return {
        "spectra": np.random.default_rng(42).standard_normal((n_spectra, n_channels)),
        "x": np.arange(n_spectra, dtype=np.int32),
        "y": np.arange(n_spectra, dtype=np.int32),
    }


# -- handler tests ----------------------------------------------------------


def test_no_components(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """Spectrum with 0 detected components -> row has ``n_components=0``."""
    event = sqs_event(MSG_BASE)
    chunk = _make_chunk(1)

    with (
        patch.object(worker.s3, "download_file"),
        patch.object(worker, "np") as mock_np,
        patch.object(worker, "os") as mock_os,
        patch.object(worker, "estimate_rms", return_value=0.1),
        patch.object(worker, "fit_gaussians", return_value=[]),
        patch.object(worker, "pq") as mock_pq,
        patch.object(worker.s3, "upload_file"),
        patch.object(worker, "uuid") as mock_uuid,
        patch.object(worker.dynamodb, "update_item"),
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        result = worker.handler(event, lambda_context)

    assert result["statusCode"] == 200
    assert result["body"]["n_spectra"] == 1

    # Inspect the table passed to write_table
    table = mock_pq.write_table.call_args[0][0]
    assert table.column("n_components").to_pylist() == [0]
    assert table.column("component_amplitudes").to_pylist() == [[]]
    assert table.column("component_means").to_pylist() == [[]]
    assert table.column("component_stddevs").to_pylist() == [[]]


def test_with_components(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """1 spectrum, 2 components -> correct row values."""
    event = sqs_event(MSG_BASE)
    chunk = _make_chunk(1)
    components = [
        FakeGaussianComponent(amplitude=1.5, mean=10.0, stddev=2.0),
        FakeGaussianComponent(amplitude=0.8, mean=30.0, stddev=3.5),
    ]

    with (
        patch.object(worker.s3, "download_file"),
        patch.object(worker, "np") as mock_np,
        patch.object(worker, "os") as mock_os,
        patch.object(worker, "estimate_rms", return_value=0.1),
        patch.object(worker, "fit_gaussians", return_value=components),
        patch.object(worker, "pq") as mock_pq,
        patch.object(worker.s3, "upload_file"),
        patch.object(worker, "uuid") as mock_uuid,
        patch.object(worker.dynamodb, "update_item"),
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        worker.handler(event, lambda_context)

    table = mock_pq.write_table.call_args[0][0]
    assert table.column("n_components").to_pylist() == [2]
    assert table.column("component_amplitudes").to_pylist() == [[1.5, 0.8]]
    assert table.column("component_means").to_pylist() == [[10.0, 30.0]]
    assert table.column("component_stddevs").to_pylist() == [[2.0, 3.5]]


def test_multiple_spectra(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """3 spectra -> all processed, correct count in response."""
    event = sqs_event(MSG_BASE)
    chunk = _make_chunk(3)

    with (
        patch.object(worker.s3, "download_file"),
        patch.object(worker, "np") as mock_np,
        patch.object(worker, "os") as mock_os,
        patch.object(worker, "estimate_rms", return_value=0.1),
        patch.object(worker, "fit_gaussians", return_value=[]),
        patch.object(worker, "pq") as mock_pq,
        patch.object(worker.s3, "upload_file"),
        patch.object(worker, "uuid") as mock_uuid,
        patch.object(worker.dynamodb, "update_item"),
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        result = worker.handler(event, lambda_context)

    assert result["body"]["n_spectra"] == 3
    table = mock_pq.write_table.call_args[0][0]
    assert len(table) == 3


@pytest.mark.parametrize(
    ("survey", "beta", "expected_prefix"),
    [
        ("grs", 5.0, "decompositions/survey=grs/beta=5.00/"),
        ("vgps", 1.5, "decompositions/survey=vgps/beta=1.50/"),
        ("ngc1234", 10.0, "decompositions/survey=ngc1234/beta=10.00/"),
    ],
    ids=["grs-default", "vgps-custom", "ngc-large-beta"],
)
def test_output_key_format(
    worker: Any,
    sqs_event: Any,
    lambda_context: MagicMock,
    survey: str,
    beta: float,
    expected_prefix: str,
) -> None:
    """Output key follows ``decompositions/survey=.../beta=.../`` pattern."""
    event = sqs_event(
        {"chunk_key": "chunks/run/chunk-0000000.npz", "survey": survey, "beta": beta, "run_id": "run-001"}
    )
    chunk = _make_chunk(1)

    with (
        patch.object(worker.s3, "download_file"),
        patch.object(worker, "np") as mock_np,
        patch.object(worker, "os") as mock_os,
        patch.object(worker, "estimate_rms", return_value=0.1),
        patch.object(worker, "fit_gaussians", return_value=[]),
        patch.object(worker, "pq"),
        patch.object(worker.s3, "upload_file"),
        patch.object(worker, "uuid") as mock_uuid,
        patch.object(worker.dynamodb, "update_item"),
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        result = worker.handler(event, lambda_context)

    assert result["body"]["output_key"].startswith(expected_prefix)


def test_s3_upload_called(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """``s3.upload_file`` is called with the correct bucket and output key."""
    event = sqs_event(MSG_BASE)
    chunk = _make_chunk(1)

    with (
        patch.object(worker.s3, "download_file"),
        patch.object(worker, "np") as mock_np,
        patch.object(worker, "os") as mock_os,
        patch.object(worker, "estimate_rms", return_value=0.1),
        patch.object(worker, "fit_gaussians", return_value=[]),
        patch.object(worker, "pq"),
        patch.object(worker.s3, "upload_file") as mock_upload,
        patch.object(worker, "uuid") as mock_uuid,
        patch.object(worker.dynamodb, "update_item"),
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        result = worker.handler(event, lambda_context)

    mock_upload.assert_called_once()
    call_args = mock_upload.call_args
    assert call_args[0][1] == "test-bucket"
    assert call_args[0][2] == result["body"]["output_key"]


def test_response_structure(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """Response contains ``statusCode=200`` and body with ``output_key`` and ``n_spectra``."""
    event = sqs_event(MSG_BASE)
    chunk = _make_chunk(2)

    with (
        patch.object(worker.s3, "download_file"),
        patch.object(worker, "np") as mock_np,
        patch.object(worker, "os") as mock_os,
        patch.object(worker, "estimate_rms", return_value=0.1),
        patch.object(worker, "fit_gaussians", return_value=[]),
        patch.object(worker, "pq"),
        patch.object(worker.s3, "upload_file"),
        patch.object(worker, "uuid") as mock_uuid,
        patch.object(worker.dynamodb, "update_item"),
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        result = worker.handler(event, lambda_context)

    assert result["statusCode"] == 200
    assert "output_key" in result["body"]
    assert "n_spectra" in result["body"]
    assert result["body"]["n_spectra"] == 2


# -- DynamoDB progress tracking tests ----------------------------------------


def test_jobs_completed_incremented_on_success(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """On success, ``dynamodb.update_item`` increments ``jobs_completed``."""
    event = sqs_event(MSG_BASE)
    chunk = _make_chunk(1)

    with (
        patch.object(worker.s3, "download_file"),
        patch.object(worker, "np") as mock_np,
        patch.object(worker, "os") as mock_os,
        patch.object(worker, "estimate_rms", return_value=0.1),
        patch.object(worker, "fit_gaussians", return_value=[]),
        patch.object(worker, "pq"),
        patch.object(worker.s3, "upload_file"),
        patch.object(worker, "uuid") as mock_uuid,
        patch.object(worker.dynamodb, "update_item") as mock_update,
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        worker.handler(event, lambda_context)

    mock_update.assert_called_once_with(
        TableName="phspectra-development-runs",
        Key={"run_id": {"S": "run-001"}},
        UpdateExpression="ADD jobs_completed :one",
        ExpressionAttributeValues={":one": {"N": "1"}},
    )


def test_jobs_failed_incremented_on_error(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """On processing error, ``dynamodb.update_item`` increments ``jobs_failed`` then re-raises."""
    event = sqs_event(MSG_BASE)
    chunk = _make_chunk(1)

    with (
        patch.object(worker.s3, "download_file"),
        patch.object(worker, "np") as mock_np,
        patch.object(worker, "os") as mock_os,
        patch.object(worker, "estimate_rms", side_effect=RuntimeError("boom")),
        patch.object(worker, "fit_gaussians", return_value=[]),
        patch.object(worker, "pq"),
        patch.object(worker.s3, "upload_file"),
        patch.object(worker, "uuid") as mock_uuid,
        patch.object(worker.dynamodb, "update_item") as mock_update,
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        with pytest.raises(RuntimeError, match="boom"):
            worker.handler(event, lambda_context)

    mock_update.assert_called_once_with(
        TableName="phspectra-development-runs",
        Key={"run_id": {"S": "run-001"}},
        UpdateExpression="ADD jobs_failed :one",
        ExpressionAttributeValues={":one": {"N": "1"}},
    )
