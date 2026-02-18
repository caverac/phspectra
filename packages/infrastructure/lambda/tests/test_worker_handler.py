"""Tests for the worker Lambda handler."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import numpy.typing as npt
import pytest

# pylint: disable=protected-access

# -- helpers -----------------------------------------------------------------

MSG_BASE = {"chunk_key": "chunks/run/chunk-0000000.npz", "survey": "grs", "params": {"beta": 5.0}, "run_id": "run-001"}


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
        patch.object(worker.dynamodb, "put_item"),
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
        patch.object(worker.dynamodb, "put_item"),
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


def test_beta_column_in_parquet(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """Beta value appears as a regular column in the Parquet table."""
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
        patch.object(worker.dynamodb, "put_item"),
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        worker.handler(event, lambda_context)

    table = mock_pq.write_table.call_args[0][0]
    assert table.column("beta").to_pylist() == [5.0]


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
        patch.object(worker.dynamodb, "put_item"),
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
        ("grs", 5.0, "decompositions/survey=grs/"),
        ("vgps", 1.5, "decompositions/survey=vgps/"),
        ("ngc1234", 10.0, "decompositions/survey=ngc1234/"),
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
    """Output key follows ``decompositions/survey=.../`` pattern (beta is a column, not a partition)."""
    event = sqs_event(
        {"chunk_key": "chunks/run/chunk-0000000.npz", "survey": survey, "params": {"beta": beta}, "run_id": "run-001"}
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
        patch.object(worker.dynamodb, "put_item"),
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
        patch.object(worker.dynamodb, "put_item"),
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
        patch.object(worker.dynamodb, "put_item"),
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


def test_build_fit_kwargs_int_cast(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """Int params are cast correctly by ``_build_fit_kwargs``."""
    msg = {**MSG_BASE, "params": {"beta": 5.0, "max_refine_iter": 2}}
    event = sqs_event(msg)
    chunk = _make_chunk(1)

    with (
        patch.object(worker.s3, "download_file"),
        patch.object(worker, "np") as mock_np,
        patch.object(worker, "os") as mock_os,
        patch.object(worker, "estimate_rms", return_value=0.1),
        patch.object(worker, "fit_gaussians", return_value=[]) as mock_fit,
        patch.object(worker, "pq"),
        patch.object(worker.s3, "upload_file"),
        patch.object(worker, "uuid") as mock_uuid,
        patch.object(worker.dynamodb, "update_item"),
        patch.object(worker.dynamodb, "put_item"),
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        worker.handler(event, lambda_context)

    kwargs = mock_fit.call_args[1]
    assert kwargs["max_refine_iter"] == 2
    assert isinstance(kwargs["max_refine_iter"], int)


# -- DynamoDB progress tracking tests ----------------------------------------


def test_jobs_completed_incremented_on_success(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """On success, ``dynamodb.update_item`` increments ``jobs_completed`` and marks chunk COMPLETED."""
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
        patch.object(worker.dynamodb, "put_item") as mock_put,
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        worker.handler(event, lambda_context)

    # put_item for IN_PROGRESS chunk status
    mock_put.assert_called_once()
    put_item = mock_put.call_args[1]["Item"]
    assert put_item["PK"] == {"S": "run-001"}
    assert put_item["SK"] == {"S": "CHUNK#chunk-0000000"}
    assert put_item["status"] == {"S": "IN_PROGRESS"}

    # update_item calls: chunk COMPLETED + run counter
    assert mock_update.call_count == 2
    # Find the run counter call (the one with ADD expression)
    run_counter_call = [c for c in mock_update.call_args_list if "ADD" in c.kwargs.get("UpdateExpression", "")]
    assert len(run_counter_call) == 1
    assert run_counter_call[0] == call(
        TableName="phspectra-development-runs",
        Key={"PK": {"S": "run-001"}, "SK": {"S": "RUN"}},
        UpdateExpression="ADD jobs_completed :one",
        ExpressionAttributeValues={":one": {"N": "1"}},
    )


def test_jobs_failed_incremented_on_error(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """On processing error, ``dynamodb.update_item`` marks chunk FAILED and increments ``jobs_failed``."""
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
        patch.object(worker.dynamodb, "put_item") as mock_put,
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        with pytest.raises(RuntimeError, match="boom"):
            worker.handler(event, lambda_context)

    # put_item for IN_PROGRESS chunk status
    mock_put.assert_called_once()

    # update_item calls: chunk FAILED + run counter
    assert mock_update.call_count == 2
    run_counter_call = [c for c in mock_update.call_args_list if "ADD" in c.kwargs.get("UpdateExpression", "")]
    assert len(run_counter_call) == 1
    assert run_counter_call[0] == call(
        TableName="phspectra-development-runs",
        Key={"PK": {"S": "run-001"}, "SK": {"S": "RUN"}},
        UpdateExpression="ADD jobs_failed :one",
        ExpressionAttributeValues={":one": {"N": "1"}},
    )


# -- Per-spectrum timeout tests -----------------------------------------------


def test_spectrum_timeout_produces_sentinel(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """A timed-out spectrum gets ``n_components = -1`` sentinel row."""
    event = sqs_event(MSG_BASE)
    chunk = _make_chunk(3)

    call_count = 0

    def _fit_side_effect(*_args: object, **_kwargs: object) -> list[object]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise worker._SpectrumTimeout
        return [FakeGaussianComponent(amplitude=1.0, mean=5.0, stddev=1.0)]

    with (
        patch.object(worker.s3, "download_file"),
        patch.object(worker, "np") as mock_np,
        patch.object(worker, "os") as mock_os,
        patch.object(worker, "estimate_rms", return_value=0.1),
        patch.object(worker, "fit_gaussians", side_effect=_fit_side_effect),
        patch.object(worker, "pq") as mock_pq,
        patch.object(worker.s3, "upload_file"),
        patch.object(worker, "uuid") as mock_uuid,
        patch.object(worker.dynamodb, "update_item"),
        patch.object(worker.dynamodb, "put_item"),
        patch.object(worker.sqs, "send_message"),
        patch.object(worker.signal_mod, "signal"),
        patch.object(worker.signal_mod, "alarm"),
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        result = worker.handler(event, lambda_context)

    table = mock_pq.write_table.call_args[0][0]
    n_components = table.column("n_components").to_pylist()
    assert n_components == [1, -1, 1]
    assert result["body"]["failed_indices"] == [1]


def test_slow_queue_message_sent_on_timeout(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """Timed-out spectra trigger an SQS message to the slow queue."""
    event = sqs_event(MSG_BASE)
    chunk = _make_chunk(2)

    call_count = 0

    def _fit_side_effect(*_args: object, **_kwargs: object) -> list[object]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise worker._SpectrumTimeout
        return []

    slow_url = "https://sqs.us-east-1.amazonaws.com/123456789012/slow-queue"

    with (
        patch.object(worker.s3, "download_file"),
        patch.object(worker, "np") as mock_np,
        patch.object(worker, "os") as mock_os,
        patch.object(worker, "estimate_rms", return_value=0.1),
        patch.object(worker, "fit_gaussians", side_effect=_fit_side_effect),
        patch.object(worker, "pq"),
        patch.object(worker.s3, "upload_file"),
        patch.object(worker, "uuid") as mock_uuid,
        patch.object(worker.dynamodb, "update_item"),
        patch.object(worker.dynamodb, "put_item"),
        patch.object(worker.sqs, "send_message") as mock_send,
        patch.object(worker, "SLOW_QUEUE_URL", slow_url),
        patch.object(worker.signal_mod, "signal"),
        patch.object(worker.signal_mod, "alarm"),
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        worker.handler(event, lambda_context)

    mock_send.assert_called_once()
    sent_body = json.loads(mock_send.call_args[1]["MessageBody"])
    assert sent_body["failed_indices"] == [0]
    assert sent_body["chunk_key"] == MSG_BASE["chunk_key"]
    assert mock_send.call_args[1]["QueueUrl"] == slow_url


def test_no_slow_queue_when_all_succeed(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """No SQS message sent when all spectra succeed."""
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
        patch.object(worker.dynamodb, "put_item"),
        patch.object(worker.sqs, "send_message") as mock_send,
        patch.object(worker, "SLOW_QUEUE_URL", "https://some-url"),
        patch.object(worker.signal_mod, "signal"),
        patch.object(worker.signal_mod, "alarm"),
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        worker.handler(event, lambda_context)

    mock_send.assert_not_called()


def test_no_slow_queue_when_env_unset(worker: Any, sqs_event: Any, lambda_context: MagicMock) -> None:
    """No SQS message sent when ``SLOW_QUEUE_URL`` is empty (backward compat)."""
    event = sqs_event(MSG_BASE)
    chunk = _make_chunk(1)

    with (
        patch.object(worker.s3, "download_file"),
        patch.object(worker, "np") as mock_np,
        patch.object(worker, "os") as mock_os,
        patch.object(worker, "estimate_rms", return_value=0.1),
        patch.object(worker, "fit_gaussians", side_effect=worker._SpectrumTimeout),
        patch.object(worker, "pq"),
        patch.object(worker.s3, "upload_file"),
        patch.object(worker, "uuid") as mock_uuid,
        patch.object(worker.dynamodb, "update_item"),
        patch.object(worker.dynamodb, "put_item"),
        patch.object(worker.sqs, "send_message") as mock_send,
        patch.object(worker, "SLOW_QUEUE_URL", ""),
        patch.object(worker.signal_mod, "signal"),
        patch.object(worker.signal_mod, "alarm"),
    ):
        mock_np.load.return_value = chunk
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_uuid.uuid4.return_value = MagicMock(hex="deadbeef")

        worker.handler(event, lambda_context)

    mock_send.assert_not_called()


def test_alarm_handler_raises_spectrum_timeout(worker: Any) -> None:
    """``_alarm_handler`` raises ``_SpectrumTimeout`` when invoked."""
    with pytest.raises(worker._SpectrumTimeout):
        worker._alarm_handler(14, None)
