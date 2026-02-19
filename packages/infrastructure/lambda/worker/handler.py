"""Worker Lambda: processes a spectrum chunk and writes Parquet results to S3."""

from __future__ import annotations

import json
import os
import signal as signal_mod
import time
import uuid
from datetime import datetime, timezone
from typing import TypedDict

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from aws_lambda_powertools.utilities.data_classes import SQSEvent
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from aws_lambda_powertools.utilities.typing import LambdaContext

from phspectra import estimate_rms, fit_gaussians

s3 = boto3.client("s3")
sqs = boto3.client("sqs")
dynamodb = boto3.client("dynamodb")

BUCKET = os.environ["BUCKET_NAME"]
TABLE_NAME = os.environ["TABLE_NAME"]
SPECTRUM_TIMEOUT = int(os.environ.get("SPECTRUM_TIMEOUT", "5"))
SLOW_QUEUE_URL = os.environ.get("SLOW_QUEUE_URL", "")
DEADLINE_MARGIN_MS = 60_000  # bail out 60 s before Lambda timeout


class _SpectrumTimeout(Exception):
    """Raised when a single spectrum exceeds the per-spectrum time budget."""


def _alarm_handler(signum: int, frame: object) -> None:
    raise _SpectrumTimeout


def handler(event: dict[str, object], context: LambdaContext) -> dict[str, object]:
    """Decompose a chunk of spectra and write Parquet results to S3.

    Expects a single SQS record whose body contains ``chunk_key``,
    ``survey``, and an optional ``params`` dict.  Downloads the ``.npz``
    chunk, runs ``estimate_rms`` and ``fit_gaussians`` on every spectrum,
    and uploads a Snappy-compressed Parquet file to
    ``decompositions/survey=<survey>/<chunk>.parquet``.

    Parameters
    ----------
    event : dict[str, object]
        Raw SQS event (``batch_size=1``).
    context : LambdaContext
        Lambda runtime context (unused).

    Returns
    -------
    dict[str, object]
        Response with ``statusCode`` 200 and the output S3 key.
    """
    parsed = SQSEvent(event)
    record: SQSRecord = next(parsed.records)
    msg = json.loads(record.body)

    chunk_key: str = msg["chunk_key"]
    survey: str = msg["survey"]
    params: dict[str, object] = msg.get("params", {})
    run_id: str = msg["run_id"]

    chunk_basename = os.path.basename(chunk_key).replace(".npz", "")
    chunk_sk = f"CHUNK#{chunk_basename}"

    _write_chunk_in_progress(run_id, chunk_sk)
    t0 = time.monotonic()

    try:
        result = _process_chunk(chunk_key, survey, params, context)
    except Exception as exc:
        duration_ms = int((time.monotonic() - t0) * 1000)
        _write_chunk_failed(run_id, chunk_sk, duration_ms, str(exc)[:1024])
        _increment_counter(run_id, "jobs_failed")
        raise

    duration_ms = int((time.monotonic() - t0) * 1000)
    body = result["body"]
    assert isinstance(body, dict)
    output_key: str = body["output_key"]
    n_spectra: int = body["n_spectra"]
    _write_chunk_completed(run_id, chunk_sk, duration_ms, output_key, n_spectra)
    _increment_counter(run_id, "jobs_completed")

    failed_indices = body.get("failed_indices", [])
    if failed_indices and SLOW_QUEUE_URL:
        sqs.send_message(
            QueueUrl=SLOW_QUEUE_URL,
            MessageBody=json.dumps(
                {
                    "chunk_key": chunk_key,
                    "output_key": output_key,
                    "survey": survey,
                    "run_id": run_id,
                    "params": params,
                    "failed_indices": failed_indices,
                }
            ),
        )

    return result


def _increment_counter(run_id: str, attribute: str) -> None:
    """Atomically increment a counter on the runs table.

    Parameters
    ----------
    run_id : str
        Partition key of the run record.
    attribute : str
        Counter attribute to increment (``jobs_completed`` or ``jobs_failed``).
    """
    dynamodb.update_item(
        TableName=TABLE_NAME,
        Key={"PK": {"S": run_id}, "SK": {"S": "RUN"}},
        UpdateExpression=f"ADD {attribute} :one",
        ExpressionAttributeValues={":one": {"N": "1"}},
    )


def _write_chunk_in_progress(run_id: str, chunk_sk: str) -> None:
    """Mark a chunk as IN_PROGRESS in DynamoDB."""
    dynamodb.put_item(
        TableName=TABLE_NAME,
        Item={
            "PK": {"S": run_id},
            "SK": {"S": chunk_sk},
            "status": {"S": "IN_PROGRESS"},
            "started_at": {"S": datetime.now(timezone.utc).isoformat()},
        },
    )


def _write_chunk_completed(run_id: str, chunk_sk: str, duration_ms: int, output_key: str, n_spectra: int) -> None:
    """Mark a chunk as COMPLETED in DynamoDB."""
    dynamodb.update_item(
        TableName=TABLE_NAME,
        Key={"PK": {"S": run_id}, "SK": {"S": chunk_sk}},
        UpdateExpression=(
            "SET #st = :status, completed_at = :completed_at, "
            "duration_ms = :dur, output_key = :okey, n_spectra = :ns"
        ),
        ExpressionAttributeNames={"#st": "status"},
        ExpressionAttributeValues={
            ":status": {"S": "COMPLETED"},
            ":completed_at": {"S": datetime.now(timezone.utc).isoformat()},
            ":dur": {"N": str(duration_ms)},
            ":okey": {"S": output_key},
            ":ns": {"N": str(n_spectra)},
        },
    )


def _write_chunk_failed(run_id: str, chunk_sk: str, duration_ms: int, error: str) -> None:
    """Mark a chunk as FAILED in DynamoDB."""
    dynamodb.update_item(
        TableName=TABLE_NAME,
        Key={"PK": {"S": run_id}, "SK": {"S": chunk_sk}},
        UpdateExpression=("SET #st = :status, completed_at = :completed_at, " "duration_ms = :dur, #err = :error"),
        ExpressionAttributeNames={"#st": "status", "#err": "error"},
        ExpressionAttributeValues={
            ":status": {"S": "FAILED"},
            ":completed_at": {"S": datetime.now(timezone.utc).isoformat()},
            ":dur": {"N": str(duration_ms)},
            ":error": {"S": error},
        },
    )


class _FitKwargs(TypedDict, total=False):
    """Typed keyword arguments for ``fit_gaussians``."""

    beta: float
    max_refine_iter: int
    snr_min: float
    mf_snr_min: float
    f_sep: float
    neg_thresh: float


def _build_fit_kwargs(params: dict[str, object]) -> _FitKwargs:
    """Cast raw JSON params to the types expected by ``fit_gaussians``.

    Parameters
    ----------
    params : dict[str, object]
        Raw keyword arguments from the SQS message.

    Returns
    -------
    _FitKwargs
        Typed keyword arguments safe to unpack into ``fit_gaussians``.
    """
    _FLOAT_KEYS = {"beta", "snr_min", "mf_snr_min", "f_sep", "neg_thresh"}
    _INT_KEYS = {"max_refine_iter"}

    kwargs = _FitKwargs()
    for key, raw in params.items():
        val = str(raw)
        if key in _FLOAT_KEYS:
            kwargs[key] = float(val)  # type: ignore[literal-required]
        elif key in _INT_KEYS:
            kwargs[key] = int(float(val))  # type: ignore[literal-required]
    return kwargs


def _process_chunk(chunk_key: str, survey: str, params: dict[str, object], context: LambdaContext) -> dict[str, object]:
    """Download a chunk, decompose spectra, and upload Parquet results.

    Parameters
    ----------
    chunk_key : str
        S3 key of the ``.npz`` chunk.
    survey : str
        Survey identifier.
    params : dict[str, object]
        ``fit_gaussians`` keyword arguments.
    context : LambdaContext
        Lambda runtime context, used for wall-clock deadline checks.

    Returns
    -------
    dict[str, object]
        Response with ``statusCode`` 200 and the output S3 key.
    """
    # Download chunk
    local_chunk = f"/tmp/{uuid.uuid4().hex}.npz"
    s3.download_file(BUCKET, chunk_key, local_chunk)
    data = np.load(local_chunk)
    os.remove(local_chunk)

    spectra = data["spectra"]
    x_coords = data["x"]
    y_coords = data["y"]

    fit_kwargs = _build_fit_kwargs(params)
    beta = fit_kwargs.get("beta", 3.5)

    signal_mod.signal(signal_mod.SIGALRM, _alarm_handler)
    failed_indices: list[int] = []

    rows: list[dict[str, object]] = []
    for idx, spectrum in enumerate(spectra):
        if context.get_remaining_time_in_millis() < DEADLINE_MARGIN_MS:
            for remaining_idx in range(idx, len(spectra)):
                rows.append(
                    {
                        "x": int(x_coords[remaining_idx]),
                        "y": int(y_coords[remaining_idx]),
                        "beta": float(beta),
                        "rms": 0.0,
                        "min_persistence": 0.0,
                        "n_components": -1,
                        "component_amplitudes": [],
                        "component_means": [],
                        "component_stddevs": [],
                    }
                )
                failed_indices.append(remaining_idx)
            break

        spectrum = np.nan_to_num(spectrum, nan=0.0)
        rms = estimate_rms(spectrum)
        min_persistence = beta * rms

        signal_mod.alarm(SPECTRUM_TIMEOUT)
        try:
            components = fit_gaussians(spectrum, **fit_kwargs)
            signal_mod.alarm(0)
        except _SpectrumTimeout:
            components = []
            failed_indices.append(idx)

        rows.append(
            {
                "x": int(x_coords[idx]),
                "y": int(y_coords[idx]),
                "beta": float(beta),
                "rms": float(rms),
                "min_persistence": float(min_persistence),
                "n_components": -1 if idx in failed_indices else len(components),
                "component_amplitudes": [c.amplitude for c in components],
                "component_means": [c.mean for c in components],
                "component_stddevs": [c.stddev for c in components],
            }
        )

    signal_mod.alarm(0)

    # Build PyArrow table
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

    # Write Parquet to /tmp then upload
    chunk_basename = os.path.basename(chunk_key).replace(".npz", ".parquet")
    output_key = f"decompositions/survey={survey}/{chunk_basename}"

    local_parquet = f"/tmp/{uuid.uuid4().hex}.parquet"
    pq.write_table(table, local_parquet, compression="snappy")
    s3.upload_file(local_parquet, BUCKET, output_key)
    os.remove(local_parquet)

    body: dict[str, object] = {
        "output_key": output_key,
        "n_spectra": len(rows),
    }
    if failed_indices:
        body["failed_indices"] = failed_indices

    return {"statusCode": 200, "body": body}
