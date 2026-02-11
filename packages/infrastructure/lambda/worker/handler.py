"""Worker Lambda: processes a spectrum chunk and writes Parquet results to S3."""

from __future__ import annotations

import json
import os
import uuid

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from aws_lambda_powertools.utilities.data_classes import SQSEvent
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from aws_lambda_powertools.utilities.typing import LambdaContext

from phspectra import estimate_rms, fit_gaussians

s3 = boto3.client("s3")

BUCKET = os.environ["BUCKET_NAME"]


def handler(event: dict[str, object], context: LambdaContext) -> dict[str, object]:
    """Decompose a chunk of spectra and write Parquet results to S3.

    Expects a single SQS record whose body contains ``chunk_key``,
    ``survey``, and ``beta``.  Downloads the ``.npz`` chunk, runs
    ``estimate_rms`` and ``fit_gaussians`` on every spectrum, and
    uploads a Snappy-compressed Parquet file to
    ``decompositions/survey=<survey>/beta=<beta>/<chunk>.parquet``.

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
    del context  # unused
    parsed = SQSEvent(event)
    record: SQSRecord = next(parsed.records)
    msg = json.loads(record.body)

    chunk_key: str = msg["chunk_key"]
    survey: str = msg["survey"]
    beta: float = float(msg["beta"])

    # Download chunk
    local_chunk = f"/tmp/{uuid.uuid4().hex}.npz"
    s3.download_file(BUCKET, chunk_key, local_chunk)
    data = np.load(local_chunk)
    os.remove(local_chunk)

    spectra = data["spectra"]
    x_coords = data["x"]
    y_coords = data["y"]

    rows: list[dict[str, object]] = []
    for idx, spectrum in enumerate(spectra):
        rms = estimate_rms(spectrum)
        min_persistence = beta * rms
        components = fit_gaussians(spectrum, beta=beta)

        rows.append(
            {
                "x": int(x_coords[idx]),
                "y": int(y_coords[idx]),
                "rms": float(rms),
                "min_persistence": float(min_persistence),
                "n_components": len(components),
                "component_amplitudes": [c.amplitude for c in components],
                "component_means": [c.mean for c in components],
                "component_stddevs": [c.stddev for c in components],
            }
        )

    # Build PyArrow table
    table = pa.table(
        {
            "x": pa.array([r["x"] for r in rows], type=pa.int32()),
            "y": pa.array([r["y"] for r in rows], type=pa.int32()),
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
    beta_str = f"{beta:.1f}"
    chunk_basename = os.path.basename(chunk_key).replace(".npz", ".parquet")
    output_key = f"decompositions/survey={survey}/beta={beta_str}/{chunk_basename}"

    local_parquet = f"/tmp/{uuid.uuid4().hex}.parquet"
    pq.write_table(table, local_parquet, compression="snappy")
    s3.upload_file(local_parquet, BUCKET, output_key)
    os.remove(local_parquet)

    return {
        "statusCode": 200,
        "body": {
            "output_key": output_key,
            "n_spectra": len(rows),
        },
    }
