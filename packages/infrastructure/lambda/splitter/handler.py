"""Splitter Lambda: reads FITS cubes or manifests, chunks spectra, fans out to SQS."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import boto3
import numpy as np
from astropy.io import fits
from aws_lambda_powertools.utilities.data_classes import EventBridgeEvent
from aws_lambda_powertools.utilities.typing import LambdaContext

s3 = boto3.client("s3")
sqs = boto3.client("sqs")
dynamodb = boto3.client("dynamodb")

BUCKET = os.environ["BUCKET_NAME"]
QUEUE_URL = os.environ["QUEUE_URL"]
TABLE_NAME = os.environ["TABLE_NAME"]
CHUNK_SIZE = 500
DEFAULT_BETA = 3.8


def handler(event: dict[str, object], context: LambdaContext) -> dict[str, object]:
    """Route an EventBridge S3 notification to the appropriate handler.

    Supports two trigger types: a ``.fits`` cube upload (chunked and fanned
    out with a default beta) or a ``manifests/*.json`` upload (which
    specifies the cube key, survey, and beta sweep values).

    Parameters
    ----------
    event : dict[str, object]
        Raw EventBridge event forwarded by the S3-to-EventBridge rule.
    context : LambdaContext
        Lambda runtime context (unused).

    Returns
    -------
    dict[str, object]
        Response with ``statusCode`` and a ``body`` containing the run
        metadata (run_id, chunk/message counts) or an error message.
    """
    del context  # unused
    parsed = EventBridgeEvent(event)
    detail = parsed.detail
    key: str = detail["object"]["key"]

    if key.startswith("manifests/") and key.endswith(".json"):
        return _handle_manifest(key)

    if key.endswith(".fits"):
        return _handle_fits(key, survey=_survey_from_key(key), beta_values=[DEFAULT_BETA])

    return {"statusCode": 400, "body": f"Unsupported key: {key}"}


def _handle_manifest(key: str) -> dict[str, object]:
    """Parse a JSON manifest from S3 and delegate to ``_handle_fits``.

    The manifest must contain ``cube_key``, ``survey``, and
    ``beta_values`` fields, allowing callers to specify a multi-beta
    sweep over a single FITS cube.

    Parameters
    ----------
    key : str
        S3 key under ``manifests/`` pointing to the JSON manifest.

    Returns
    -------
    dict[str, object]
        Forwarded result from ``_handle_fits``.
    """
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    manifest = json.loads(resp["Body"].read())
    cube_key = manifest["cube_key"]
    survey = manifest["survey"]
    beta_values = [float(b) for b in manifest["beta_values"]]
    return _handle_fits(cube_key, survey=survey, beta_values=beta_values)


def _handle_fits(key: str, *, survey: str, beta_values: list[float]) -> dict[str, object]:
    """Download a 3-D FITS cube, chunk its spectra, and fan out to SQS.

    The cube is reshaped from ``(n_channels, ny, nx)`` into individual
    spectra, split into chunks of ``CHUNK_SIZE``, uploaded as ``.npz``
    files, and one SQS message is sent per ``(chunk, beta)`` pair.

    Parameters
    ----------
    key : str
        S3 key of the ``.fits`` cube.
    survey : str
        Survey identifier carried through to worker messages.
    beta_values : list[float]
        Persistence thresholds (in multiples of RMS) to sweep.

    Returns
    -------
    dict[str, object]
        Response with ``statusCode`` 200 and run metadata.

    Raises
    ------
    ValueError
        If the FITS primary HDU is not a 3-D array.
    """
    run_id = str(uuid.uuid4())

    # Download FITS to /tmp
    local_path = f"/tmp/{os.path.basename(key)}"
    s3.download_file(BUCKET, key, local_path)

    with fits.open(local_path) as hdul:
        data = hdul[0].data  # pylint: disable=no-member

    os.remove(local_path)

    # FITS cubes are typically (n_channels, ny, nx) -- reshape to (n_spectra, n_channels)
    if data.ndim == 3:
        n_channels, ny, nx = data.shape
        spectra = data.reshape(n_channels, -1).T  # (n_spectra, n_channels)
        ys, xs = np.mgrid[0:ny, 0:nx]
        x_coords = xs.ravel()
        y_coords = ys.ravel()
    else:
        raise ValueError(f"Expected 3D FITS cube, got shape {data.shape}")

    # Replace NaN with 0
    spectra = np.nan_to_num(spectra, nan=0.0).astype(np.float64)

    n_spectra = spectra.shape[0]
    chunk_keys: list[str] = []

    for start in range(0, n_spectra, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_spectra)
        chunk_key = f"chunks/{run_id}/chunk-{start:07d}.npz"

        # Save chunk as .npz to /tmp, then upload
        chunk_path = f"/tmp/chunk-{start:07d}.npz"
        np.savez_compressed(
            chunk_path,
            spectra=spectra[start:end],
            x=x_coords[start:end],
            y=y_coords[start:end],
        )
        s3.upload_file(chunk_path, BUCKET, chunk_key)
        os.remove(chunk_path)
        chunk_keys.append(chunk_key)

    # Send one SQS message per (chunk, beta) pair
    messages_sent = 0
    for chunk_key in chunk_keys:
        for beta in beta_values:
            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps(
                    {
                        "chunk_key": chunk_key,
                        "survey": survey,
                        "beta": beta,
                        "run_id": run_id,
                    }
                ),
            )
            messages_sent += 1

    dynamodb.put_item(
        TableName=TABLE_NAME,
        Item={
            "run_id": {"S": run_id},
            "survey": {"S": survey},
            "created_at": {"S": datetime.now(timezone.utc).isoformat()},
            "jobs_total": {"N": str(messages_sent)},
            "jobs_completed": {"N": "0"},
            "jobs_failed": {"N": "0"},
            "beta_values": {"L": [{"N": str(b)} for b in beta_values]},
            "n_spectra": {"N": str(n_spectra)},
            "n_chunks": {"N": str(len(chunk_keys))},
        },
    )

    return {
        "statusCode": 200,
        "body": {
            "run_id": run_id,
            "n_spectra": n_spectra,
            "n_chunks": len(chunk_keys),
            "n_messages": messages_sent,
            "beta_values": beta_values,
        },
    }


def _survey_from_key(key: str) -> str:
    """Derive a survey name from an S3 key by taking the lowercase file stem.

    Parameters
    ----------
    key : str
        S3 object key (e.g. ``uploads/NGC1234.fits``).

    Returns
    -------
    str
        Lowercase filename without extension (e.g. ``ngc1234``).
    """
    basename = os.path.basename(key)
    return os.path.splitext(basename)[0].lower()
