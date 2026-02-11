"""Splitter Lambda: reads FITS cubes or manifests, chunks spectra, fans out to SQS."""

from __future__ import annotations

import json
import os
import uuid

import boto3
import numpy as np
from astropy.io import fits

s3 = boto3.client("s3")
sqs = boto3.client("sqs")

BUCKET = os.environ["BUCKET_NAME"]
QUEUE_URL = os.environ["QUEUE_URL"]
CHUNK_SIZE = 500
DEFAULT_BETA = 5.0


def handler(event: dict, context: object) -> dict:
    """Entry point for EventBridge S3 notifications."""
    detail = event.get("detail", {})
    key = detail["object"]["key"]

    if key.startswith("manifests/") and key.endswith(".json"):
        return _handle_manifest(key)
    elif key.endswith(".fits"):
        return _handle_fits(
            key, survey=_survey_from_key(key), beta_values=[DEFAULT_BETA]
        )
    else:
        return {"statusCode": 400, "body": f"Unsupported key: {key}"}


def _handle_manifest(key: str) -> dict:
    """Read a JSON manifest and fan out a beta sweep."""
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    manifest = json.loads(resp["Body"].read())
    cube_key = manifest["cube_key"]
    survey = manifest["survey"]
    beta_values = [float(b) for b in manifest["beta_values"]]
    return _handle_fits(cube_key, survey=survey, beta_values=beta_values)


def _handle_fits(key: str, *, survey: str, beta_values: list[float]) -> dict:
    """Read a FITS cube, chunk spectra, and send SQS messages."""
    run_id = str(uuid.uuid4())

    # Download FITS to /tmp
    local_path = f"/tmp/{os.path.basename(key)}"
    s3.download_file(BUCKET, key, local_path)

    with fits.open(local_path) as hdul:
        data = hdul[0].data  # type: ignore[union-attr]

    os.remove(local_path)

    # FITS cubes are typically (n_channels, ny, nx) — reshape to (n_spectra, n_channels)
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
    """Extract a survey name from the FITS key, defaulting to the filename stem."""
    basename = os.path.basename(key)
    return os.path.splitext(basename)[0].lower()
