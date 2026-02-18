"""Slow worker Lambda: re-processes timed-out spectra and patches parquet in place."""

from __future__ import annotations

import json
import os
import uuid
from typing import TypedDict

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from aws_lambda_powertools.utilities.data_classes import SQSEvent
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from aws_lambda_powertools.utilities.typing import LambdaContext

from phspectra import fit_gaussians

s3 = boto3.client("s3")

BUCKET = os.environ["BUCKET_NAME"]

_FLOAT_KEYS = {"beta", "min_persistence", "snr_min", "mf_snr_min", "f_sep", "neg_thresh"}
_INT_KEYS = {"max_refine_iter"}
_BOOL_KEYS = {"refine"}


class _FitKwargs(TypedDict, total=False):
    """Typed keyword arguments for ``fit_gaussians``."""

    beta: float
    min_persistence: float
    refine: bool
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
    kwargs = _FitKwargs()
    for key, raw in params.items():
        val = str(raw)
        if key in _FLOAT_KEYS:
            kwargs[key] = float(val)  # type: ignore[literal-required]
        elif key in _INT_KEYS:
            kwargs[key] = int(float(val))  # type: ignore[literal-required]
        elif key in _BOOL_KEYS:
            kwargs[key] = val.lower() not in ("false", "0", "")  # type: ignore[literal-required]
    return kwargs


def handler(event: dict[str, object], context: LambdaContext) -> dict[str, object]:
    """Re-process timed-out spectra and patch the existing parquet file.

    Parameters
    ----------
    event : dict[str, object]
        Raw SQS event (``batch_size=1``).
    context : LambdaContext
        Lambda runtime context (unused).

    Returns
    -------
    dict[str, object]
        Response with ``statusCode`` 200 and the patched S3 key.
    """
    del context  # unused
    parsed = SQSEvent(event)
    record: SQSRecord = next(parsed.records)
    msg = json.loads(record.body)

    chunk_key: str = msg["chunk_key"]
    output_key: str = msg["output_key"]
    params: dict[str, object] = msg.get("params", {})
    failed_indices: list[int] = msg["failed_indices"]

    # Download chunk .npz
    local_chunk = f"/tmp/{uuid.uuid4().hex}.npz"
    s3.download_file(BUCKET, chunk_key, local_chunk)
    data = np.load(local_chunk)
    os.remove(local_chunk)

    spectra = data["spectra"]

    # Download existing parquet
    local_parquet = f"/tmp/{uuid.uuid4().hex}.parquet"
    s3.download_file(BUCKET, output_key, local_parquet)
    table = pq.read_table(local_parquet)

    # Re-process each failed spectrum (no timeout)
    fit_kwargs = _build_fit_kwargs(params)

    n_components = table.column("n_components").to_pylist()
    amplitudes = table.column("component_amplitudes").to_pylist()
    means = table.column("component_means").to_pylist()
    stddevs = table.column("component_stddevs").to_pylist()

    for idx in failed_indices:
        components = fit_gaussians(spectra[idx], **fit_kwargs)
        n_components[idx] = len(components)
        amplitudes[idx] = [c.amplitude for c in components]
        means[idx] = [c.mean for c in components]
        stddevs[idx] = [c.stddev for c in components]

    # Rebuild table with patched columns
    patched = (
        table.set_column(
            table.schema.get_field_index("n_components"),
            "n_components",
            pa.array(n_components, type=pa.int32()),
        )
        .set_column(
            table.schema.get_field_index("component_amplitudes"),
            "component_amplitudes",
            pa.array(amplitudes, type=pa.list_(pa.float64())),
        )
        .set_column(
            table.schema.get_field_index("component_means"),
            "component_means",
            pa.array(means, type=pa.list_(pa.float64())),
        )
        .set_column(
            table.schema.get_field_index("component_stddevs"),
            "component_stddevs",
            pa.array(stddevs, type=pa.list_(pa.float64())),
        )
    )

    # Overwrite parquet at the same S3 key
    pq.write_table(patched, local_parquet, compression="snappy")
    s3.upload_file(local_parquet, BUCKET, output_key)
    os.remove(local_parquet)

    return {
        "statusCode": 200,
        "body": {"output_key": output_key, "patched_indices": failed_indices},
    }
