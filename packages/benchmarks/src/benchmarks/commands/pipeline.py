"""``benchmarks pipeline`` -- upload FITS to S3 and monitor DynamoDB progress.

Uploads a FITS file (or manifest) to S3, waits for the splitter Lambda to
create a run record, and polls DynamoDB with a rich progress bar until all
worker jobs complete.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import click
from benchmarks._console import console
from botocore.exceptions import ClientError
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


def _bucket_name(env: str) -> str:
    """Return the S3 bucket name for the given environment."""
    return f"phspectra-{env}-data"


def _table_name(env: str) -> str:
    """Return the DynamoDB table name for the given environment."""
    del env  # table name is not environment-prefixed
    return "phspectra-runs"


def _survey_from_path(path: str) -> str:
    """Derive a lowercase survey name from a file path stem."""
    return Path(path).stem.lower()


def _s3_key_exists(s3_client: Any, bucket: str, key: str) -> bool:
    """Return True if an S3 object exists at the given key."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def _upload_fits(s3_client: Any, bucket: str, local_path: str, cube_key: str) -> None:
    """Upload a FITS file to S3, skipping if already present."""
    if _s3_key_exists(s3_client, bucket, cube_key):
        console.print(
            f"Cube already exists at [blue]s3://{bucket}/{cube_key}[/blue], skipping upload.",
            style="yellow",
        )
        return
    console.print(f"Uploading FITS to [blue]s3://{bucket}/{cube_key}[/blue]")
    s3_client.upload_file(local_path, bucket, cube_key)
    console.print("  Upload complete.", style="green")


ALLOWED_PARAMS = {
    "beta",
    "max_refine_iter",
    "snr_min",
    "mf_snr_min",
    "f_sep",
    "neg_thresh",
}


def _parse_params(raw: tuple[str, ...]) -> dict[str, object]:
    """Parse ``--param key=value`` strings into a dict, JSON-decoding values."""
    params: dict[str, object] = {}
    for item in raw:
        if "=" not in item:
            raise click.UsageError(f"Invalid --param format (expected key=value): {item!r}")
        key, raw_value = item.split("=", 1)
        if key not in ALLOWED_PARAMS:
            raise click.UsageError(f"Unknown param {key!r}. Allowed: {', '.join(sorted(ALLOWED_PARAMS))}")
        params[key] = json.loads(raw_value)
    return params


def _upload_manifest(
    s3_client: Any,
    bucket: str,
    cube_key: str,
    survey: str,
    params: dict[str, object],
) -> str:
    """Build and upload a manifest JSON to S3. Returns the manifest key."""
    manifest_key = f"manifests/{uuid.uuid4()}.json"
    manifest: dict[str, object] = {"cube_key": cube_key, "survey": survey}
    if params:
        manifest["params"] = params
    body = json.dumps(manifest)
    console.print(f"Uploading manifest to [blue]s3://{bucket}/{manifest_key}[/blue]")
    s3_client.put_object(Bucket=bucket, Key=manifest_key, Body=body)
    console.print("  Manifest uploaded.", style="green")
    return manifest_key


def _discover_run_id(
    dynamodb_client: Any,
    table: str,
    survey: str,
    not_before: str,
    timeout: float = 180,
    interval: float = 2,
) -> str:
    """Query GSI1 for a run matching *survey* created at or after *not_before*.

    Returns the ``run_id`` of the most recent matching item.  Calls
    ``sys.exit(1)`` if no item is found within *timeout* seconds.
    """
    console.print("Waiting for run record in DynamoDB ...", style="bold cyan")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = dynamodb_client.query(
            TableName=table,
            IndexName="GSI1",
            KeyConditionExpression="GSI1_PK = :s AND GSI1_SK >= :t",
            ExpressionAttributeValues={
                ":s": {"S": survey},
                ":t": {"S": not_before},
            },
            ScanIndexForward=False,
            Limit=1,
        )
        items = resp.get("Items", [])
        if items:
            run_id: str = items[0]["PK"]["S"]
            console.print(f"  Discovered run [bold]{run_id}[/bold]", style="green")
            return run_id
        time.sleep(interval)

    console.print("Timed out waiting for run record.", style="bold red")
    sys.exit(1)


def _get_run_item(dynamodb_client: Any, table: str, run_id: str) -> dict[str, Any]:
    """Fetch a single run item from DynamoDB. Exits on missing item."""
    resp = dynamodb_client.get_item(TableName=table, Key={"PK": {"S": run_id}, "SK": {"S": "RUN"}})
    item: dict[str, Any] | None = resp.get("Item")
    if item is None:
        console.print(f"Run [bold]{run_id}[/bold] not found.", style="bold red")
        sys.exit(1)
    return item


def _poll_progress(
    dynamodb_client: Any,
    table: str,
    run_id: str,
    poll_interval: float = 5,
    stall_timeout: float = 1800,
) -> dict[str, Any]:
    """Poll DynamoDB until all jobs are done. Returns the final item.

    If progress (``jobs_completed + jobs_failed``) does not change for
    *stall_timeout* seconds the function prints a warning and exits.
    """
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[failed_text]}"),
        console=console,
    )
    last_done = -1
    last_change = time.monotonic()
    with progress:
        task_id = progress.add_task("Jobs", total=None, failed_text="")
        while True:
            item = _get_run_item(dynamodb_client, table, run_id)
            jobs_total = int(item.get("jobs_total", {}).get("N", "0"))
            completed = int(item.get("jobs_completed", {}).get("N", "0"))
            failed = int(item.get("jobs_failed", {}).get("N", "0"))
            done = completed + failed

            if done != last_done:
                last_done = done
                last_change = time.monotonic()

            progress.update(task_id, total=jobs_total, completed=done)
            if failed:
                progress.update(task_id, failed_text=f"[red]{failed} failed[/red]")
            if 0 < jobs_total <= done:
                break

            if 0 < stall_timeout <= time.monotonic() - last_change:
                console.print(
                    f"\nNo progress for {stall_timeout:.0f}s — possible stall "
                    f"(workers may have timed out). Exiting.",
                    style="bold red",
                )
                sys.exit(1)

            time.sleep(poll_interval)

    return item


@click.command()
@click.argument("fits_file", required=False, type=click.Path(exists=True))
@click.option("--manifest", "manifest_mode", is_flag=True, help="Manifest-only mode.")
@click.option("--cube-key", default=None, help="S3 key for the cube (manifest mode).")
@click.option("--survey", default=None, help="Survey name (default: filename stem).")
@click.option(
    "--param",
    "raw_params",
    multiple=True,
    type=str,
    help="fit_gaussians param as key=value (e.g. --param beta=3.5).",
)
@click.option("--run-id", default=None, help="Resume polling an existing run (skip upload).")
@click.option(
    "--stall-timeout",
    default=1800.0,
    show_default=True,
    type=float,
    help="Seconds without progress before exiting (0 to disable).",
)
@click.option(
    "--environment",
    default="development",
    show_default=True,
    help="Environment name for bucket/table prefix.",
)
@click.option(
    "--poll-interval",
    default=5.0,
    show_default=True,
    type=float,
    help="Seconds between DynamoDB polls.",
)
def pipeline(
    fits_file: str | None,
    manifest_mode: bool,
    cube_key: str | None,
    survey: str | None,
    raw_params: tuple[str, ...],
    run_id: str | None,
    stall_timeout: float,
    environment: str,
    poll_interval: float,
) -> None:
    """Upload FITS to S3 and monitor pipeline progress in DynamoDB."""
    # -- clients --------------------------------------------------------------
    dynamodb_client = boto3.client("dynamodb")
    table = _table_name(environment)

    # -- resume existing run --------------------------------------------------
    if run_id is not None:
        if fits_file is not None or manifest_mode or cube_key is not None:
            raise click.UsageError("--run-id cannot be combined with FITS_FILE, --manifest, or --cube-key.")
        console.print(f"Resuming run [bold]{run_id}[/bold]", style="bold cyan")
        _get_run_item(dynamodb_client, table, run_id)  # validate it exists
        _finish_pipeline(dynamodb_client, table, run_id, poll_interval, stall_timeout)
        return

    # -- validation -----------------------------------------------------------
    if manifest_mode:
        if fits_file is not None:
            raise click.UsageError("FITS_FILE is not allowed with --manifest.")
        if cube_key is None:
            raise click.UsageError("--cube-key is required with --manifest.")
        if survey is None:
            raise click.UsageError("--survey is required with --manifest.")
    else:
        if fits_file is None:
            raise click.UsageError("FITS_FILE is required in direct mode.")

    params = _parse_params(raw_params)

    # -- clients --------------------------------------------------------------
    s3_client = boto3.client("s3")
    bucket = _bucket_name(environment)

    # -- upload ---------------------------------------------------------------
    if not manifest_mode:
        survey = survey or _survey_from_path(fits_file)  # type: ignore[arg-type]
        cube_key = f"cubes/{survey}.fits"
        _upload_fits(s3_client, bucket, fits_file, cube_key)  # type: ignore[arg-type]

    not_before = datetime.now(timezone.utc).isoformat()
    _upload_manifest(s3_client, bucket, cube_key, survey, params)  # type: ignore[arg-type]

    # -- discover run ---------------------------------------------------------
    run_id = _discover_run_id(dynamodb_client, table, survey, not_before)  # type: ignore[arg-type]

    _finish_pipeline(dynamodb_client, table, run_id, poll_interval, stall_timeout)


def _finish_pipeline(
    dynamodb_client: Any,
    table: str,
    run_id: str,
    poll_interval: float,
    stall_timeout: float,
) -> None:
    """Poll progress and print summary."""
    final = _poll_progress(dynamodb_client, table, run_id, poll_interval, stall_timeout)

    completed = int(final.get("jobs_completed", {}).get("N", "0"))
    failed = int(final.get("jobs_failed", {}).get("N", "0"))
    total = int(final.get("jobs_total", {}).get("N", "0"))

    if failed:
        console.print(
            f"\nDone: {completed}/{total} passed, " f"[red]{failed} failed[/red].",
            style="bold yellow",
        )
    else:
        console.print(f"\nDone: {completed}/{total} passed.", style="bold green")
