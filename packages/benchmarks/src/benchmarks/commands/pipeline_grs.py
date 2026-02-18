"""``benchmarks pipeline-grs`` -- batch GRS tile submission and monitoring.

Submits all GRS FITS tiles found in a directory to the Lambda pipeline
individually and monitors progress with a multi-row Rich progress display.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import click
from benchmarks._console import console
from benchmarks.commands.pipeline import (
    _bucket_name,
    _discover_run_id,
    _get_run_item,
    _parse_params,
    _table_name,
    _upload_fits,
    _upload_manifest,
)
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


def _survey_from_grs_filename(filename: str) -> str:
    """Derive a survey name from a GRS FITS filename.

    Strips the ``-cube`` suffix (if present), lowercases the stem.

    Examples
    --------
    >>> _survey_from_grs_filename("grs-15-cube.fits")
    'grs-15'
    >>> _survey_from_grs_filename("GRS-15.fits")
    'grs-15'
    """
    stem = Path(filename).stem.lower()
    if stem.endswith("-cube"):
        stem = stem[: -len("-cube")]
    return stem


def _poll_all_progress(
    dynamodb_client: Any,
    table: str,
    runs: dict[str, str],
    poll_interval: float = 5,
    stall_timeout: float = 1800,
) -> dict[str, dict[str, Any]]:
    """Poll DynamoDB for multiple runs until all complete.

    Parameters
    ----------
    dynamodb_client:
        Boto3 DynamoDB client.
    table:
        DynamoDB table name.
    runs:
        Mapping of survey name to run_id.
    poll_interval:
        Seconds between polls.
    stall_timeout:
        Seconds without any progress before exiting.

    Returns
    -------
    dict[str, dict]
        Final DynamoDB items keyed by survey name.
    """
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[failed_text]}"),
        console=console,
    )

    tasks: dict[str, Any] = {}
    finished: dict[str, dict[str, Any]] = {}
    last_total_done = -1
    last_change = time.monotonic()

    with progress:
        for survey in runs:
            tasks[survey] = progress.add_task(survey, total=None, failed_text="")

        while True:
            total_done = 0
            all_done = True

            for survey, run_id in runs.items():
                if survey in finished:  # pylint: disable=consider-using-get
                    item = finished[survey]
                else:
                    item = _get_run_item(dynamodb_client, table, run_id)

                jobs_total = int(item.get("jobs_total", {}).get("N", "0"))
                completed = int(item.get("jobs_completed", {}).get("N", "0"))
                failed = int(item.get("jobs_failed", {}).get("N", "0"))
                done = completed + failed
                total_done += done

                progress.update(tasks[survey], total=jobs_total, completed=done)
                if failed:
                    progress.update(tasks[survey], failed_text=f"[red]{failed} failed[/red]")

                if 0 < jobs_total <= done:
                    if survey not in finished:
                        finished[survey] = item
                else:
                    all_done = False

            if total_done != last_total_done:
                last_total_done = total_done
                last_change = time.monotonic()

            if all_done and finished:
                break

            if 0 < stall_timeout <= time.monotonic() - last_change:
                console.print(
                    f"\nNo progress for {stall_timeout:.0f}s — possible stall "
                    f"(workers may have timed out). Exiting.",
                    style="bold red",
                )
                sys.exit(1)

            time.sleep(poll_interval)

    return finished


@click.command("pipeline-grs")
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing GRS FITS tiles.",
)
@click.option(
    "--param",
    "raw_params",
    multiple=True,
    type=str,
    help="fit_gaussians param as key=value (e.g. --param beta=3.5).",
)
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
def pipeline_grs(
    input_dir: str,
    raw_params: tuple[str, ...],
    stall_timeout: float,
    environment: str,
    poll_interval: float,
) -> None:
    """Submit all GRS FITS tiles to the pipeline and monitor progress."""
    params = _parse_params(raw_params)

    # -- discover tiles -------------------------------------------------------
    fits_paths = sorted(Path(input_dir).glob("*.fits"))
    if not fits_paths:
        console.print("No FITS files found in input directory.", style="bold red")
        return

    surveys = {p: _survey_from_grs_filename(p.name) for p in fits_paths}
    console.print(
        f"Found {len(fits_paths)} tile(s): " + ", ".join(surveys.values()),
        style="bold cyan",
    )

    # -- clients --------------------------------------------------------------
    s3_client = boto3.client("s3")
    dynamodb_client = boto3.client("dynamodb")
    bucket = _bucket_name(environment)
    table = _table_name(environment)

    # -- upload FITS ----------------------------------------------------------
    cube_keys: dict[str, str] = {}
    for path, survey in surveys.items():
        cube_key = f"cubes/{survey}.fits"
        _upload_fits(s3_client, bucket, str(path), cube_key)
        cube_keys[survey] = cube_key

    # -- upload manifests (triggers splitters) --------------------------------
    not_before = datetime.now(timezone.utc).isoformat()
    for survey, cube_key in cube_keys.items():
        _upload_manifest(s3_client, bucket, cube_key, survey, params)

    # -- discover runs --------------------------------------------------------
    run_ids: dict[str, str] = {}
    for survey in cube_keys:
        run_id = _discover_run_id(dynamodb_client, table, survey, not_before)
        run_ids[survey] = run_id

    # -- poll all runs --------------------------------------------------------
    finals = _poll_all_progress(dynamodb_client, table, run_ids, poll_interval, stall_timeout)

    # -- summary --------------------------------------------------------------
    total_completed = 0
    total_failed = 0
    total_jobs = 0
    for item in finals.values():
        total_completed += int(item.get("jobs_completed", {}).get("N", "0"))
        total_failed += int(item.get("jobs_failed", {}).get("N", "0"))
        total_jobs += int(item.get("jobs_total", {}).get("N", "0"))

    n_tiles = len(finals)
    if total_failed:
        console.print(
            f"\nDone: {total_completed}/{total_jobs} passed across {n_tiles} tile(s), "
            f"[red]{total_failed} failed[/red].",
            style="bold yellow",
        )
    else:
        console.print(
            f"\nDone: {total_completed}/{total_jobs} passed across {n_tiles} tile(s).",
            style="bold green",
        )
