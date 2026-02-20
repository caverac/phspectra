"""``benchmarks download`` -- fetch benchmark resources from S3.

Downloads the GRS test-field FITS cube and the GaussPy+ decomposition
catalog from the public phspectra resources bucket.  Files are cached
locally so subsequent commands can run offline.
"""

from __future__ import annotations

import os
import sys
from urllib.error import HTTPError
from urllib.request import urlretrieve

import click
from benchmarks._console import console, err_console
from benchmarks._constants import (
    CACHE_DIR,
    RESOURCE_CATALOG,
    RESOURCE_FITS,
    RESOURCE_PRECOMPUTE_DB,
    RESOURCE_SPECTRA_NPZ,
    RESOURCES_BASE_URL_TEMPLATE,
)


def _download_resource(base_url: str, key: str, dest: str, force: bool) -> None:
    """Download a single resource from S3 if not cached."""
    if os.path.exists(dest) and not force:
        console.print(f"  Using cached [blue]{dest}[/blue]")
        return
    url = f"{base_url}/{key}"
    console.print(f"  Downloading {url} ...")
    try:
        urlretrieve(url, dest)
    except HTTPError as exc:
        err_console.print(f"  ERROR: failed to download {url} ({exc.code} {exc.reason})")
        sys.exit(1)
    console.print(f"  Saved to [blue]{dest}[/blue]")


def _download_optional(base_url: str, key: str, dest: str, force: bool) -> None:
    """Download a resource from S3, skipping if not found."""
    if os.path.exists(dest) and not force:
        console.print(f"  Using cached [blue]{dest}[/blue]")
        return
    url = f"{base_url}/{key}"
    console.print(f"  Downloading {url} ...")
    try:
        urlretrieve(url, dest)
    except HTTPError as exc:
        if exc.code in (403, 404):
            console.print("  Skipped (not available in bucket)", style="yellow")
            return
        err_console.print(f"  ERROR: failed to download {url} ({exc.code} {exc.reason})")
        sys.exit(1)
    console.print(f"  Saved to [blue]{dest}[/blue]")


@click.command()
@click.option("--cache-dir", default=CACHE_DIR, show_default=True, help="Cache directory.")
@click.option("--force", is_flag=True, help="Re-download even if cached.")
@click.option(
    "--environment",
    default="development",
    show_default=True,
    type=click.Choice(["development", "production"]),
    help="S3 environment to download from.",
)
def download(cache_dir: str, force: bool, environment: str) -> None:
    """Download benchmark resources (FITS cube + catalog) from S3."""
    os.makedirs(cache_dir, exist_ok=True)
    base_url = RESOURCES_BASE_URL_TEMPLATE.format(environment=environment)

    fits_path = os.path.join(cache_dir, RESOURCE_FITS)
    catalog_path = os.path.join(cache_dir, RESOURCE_CATALOG)

    if force:
        for p in (fits_path, catalog_path):
            if os.path.exists(p):
                os.remove(p)
                console.print(f"  Removed cached [blue]{p}[/blue]")

    console.print("Step 1: GRS test field FITS", style="bold cyan")
    _download_resource(base_url, RESOURCE_FITS, fits_path, force)

    console.print("\nStep 2: GaussPy+ catalog", style="bold cyan")
    _download_resource(base_url, RESOURCE_CATALOG, catalog_path, force)

    console.print("\nStep 3: Pre-computed results (optional)", style="bold cyan")
    precompute_dir = os.path.join(cache_dir, "compare-docker")
    os.makedirs(precompute_dir, exist_ok=True)
    db_path = os.path.join(precompute_dir, RESOURCE_PRECOMPUTE_DB)
    _download_optional(base_url, RESOURCE_PRECOMPUTE_DB, db_path, force)
    npz_path = os.path.join(precompute_dir, RESOURCE_SPECTRA_NPZ)
    _download_optional(base_url, RESOURCE_SPECTRA_NPZ, npz_path, force)

    console.print(f"\nDone.  All data cached in [blue]{cache_dir}[/blue]", style="bold green")
