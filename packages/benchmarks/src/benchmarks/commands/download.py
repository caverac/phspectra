"""``benchmarks download`` — fetch and cache survey data."""

from __future__ import annotations

import os

import click
from benchmarks._console import console
from benchmarks._constants import CACHE_DIR, GAUSSPY_FITS_URL
from benchmarks._data import ensure_catalog, ensure_fits, fits_bounds


@click.command()
@click.option("--cache-dir", default=CACHE_DIR, show_default=True, help="Cache directory.")
@click.option("--force", is_flag=True, help="Re-download even if cached.")
def download(cache_dir: str, force: bool) -> None:
    """Download and cache FITS cube + VizieR catalog."""
    os.makedirs(cache_dir, exist_ok=True)

    fits_path = os.path.join(cache_dir, "grs-test-field.fits")
    catalog_path = os.path.join(cache_dir, "gausspy-catalog.votable")

    if force:
        for p in (fits_path, catalog_path):
            if os.path.exists(p):
                os.remove(p)
                console.print(f"  Removed cached [blue]{p}[/blue]")

    console.print("Step 1: GRS test field FITS", style="bold cyan")
    header, _ = ensure_fits(GAUSSPY_FITS_URL, fits_path)

    console.print("\nStep 2: GaussPy+ catalog", style="bold cyan")
    bounds = fits_bounds(header)
    catalog = ensure_catalog(*bounds, path=catalog_path)
    console.print(f"  {len(catalog)} component rows")

    console.print(f"\nDone.  All data cached in [blue]{cache_dir}[/blue]", style="bold green")
