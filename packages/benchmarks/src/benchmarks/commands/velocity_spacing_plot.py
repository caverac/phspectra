"""``benchmarks velocity-spacing-plot`` -- velocity spacing distribution.

Visualises the distribution of velocity separations between adjacent
fitted Gaussian components within each spectrum across selected GRS
tiles.  Supports Section 3.3 of the pre-print (survey-scale application).
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from benchmarks._console import console
from benchmarks._constants import CACHE_DIR, NOISE_SIGMA
from benchmarks._plotting import configure_axes, docs_figure
from benchmarks.commands.survey_map import (
    DecompositionData,
    _download_decompositions,
    _load_parquet_data,
    _table_to_arrays,
    _velocity_axis,
)
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

TILES = ["grs-26", "grs-28", "grs-30", "grs-32", "grs-34"]

AMP_THRESHOLD = 3 * NOISE_SIGMA
BIN_WIDTH = 0.5  # km/s


def _survey_from_grs_filename(filename: str) -> str:
    """Derive a survey name from a GRS FITS filename.

    Strips the ``-cube`` suffix (if present), lowercases the stem.
    """
    stem = Path(filename).stem.lower()
    if stem.endswith("-cube"):
        stem = stem[: -len("-cube")]
    return stem


def _read_first_header(input_dir: str) -> fits.Header:
    """Read the FITS header from the first matching tile."""
    for path in sorted(Path(input_dir).glob("*.fits")):
        survey = _survey_from_grs_filename(path.name)
        if survey in TILES:
            with fits.open(str(path)) as hdul:
                return hdul[0].header.copy()  # pylint: disable=no-member
    msg = f"No matching GRS tiles found in {input_dir}"
    raise FileNotFoundError(msg)


def _load_all_tiles(
    input_dir: str,
    environment: str,
    cache_dir: str,
    force: bool,
) -> DecompositionData:
    """Download and concatenate decomposition data for all tiles."""
    all_x: list[npt.NDArray[np.int32]] = []
    all_y: list[npt.NDArray[np.int32]] = []
    all_ncomp: list[npt.NDArray[np.int32]] = []
    all_amps: list[list[float]] = []
    all_means: list[list[float]] = []
    all_stddevs: list[list[float]] = []

    found_tiles = {_survey_from_grs_filename(p.name) for p in Path(input_dir).glob("*.fits")}

    for tile_name in TILES:
        if tile_name not in found_tiles:
            console.print(f"  Tile {tile_name} not in input dir, skipping.", style="yellow")
            continue

        console.print(f"  Tile [bold]{tile_name}[/bold] ...", style="cyan")
        paths = _download_decompositions(tile_name, environment, cache_dir, force)
        if not paths:
            console.print(f"    No data for {tile_name}, skipping.", style="yellow")
            continue

        table = _load_parquet_data(paths)
        data = _table_to_arrays(table)

        all_x.append(data.x)
        all_y.append(data.y)
        all_ncomp.append(data.n_components)
        all_amps.extend(data.component_amplitudes)
        all_means.extend(data.component_means)
        all_stddevs.extend(data.component_stddevs)

    return DecompositionData(
        x=np.concatenate(all_x) if all_x else np.array([], dtype=np.int32),
        y=np.concatenate(all_y) if all_y else np.array([], dtype=np.int32),
        n_components=np.concatenate(all_ncomp) if all_ncomp else np.array([], dtype=np.int32),
        component_amplitudes=all_amps,
        component_means=all_means,
        component_stddevs=all_stddevs,
    )


def _compute_spacings(
    data: DecompositionData,
    velocity: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Compute adjacent velocity spacings for multi-component spectra.

    Parameters
    ----------
    data:
        Decomposition results.
    velocity:
        Velocity axis in m/s (one entry per channel).

    Returns
    -------
    spacings:
        All adjacent velocity separations in km/s.
    ncomp_of_spacing:
        Number of significant components in the parent spectrum for
        each spacing entry.
    """
    spacings: list[float] = []
    ncomp_tags: list[int] = []
    n_vel = len(velocity)

    for i in range(len(data.x)):
        amps = data.component_amplitudes[i]
        means = data.component_means[i]

        # Filter by amplitude threshold
        keep = [j for j, a in enumerate(amps) if a >= AMP_THRESHOLD]
        if len(keep) < 2:
            continue

        # Convert channel means to velocity in km/s
        vels = []
        for j in keep:
            ch = int(np.clip(round(means[j]), 0, n_vel - 1))
            vels.append(velocity[ch] / 1000.0)

        vels_sorted = sorted(vels)
        n_sig = len(vels_sorted)

        for k in range(n_sig - 1):
            dv = vels_sorted[k + 1] - vels_sorted[k]
            spacings.append(dv)
            ncomp_tags.append(n_sig)

    return (
        np.array(spacings, dtype=np.float64),
        np.array(ncomp_tags, dtype=np.int32),
    )


@docs_figure("velocity-spacing-plot.png")
def _build_figure(
    spacings: npt.NDArray[np.float64],
    ncomp_tags: npt.NDArray[np.int32],
) -> Figure:
    """Build the velocity-spacing histogram figure."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.95)

    v_max = float(np.percentile(spacings, 99.5)) if len(spacings) else 40.0

    # Each series gets a different number of bins and linestyle
    groups: list[tuple[npt.NDArray[np.bool_], str, str, int]] = [
        (np.ones(len(spacings), dtype=bool), f"All ($n = {len(spacings):,}$)", "-", 30),
        (ncomp_tags == 2, f"$N = 2$ ($n = {int((ncomp_tags == 2).sum()):,}$)", "--", 31),
        (
            (ncomp_tags >= 3) & (ncomp_tags <= 4),
            f"$N = 3$-$4$ ($n = {int(((ncomp_tags >= 3) & (ncomp_tags <= 4)).sum()):,}$)",
            "-.",
            32,
        ),
        (
            ncomp_tags >= 5,
            f"$N \\geq 5$ ($n = {int((ncomp_tags >= 5).sum()):,}$)",
            ":",
            33,
        ),
    ]

    legend_handles: list[Patch] = []
    for mask, label, ls, n_bins in groups:
        subset = spacings[mask]
        if len(subset) == 0:
            continue
        bins = np.linspace(0, v_max, n_bins + 1)
        counts, edges = np.histogram(subset, bins=bins, density=True)
        ax.stairs(counts, edges, color="k", linewidth=1.2, linestyle=ls)
        legend_handles.append(
            Patch(facecolor="none", edgecolor="k", linewidth=1.2, linestyle=ls, label=label),
        )

    ax.set_xlabel(r"$\Delta v$ (km s$^{-1}$)")
    ax.set_ylabel("Normalised density")
    ax.set_xlim(0, v_max)
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    configure_axes(ax)

    return fig


@click.command("velocity-spacing-plot")
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing GRS FITS tiles.",
)
@click.option(
    "--environment",
    default="development",
    show_default=True,
    help="AWS environment.",
)
@click.option("--cache-dir", default=CACHE_DIR, show_default=True, help="Cache directory.")
@click.option("--force", is_flag=True, help="Re-download even if cached.")
def velocity_spacing_plot(
    input_dir: str,
    environment: str,
    cache_dir: str,
    force: bool,
) -> None:
    """Histogram of velocity spacings between adjacent Gaussian components."""
    console.print("Reading tile headers ...", style="bold cyan")
    header = _read_first_header(input_dir)
    velocity = _velocity_axis(header)

    console.print("Downloading decomposition data ...", style="bold cyan")
    data = _load_all_tiles(input_dir, environment, cache_dir, force)
    if len(data.x) == 0:
        console.print("No decomposition data found. Aborting.", style="bold red")
        return

    console.print("Computing velocity spacings ...", style="bold cyan")
    spacings, ncomp_tags = _compute_spacings(data, velocity)
    console.print(f"  {len(spacings):,} spacings from {len(data.x):,} spectra.", style="green")

    console.print("Building figure ...", style="bold cyan")
    _build_figure(spacings, ncomp_tags)
    console.print("\nDone.", style="bold green")
