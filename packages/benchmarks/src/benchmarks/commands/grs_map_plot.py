"""``benchmarks grs-map-plot`` -- multi-tile GRS survey visualisation.

Downloads decomposition results from all GRS tiles, remaps pixel
coordinates to a global grid using CRPIX offsets, and renders the
2x2 survey figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from benchmarks._console import console
from benchmarks._constants import CACHE_DIR
from benchmarks._plotting import docs_figure
from benchmarks.commands.pipeline_grs import _survey_from_grs_filename
from benchmarks.commands.survey_map import (
    DecompositionData,
    _download_decompositions,
    _galactic_extent,
    _load_parquet_data,
    _panel_bivariate,
    _panel_complexity,
    _panel_dominant_velocity,
    _panel_velocity_rgb,
    _table_to_arrays,
    _velocity_axis,
)
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec


@dataclass
class TileInfo:
    """Metadata for a single GRS FITS tile."""

    path: Path
    survey: str
    naxis1: int
    naxis2: int
    naxis3: int
    crpix1: float
    crpix2: float
    header: fits.Header


@dataclass
class GlobalGrid:
    """Global grid computed from all GRS tiles."""

    nx: int
    ny: int
    naxis3: int
    offsets: dict[str, tuple[int, int]]
    header: fits.Header


def _read_tile_infos(input_dir: str) -> list[TileInfo]:
    """Read FITS headers from all tiles in a directory.

    Parameters
    ----------
    input_dir:
        Directory containing ``*.fits`` files.

    Returns
    -------
    list[TileInfo]
        Tile metadata sorted by filename.
    """
    tiles: list[TileInfo] = []
    for path in sorted(Path(input_dir).glob("*.fits")):
        with fits.open(str(path)) as hdul:
            header = hdul[0].header.copy()  # pylint: disable=no-member
        tiles.append(
            TileInfo(
                path=path,
                survey=_survey_from_grs_filename(path.name),
                naxis1=int(header["NAXIS1"]),
                naxis2=int(header["NAXIS2"]),
                naxis3=int(header["NAXIS3"]),
                crpix1=float(header["CRPIX1"]),
                crpix2=float(header["CRPIX2"]),
                header=header,
            )
        )
    return tiles


def _compute_global_grid(tiles: list[TileInfo]) -> GlobalGrid:
    """Compute a global pixel grid from tile CRPIX offsets.

    All GRS tiles share the same CAR projection (CRVAL1=0, CRVAL2=0)
    and pixel scale — only CRPIX1/2 and NAXIS1/2/3 differ.  Coordinate
    remapping is exact integer arithmetic on CRPIX values.

    Parameters
    ----------
    tiles:
        List of tile metadata.

    Returns
    -------
    GlobalGrid
        Global grid dimensions, per-tile offsets, and synthetic header.
    """
    # For each tile, absolute offset of pixel 0 = 0 - (CRPIX - 1)
    x_mins: list[float] = []
    x_maxs: list[float] = []
    y_mins: list[float] = []
    y_maxs: list[float] = []

    for t in tiles:
        x0 = 0 - (t.crpix1 - 1)
        y0 = 0 - (t.crpix2 - 1)
        x_mins.append(x0)
        x_maxs.append(x0 + t.naxis1 - 1)
        y_mins.append(y0)
        y_maxs.append(y0 + t.naxis2 - 1)

    min_x = min(x_mins)
    max_x = max(x_maxs)
    min_y = min(y_mins)
    max_y = max(y_maxs)

    nx = round(max_x - min_x) + 1
    ny = round(max_y - min_y) + 1
    naxis3 = max(t.naxis3 for t in tiles)

    crpix1_global = 1 - min_x
    crpix2_global = 1 - min_y

    # Per-tile pixel offset into global grid
    offsets: dict[str, tuple[int, int]] = {}
    for t in tiles:
        x_off = round(crpix1_global - t.crpix1)
        y_off = round(crpix2_global - t.crpix2)
        offsets[t.survey] = (x_off, y_off)

    # Build a synthetic global header from the first tile
    header = tiles[0].header.copy()
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["NAXIS3"] = naxis3
    header["CRPIX1"] = crpix1_global
    header["CRPIX2"] = crpix2_global

    return GlobalGrid(nx=nx, ny=ny, naxis3=naxis3, offsets=offsets, header=header)


def _load_global_data(
    tiles: list[TileInfo],
    grid: GlobalGrid,
    environment: str,
    cache_dir: str,
    force: bool,
) -> DecompositionData:
    """Download and remap decomposition data from all tiles.

    Parameters
    ----------
    tiles:
        Tile metadata.
    grid:
        Global grid with per-tile offsets.
    environment:
        AWS environment name.
    cache_dir:
        Local cache directory.
    force:
        Re-download even if cached.

    Returns
    -------
    DecompositionData
        Concatenated data with coordinates remapped to the global grid.
    """
    all_x: list[npt.NDArray[np.int32]] = []
    all_y: list[npt.NDArray[np.int32]] = []
    all_ncomp: list[npt.NDArray[np.int32]] = []
    all_amps: list[list[float]] = []
    all_means: list[list[float]] = []
    all_stddevs: list[list[float]] = []

    for tile in tiles:
        console.print(f"  Tile [bold]{tile.survey}[/bold] ...", style="cyan")
        paths = _download_decompositions(tile.survey, environment, cache_dir, force)
        if not paths:
            console.print(f"    No data for {tile.survey}, skipping.", style="yellow")
            continue

        table = _load_parquet_data(paths)
        data = _table_to_arrays(table)

        x_off, y_off = grid.offsets[tile.survey]
        all_x.append(data.x + x_off)
        all_y.append(data.y + y_off)
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


@docs_figure("grs-map-plot.png")
def _build_grs_figure(
    data: DecompositionData,
    velocity: npt.NDArray[np.float64],
    grid: GlobalGrid,
) -> Figure:
    """Construct the 2x2 GRS survey map figure.

    Same layout as ``survey_map._build_figure`` but uses the global
    grid dimensions and a wider aspect ratio for the ~40 deg x ~2 deg
    GRS field.
    """
    nx = grid.nx
    ny = grid.ny
    extent = _galactic_extent(grid.header, nx, ny)

    fig = plt.figure(figsize=(16, 5))
    outer = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1, 1.06],
        left=0.06,
        right=0.97,
        bottom=0.10,
        top=0.97,
        wspace=0.05,
        hspace=0.05,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_c = fig.add_subplot(outer[1, 0], sharex=ax_a)

    def _split_with_cbar(cell: SubplotSpec) -> tuple[Axes, Axes]:
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=cell, width_ratios=[1, 0.04], wspace=0.05)
        return fig.add_subplot(inner[0]), fig.add_subplot(inner[1])

    ax_b, cax_b = _split_with_cbar(outer[0, 1])
    ax_d, cax_d = _split_with_cbar(outer[1, 1])

    ax_b.sharey(ax_a)
    ax_c.sharex(ax_a)
    ax_d.sharex(ax_b)
    ax_d.sharey(ax_c)

    _panel_velocity_rgb(ax_a, data, velocity, nx, ny, extent)
    _panel_complexity(ax_b, data, nx, ny, extent, cax_b)
    _panel_bivariate(ax_c, data, velocity, nx, ny, extent)
    _panel_dominant_velocity(ax_d, data, velocity, nx, ny, extent, cax_d)

    ax_a.tick_params(labelbottom=False)
    ax_b.tick_params(labelbottom=False, labelleft=False)
    ax_d.tick_params(labelleft=False)
    ax_c.set_xlabel("Galactic longitude (deg)")
    ax_d.set_xlabel("Galactic longitude (deg)")
    ax_a.set_ylabel("Galactic latitude (deg)")
    ax_c.set_ylabel("Galactic latitude (deg)")

    return fig


@click.command("grs-map-plot")
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
def grs_map_plot(
    input_dir: str,
    environment: str,
    cache_dir: str,
    force: bool,
) -> None:
    """Generate 2x2 survey visualisation from all GRS tiles."""
    console.print("Reading tile headers ...", style="bold cyan")
    tiles = _read_tile_infos(input_dir)
    if not tiles:
        console.print("No FITS tiles found. Aborting.", style="bold red")
        return

    console.print(f"Found {len(tiles)} tile(s).", style="bold cyan")

    console.print("Computing global grid ...", style="bold cyan")
    grid = _compute_global_grid(tiles)
    console.print(
        f"  Global grid: {grid.nx} x {grid.ny} x {grid.naxis3}",
        style="green",
    )

    console.print("Downloading decomposition data ...", style="bold cyan")
    data = _load_global_data(tiles, grid, environment, cache_dir, force)
    if len(data.x) == 0:
        console.print("No decomposition data found. Aborting.", style="bold red")
        return

    velocity = _velocity_axis(grid.header)

    console.print("Building figure ...", style="bold cyan")
    _build_grs_figure(data, velocity, grid)
    console.print("\nDone.", style="bold green")
