"""``benchmarks survey-map-plot`` -- 2x2 survey visualisation from full-field decomposition.

Panel layout
------------
+-------------------------------+--------------------------------+
| (a) Velocity RGB composite    | (b) Topological complexity     |
|                               |                                |
| Three velocity bins mapped    | Number of Gaussian components  |
| to R, G, B from the decomp-   | detected per pixel.  Lights    |
| osed (not raw) Gaussians.     | up at cloud boundaries,        |
| Sharper than moment-0 RGB     | outflows, and shock fronts --  |
| because noise is removed by   | a quantity unique to persist-  |
| the fit.                      | ent-homology decomposition.    |
+-------------------------------+--------------------------------+
| (c) Amplitude--velocity       | (d) Dominant velocity field    |
|     bivariate colormap        |                                |
|                               | Centroid velocity of the       |
| 2-D perceptual colormap where | brightest component per        |
| hue encodes centroid velocity | pixel.  Reveals bulk gas       |
| and luminance encodes peak    | motions hidden by moment-1     |
| amplitude.  Every pixel       | blending when multiple clouds  |
| communicates two physical     | overlap along the LOS.         |
| quantities simultaneously.    |                                |
+-------------------------------+--------------------------------+
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import boto3
import click
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.wcs import WCS
from benchmarks._console import console
from benchmarks._constants import CACHE_DIR, DATA_BUCKET_TEMPLATE
from benchmarks._data import ensure_fits
from benchmarks._plotting import configure_axes, docs_figure
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec


@dataclass
class DecompositionData:
    """Decomposition results loaded from Parquet files."""

    x: npt.NDArray[np.int32]
    y: npt.NDArray[np.int32]
    n_components: npt.NDArray[np.int32]
    component_amplitudes: list[list[float]]
    component_means: list[list[float]]
    component_stddevs: list[list[float]]


def _download_decompositions(
    survey: str,
    environment: str,
    cache_dir: str,
    force: bool,
) -> list[str]:
    """Download Parquet decomposition files from S3.

    Parameters
    ----------
    survey:
        Survey name (S3 partition key).
    environment:
        AWS environment name.
    cache_dir:
        Local cache directory.
    force:
        Re-download even if cached.

    Returns
    -------
    list[str]
        Paths to downloaded Parquet files.
    """
    bucket = DATA_BUCKET_TEMPLATE.format(environment=environment)
    prefix = f"decompositions/survey={survey}/"
    local_dir = os.path.join(cache_dir, "decompositions", survey)
    os.makedirs(local_dir, exist_ok=True)

    client = boto3.client("s3")
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                keys.append(key)

    if not keys:
        console.print(f"  No Parquet files found under s3://{bucket}/{prefix}", style="yellow")
        return []

    paths: list[str] = []
    for key in keys:
        filename = os.path.basename(key)
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path) and not force:
            console.print(f"  Using cached [blue]{local_path}[/blue]")
        else:
            console.print(f"  Downloading s3://{bucket}/{key} ...")
            client.download_file(bucket, key, local_path)
            console.print(f"  Saved to [blue]{local_path}[/blue]")
        paths.append(local_path)

    return paths


def _load_parquet_data(paths: list[str]) -> pa.Table:
    """Load and concatenate Parquet files into a single PyArrow table.

    Parameters
    ----------
    paths:
        Paths to Parquet files.

    Returns
    -------
    pa.Table
        Concatenated table.
    """
    tables = [pq.read_table(p) for p in paths]
    return pa.concat_tables(tables)


def _table_to_arrays(table: pa.Table) -> DecompositionData:
    """Convert a PyArrow table to numpy arrays and lists.

    Parameters
    ----------
    table:
        Table with columns x, y, n_components, component_amplitudes,
        component_means, component_stddevs.

    Returns
    -------
    DecompositionData
        Extracted arrays.
    """
    return DecompositionData(
        x=table.column("x").to_numpy().astype(np.int32),
        y=table.column("y").to_numpy().astype(np.int32),
        n_components=table.column("n_components").to_numpy().astype(np.int32),
        component_amplitudes=table.column("component_amplitudes").to_pylist(),
        component_means=table.column("component_means").to_pylist(),
        component_stddevs=table.column("component_stddevs").to_pylist(),
    )


def _velocity_axis(header: fits.Header) -> npt.NDArray[np.float64]:
    """Build the velocity axis from the FITS header.

    Parameters
    ----------
    header:
        FITS header with CRVAL3, CRPIX3, CDELT3, NAXIS3.

    Returns
    -------
    npt.NDArray[np.float64]
        Velocity in m/s for each channel.
    """
    n_chan = int(header["NAXIS3"])
    crval3 = float(header["CRVAL3"])
    crpix3 = float(header["CRPIX3"])
    cdelt3 = float(header["CDELT3"])
    channels = np.arange(n_chan, dtype=np.float64)
    return crval3 + (channels - (crpix3 - 1)) * cdelt3


def _galactic_extent(
    header: fits.Header,
    nx: int,
    ny: int,
) -> tuple[float, float, float, float]:
    """Compute galactic coordinate extent from a FITS header.

    Parameters
    ----------
    header:
        FITS header with spatial WCS keywords (CTYPE1/2, CRVAL1/2,
        CRPIX1/2, CDELT1/2).
    nx:
        Number of pixels along the x axis.
    ny:
        Number of pixels along the y axis.

    Returns
    -------
    tuple[float, float, float, float]
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.
    """
    wcs = WCS(header, naxis=2)
    lon0, lat0 = wcs.pixel_to_world_values(0, 0)
    lon1, lat1 = wcs.pixel_to_world_values(nx - 1, ny - 1)
    lon_min, lon_max = sorted([float(lon0), float(lon1)])
    lat_min, lat_max = sorted([float(lat0), float(lat1)])
    return lon_min, lon_max, lat_min, lat_max


def _corner_label(ax: Axes, text: str) -> None:
    """Place a bold black-on-yellow corner label."""
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="black",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#f5e6a3",
            "edgecolor": "black",
            "linewidth": 0.8,
        },
    )


def _panel_velocity_rgb(
    ax: Axes,
    data: DecompositionData,
    velocity: npt.NDArray[np.float64],
    nx: int,
    ny: int,
    extent: tuple[float, float, float, float],
) -> None:
    """Panel (a): velocity RGB composite.

    Maps velocity terciles to R (low), G (mid), B (high) channels.
    """
    n_vel = len(velocity)
    rgb = np.zeros((ny, nx, 3), dtype=np.float64)

    v_min, v_max = velocity.min(), velocity.max()
    v_edges = np.linspace(v_min, v_max, 4)

    for i, _ in enumerate(data.x):
        xi, yi = int(data.x[i]), int(data.y[i])
        for amp, mean_ch in zip(data.component_amplitudes[i], data.component_means[i]):
            ch = int(np.clip(round(mean_ch), 0, n_vel - 1))
            v = velocity[ch]
            if v < v_edges[1]:
                rgb[yi, xi, 0] += amp
            elif v < v_edges[2]:
                rgb[yi, xi, 1] += amp
            else:
                rgb[yi, xi, 2] += amp

    for c in range(3):
        channel = rgb[:, :, c]
        p99 = np.percentile(channel[channel > 0], 99) if np.any(channel > 0) else 1.0
        rgb[:, :, c] = np.clip(channel / p99, 0, 1)

    ax.imshow(rgb, origin="lower", aspect="auto", extent=extent)
    _corner_label(ax, "(a)")
    configure_axes(ax)


def _panel_complexity(
    ax: Axes,
    data: DecompositionData,
    nx: int,
    ny: int,
    extent: tuple[float, float, float, float],
    cax: Axes,
) -> None:
    """Panel (b): topological complexity (number of components per pixel)."""
    grid = np.full((ny, nx), np.nan, dtype=np.float64)
    for i, _ in enumerate(data.x):
        xi, yi = int(data.x[i]), int(data.y[i])
        if data.n_components[i] > 0:
            grid[yi, xi] = data.n_components[i]

    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", extent=extent)
    plt.colorbar(im, cax=cax, label="$N_{\\mathrm{components}}$")
    _corner_label(ax, "(b)")
    configure_axes(ax)


def _panel_dominant_velocity(
    ax: Axes,
    data: DecompositionData,
    velocity: npt.NDArray[np.float64],
    nx: int,
    ny: int,
    extent: tuple[float, float, float, float],
    cax: Axes,
) -> None:
    """Panel (d): dominant velocity field (brightest component per pixel)."""
    n_vel = len(velocity)
    grid = np.full((ny, nx), np.nan, dtype=np.float64)

    for i, _ in enumerate(data.x):
        xi, yi = int(data.x[i]), int(data.y[i])
        amps = data.component_amplitudes[i]
        means = data.component_means[i]
        if not amps:
            continue
        idx = int(np.argmax(amps))
        ch = int(np.clip(round(means[idx]), 0, n_vel - 1))
        grid[yi, xi] = velocity[ch] / 1000.0  # m/s -> km/s

    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="coolwarm", extent=extent)
    plt.colorbar(im, cax=cax, label="$v_{\\mathrm{LSR}}$ (km s$^{-1}$)")
    _corner_label(ax, "(d)")
    configure_axes(ax)


def _panel_bivariate(
    ax: Axes,
    data: DecompositionData,
    velocity: npt.NDArray[np.float64],
    nx: int,
    ny: int,
    extent: tuple[float, float, float, float],
) -> None:
    """Panel (c): amplitude-velocity bivariate colormap.

    Hue encodes velocity, value (brightness) encodes amplitude.
    """
    n_vel = len(velocity)
    dom_vel = np.full((ny, nx), np.nan, dtype=np.float64)
    dom_amp = np.full((ny, nx), np.nan, dtype=np.float64)

    for i, _ in enumerate(data.x):
        xi, yi = int(data.x[i]), int(data.y[i])
        amps = data.component_amplitudes[i]
        means = data.component_means[i]
        if not amps:
            continue
        idx = int(np.argmax(amps))
        ch = int(np.clip(round(means[idx]), 0, n_vel - 1))
        dom_vel[yi, xi] = velocity[ch]
        dom_amp[yi, xi] = amps[idx]

    valid = ~np.isnan(dom_vel)
    v_min = float(dom_vel[valid].min()) if np.any(valid) else 0.0
    v_max = float(dom_vel[valid].max()) if np.any(valid) else 1.0
    a_p99 = float(np.percentile(dom_amp[valid], 99)) if np.any(valid) else 1.0

    hsv = np.zeros((ny, nx, 3), dtype=np.float64)
    if v_max > v_min:
        hsv[:, :, 0] = np.where(valid, (dom_vel - v_min) / (v_max - v_min), 0)
    hsv[:, :, 1] = np.where(valid, 1.0, 0)
    if a_p99 > 0:
        hsv[:, :, 2] = np.where(valid, np.clip(dom_amp / a_p99, 0, 1), 0)

    rgb = mcolors.hsv_to_rgb(hsv)
    ax.imshow(rgb, origin="lower", aspect="auto", extent=extent)

    _corner_label(ax, "(c)")
    configure_axes(ax)


@docs_figure("survey-map-plot.png")
def _build_figure(
    data: DecompositionData,
    velocity: npt.NDArray[np.float64],
    header: fits.Header,
) -> Figure:
    """Construct the 2x2 survey map figure.

    Layout (colourbar panels on the right column):

    +-------+-------+--+
    | (a)   | (b)   |cb|
    +-------+-------+--+
    | (c)   | (d)   |cb|
    +-------+-------+--+
    """
    nx = int(data.x.max()) + 1
    ny = int(data.y.max()) + 1
    extent = _galactic_extent(header, nx, ny)

    fig = plt.figure(figsize=(8, 7))
    outer = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1, 1.06],
        left=0.08,
        right=0.96,
        bottom=0.06,
        top=0.97,
        wspace=0.05,
        hspace=0.05,
    )

    # Left column: plain image panels
    ax_a = fig.add_subplot(outer[0, 0])
    ax_c = fig.add_subplot(outer[1, 0], sharex=ax_a)

    # Right column: image + thin colorbar side-by-side
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

    # Shared axis labels — only on outer edges
    ax_a.tick_params(labelbottom=False)
    ax_b.tick_params(labelbottom=False, labelleft=False)
    ax_d.tick_params(labelleft=False)
    ax_c.set_xlabel("Galactic longitude (deg)")
    ax_d.set_xlabel("Galactic longitude (deg)")
    ax_a.set_ylabel("Galactic latitude (deg)")
    ax_c.set_ylabel("Galactic latitude (deg)")

    return fig


@click.command("survey-map-plot")
@click.option("--survey", required=True, help="Survey name (S3 partition key).")
@click.option(
    "--environment",
    default="development",
    show_default=True,
    help="AWS environment.",
)
@click.option(
    "--fits-file",
    default=None,
    type=click.Path(exists=True),
    help="FITS cube path for WCS velocity axis (default: auto-download).",
)
@click.option("--cache-dir", default=CACHE_DIR, show_default=True, help="Cache directory.")
@click.option("--force", is_flag=True, help="Re-download even if cached.")
def survey_map(
    survey: str,
    environment: str,
    fits_file: str | None,
    cache_dir: str,
    force: bool,
) -> None:
    """Generate 2x2 survey visualisation from full-field decomposition."""
    console.print("Downloading decomposition data ...", style="bold cyan")
    paths = _download_decompositions(survey, environment, cache_dir, force)
    if not paths:
        console.print("No data found. Aborting.", style="bold red")
        return

    console.print("Loading Parquet data ...", style="bold cyan")
    table = _load_parquet_data(paths)
    data = _table_to_arrays(table)

    console.print("Loading FITS header ...", style="bold cyan")
    if fits_file:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header.copy()  # pylint: disable=no-member
    else:
        header, _ = ensure_fits()
    velocity = _velocity_axis(header)

    console.print("Building figure ...", style="bold cyan")
    _build_figure(data, velocity, header)
    console.print("\nDone.", style="bold green")
