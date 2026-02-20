"""Shared survey-map utilities: data loading, WCS helpers, and panel renderers.

These building blocks are used by ``grs-map-plot`` (multi-tile strip)
and ``correlation-plot`` (two-point autocorrelation).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import boto3
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.wcs import WCS
from benchmarks._console import console
from benchmarks._constants import DATA_BUCKET_TEMPLATE
from benchmarks._plotting import configure_axes
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.ndimage import binary_opening


@dataclass
class DecompositionData:
    """Decomposition results loaded from Parquet files."""

    x: npt.NDArray[np.int32]
    y: npt.NDArray[np.int32]
    n_components: npt.NDArray[np.int32]
    component_amplitudes: list[list[float]]
    component_means: list[list[float]]
    component_stddevs: list[list[float]]


def _filter_sparse_pixels(
    data: DecompositionData,
    nx: int,
    ny: int,
) -> DecompositionData:
    """Remove pixels in regions with sparse component coverage.

    Builds a mask of pixels that have at least one Gaussian component
    (``n_components > 0``) and applies binary opening with a 3x3
    structuring element.  Pixels whose component-coverage neighborhood
    does not survive the opening — isolated detections surrounded by
    zero-component pixels at tile edges — are dropped.
    """
    xi = data.x.astype(int)
    yi = data.y.astype(int)
    has_comp = data.n_components > 0

    coverage = np.zeros((ny, nx), dtype=bool)
    valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    coverage[yi[valid & has_comp], xi[valid & has_comp]] = True

    dense = binary_opening(coverage, structure=np.ones((3, 3)))
    keep = dense[yi, xi]

    return DecompositionData(
        x=data.x[keep],
        y=data.y[keep],
        n_components=data.n_components[keep],
        component_amplitudes=[a for a, k in zip(data.component_amplitudes, keep) if k],
        component_means=[m for m, k in zip(data.component_means, keep) if k],
        component_stddevs=[s for s, k in zip(data.component_stddevs, keep) if k],
    )


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
        if data.n_components[i] >= 0:
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
        if data.n_components[i] < 0:
            continue
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
