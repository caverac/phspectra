"""``benchmarks performance-plot`` -- generate timing histogram from saved data.

Reads the SQLite outputs of ``benchmarks pre-compute`` and produces a histogram
comparing per-spectrum wall-clock time for phspectra and GaussPy+.
"""

from __future__ import annotations

import os
import sys

import click
import numpy as np
import numpy.typing as npt
from benchmarks._console import console, err_console
from benchmarks._constants import CACHE_DIR
from benchmarks._database import load_pixels, load_run
from benchmarks._plotting import configure_axes, docs_figure
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@docs_figure("performance-benchmark.png")
def _plot_timing(
    ph_ms: npt.NDArray[np.float64],
    gp_ms: npt.NDArray[np.float64],
) -> Figure:
    """Build overlaid histograms of per-spectrum wall-clock time.

    Parameters
    ----------
    ph_ms : npt.NDArray[np.float64]
        phspectra times in milliseconds.
    gp_ms : npt.NDArray[np.float64]
        GaussPy+ times in milliseconds.

    Returns
    -------
    Figure
        Single-axes matplotlib figure.
    """
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

    clip = max(np.percentile(ph_ms, 99), np.percentile(gp_ms, 99))
    bins = np.linspace(0, clip, 40)
    ph_counts, ph_edges, _ = ax.hist(
        ph_ms,
        bins=bins,  # type: ignore[arg-type]
        alpha=0.7,
        color="k",
        label="PHSpectra",
    )
    gp_counts, gp_edges, _ = ax.hist(
        gp_ms,
        bins=bins,  # type: ignore[arg-type]
        alpha=0.2,
        color="#4d4d4d",
        label="GaussPy+",
    )
    ax.stairs(ph_counts, ph_edges, color="k", linewidth=1.2)
    ax.stairs(gp_counts, gp_edges, color="#4d4d4d", linewidth=1.2)

    ax.set_xlabel("Time per spectrum (ms)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", frameon=False)
    configure_axes(ax)
    return fig


@click.command("performance-plot")
@click.option(
    "--data-dir",
    default=os.path.join(CACHE_DIR, "compare-docker"),
    show_default=True,
    help="Directory containing pre-compute.db.",
)
def performance_plot(data_dir: str) -> None:
    """Generate performance plot from saved comparison data."""
    db_path = os.path.join(data_dir, "pre-compute.db")
    if not os.path.exists(db_path):
        err_console.print(f"ERROR: pre-compute.db not found in {data_dir}.\nRun ``benchmarks pre-compute`` first.")
        sys.exit(1)

    console.print("Loading saved comparison data ...", style="bold cyan")
    ph_run = load_run(db_path, "phspectra")
    gp_run = load_run(db_path, "gausspyplus")

    ph_pixel_rows = load_pixels(db_path, "phspectra")
    gp_pixel_rows = load_pixels(db_path, "gausspyplus")

    ph_ms = np.array([r["time_s"] for r in ph_pixel_rows]) * 1000
    gp_ms = np.array([r["time_s"] for r in gp_pixel_rows]) * 1000
    n_spectra = len(ph_ms)
    ph_total = ph_run["total_time_s"]
    gp_total = gp_run["total_time_s"]
    speedup = gp_total / max(ph_total, 1e-9)

    ph_mean_n = float(np.mean([r["n_components"] for r in ph_pixel_rows]))
    gp_mean_n = float(np.mean([r["n_components"] for r in gp_pixel_rows]))

    console.print(f"  {n_spectra} spectra", style="green")
    console.print(
        f"\n  {'Metric':<30} {'PHSpectra':>12} {'GaussPy+':>12} {'Factor':>8}",
        style="bold",
    )
    console.print(f"  {'─' * 62}")
    console.print(f"  {'Total time':<30} {ph_total:>10.1f} s {gp_total:>10.1f} s {speedup:>6.1f}x")
    console.print(
        f"  {'Mean per spectrum':<30} {ph_ms.mean():>9.1f} ms "
        f"{gp_ms.mean():>9.1f} ms {gp_ms.mean() / max(ph_ms.mean(), 1e-9):>6.1f}x"
    )
    console.print(f"  {'Median per spectrum':<30} {np.median(ph_ms):>9.1f} ms {np.median(gp_ms):>9.1f} ms")
    console.print(f"  {'Std dev per spectrum':<30} {ph_ms.std():>9.1f} ms {gp_ms.std():>9.1f} ms")
    console.print(f"  {'P95 per spectrum':<30} {np.percentile(ph_ms, 95):>9.1f} ms {np.percentile(gp_ms, 95):>9.1f} ms")
    console.print(f"  {'P99 per spectrum':<30} {np.percentile(ph_ms, 99):>9.1f} ms {np.percentile(gp_ms, 99):>9.1f} ms")
    console.print(f"  {'Mean N components':<30} {ph_mean_n:>12.2f} {gp_mean_n:>12.2f}")
    console.print(f"\n  Speedup: [bold yellow]{speedup:.1f}x[/bold yellow]")

    _plot_timing(ph_ms, gp_ms)
    console.print("Done.", style="bold green")
