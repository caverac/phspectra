"""``benchmarks ncomp-rms-plot`` -- N components vs residual RMS scatter.

Reads saved comparison data from SQLite and produces a two-panel (stacked)
figure showing the number of fitted components vs residual RMS for phspectra
and GaussPy+.
"""

from __future__ import annotations

import os

import click
import numpy as np
import numpy.typing as npt
from benchmarks._console import console, err_console
from benchmarks._constants import CACHE_DIR
from benchmarks._database import load_pixels
from benchmarks._plotting import configure_axes, docs_figure
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _annotate_panel(ax: Axes, rms: npt.NDArray[np.float64], ncomp: npt.NDArray[np.intp]) -> None:
    """Add mean-component annotations to the left and right of the threshold.

    Parameters
    ----------
    ax : Axes
        Target axes.
    rms : np.ndarray
        Residual RMS values.
    ncomp : np.ndarray
        Component counts.
    """
    lo = rms <= 0.2
    hi = rms > 0.2
    mean_lo = ncomp[lo].mean() if lo.sum() > 0 else 0
    mean_hi = ncomp[hi].mean() if hi.sum() > 0 else 0
    ax.text(
        0.03,
        0.95,
        f"RMS $\\leq$ 0.2: $\\langle N \\rangle$ = {mean_lo:.1f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.text(
        0.97,
        0.95,
        f"RMS > 0.2: $\\langle N \\rangle$ = {mean_hi:.1f}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
    )


@docs_figure("ncomp-vs-rms.png")
def _build_ncomp_rms(
    ph_rms: npt.NDArray[np.float64],
    ph_ncomp: npt.NDArray[np.intp],
    gp_rms: npt.NDArray[np.float64],
    gp_ncomp: npt.NDArray[np.intp],
) -> Figure:
    """Build stacked scatter panels of N components vs residual RMS.

    Parameters
    ----------
    ph_rms, ph_ncomp : np.ndarray
        PHSpectra residual RMS and component counts.
    gp_rms, gp_ncomp : np.ndarray
        GaussPy+ residual RMS and component counts.

    Returns
    -------
    Figure
        Two-panel stacked matplotlib figure.
    """
    fig: Figure
    fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.11, top=0.95, hspace=0.12)

    for ax, rms, ncomp, label in [
        (axes[0], ph_rms, ph_ncomp, "PHSpectra"),
        (axes[1], gp_rms, gp_ncomp, "GaussPy+"),
    ]:
        ax.scatter(rms, ncomp, s=10, alpha=0.5, color="0.3", edgecolors="none")
        ax.axvline(0.2, color="k", linestyle="--", linewidth=1.5)
        ax.set_ylabel(f"$N$ components ({label})")
        ax.set_xlim(0.05, 0.40)
        ax.set_ylim(-0.1, 20)
        _annotate_panel(ax, rms, ncomp)
        configure_axes(ax)

    axes[1].set_xlabel("Residual RMS (K)")

    return fig


@click.command("ncomp-rms-plot")
@click.option(
    "--data-dir",
    default=os.path.join(CACHE_DIR, "compare-docker"),
    show_default=True,
    help="Directory containing pre-compute.db.",
)
def ncomp_rms_plot(data_dir: str) -> None:
    """Generate N-components vs residual RMS scatter from saved comparison data."""
    db_path = os.path.join(data_dir, "pre-compute.db")
    if not os.path.exists(db_path):
        err_console.print(f"ERROR: pre-compute.db not found in {data_dir}")
        raise SystemExit(1)

    console.print("Loading comparison data...", style="bold cyan")

    ph_pixel_rows = load_pixels(db_path, "phspectra")
    gp_pixel_rows = load_pixels(db_path, "gausspyplus")

    ph_rms = np.array([r["rms"] for r in ph_pixel_rows])
    ph_ncomp = np.array([r["n_components"] for r in ph_pixel_rows], dtype=int)
    gp_rms = np.array([r["rms"] for r in gp_pixel_rows])
    gp_ncomp = np.array([r["n_components"] for r in gp_pixel_rows], dtype=int)

    n_spectra = len(ph_pixel_rows)
    console.print(f"  {n_spectra} spectra loaded from [blue]{data_dir}[/blue]")

    # Component count statistics
    console.print(
        f"  Mean N components: phspectra {ph_ncomp.mean():.2f}, GP+ {gp_ncomp.mean():.2f}",
        style="green",
    )
    lo = ph_rms <= 0.2
    hi = ph_rms > 0.2
    if lo.sum() > 0:
        console.print(
            f"  RMS <= 0.2 K ({lo.sum()} spectra): "
            f"phspectra {ph_ncomp[lo].mean():.2f}, GP+ {gp_ncomp[lo].mean():.2f} mean components",
            style="green",
        )
    if hi.sum() > 0:
        console.print(
            f"  RMS >  0.2 K ({hi.sum()} spectra): "
            f"phspectra {ph_ncomp[hi].mean():.2f}, GP+ {gp_ncomp[hi].mean():.2f} mean components",
            style="green",
        )

    _build_ncomp_rms(ph_rms, ph_ncomp, gp_rms, gp_ncomp)
    console.print("Done.", style="bold green")
