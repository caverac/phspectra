"""``benchmarks ncomp-rms-plot`` -- N components vs residual RMS scatter.

Reads saved comparison data and produces a two-panel (stacked) figure
showing the number of fitted components vs residual RMS for phspectra
and GaussPy+.
"""

from __future__ import annotations

import json
import os

import click
import numpy as np
from benchmarks._console import console, err_console
from benchmarks._constants import CACHE_DIR
from benchmarks._plotting import configure_axes, docs_figure
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _compute_rms_and_ncomp(
    signals: np.ndarray,
    data: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-spectrum residual RMS and component counts.

    Parameters
    ----------
    signals : np.ndarray
        Array of shape ``(n_spectra, n_channels)``.
    data : dict
        Loaded JSON with ``amplitudes_fit``, ``means_fit``,
        ``stddevs_fit`` keys.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(rms, n_components)`` arrays of length ``n_spectra``.
    """
    n = len(signals)
    x = np.arange(signals.shape[1], dtype=np.float64)
    rms = np.empty(n)
    ncomp = np.empty(n, dtype=int)
    for i in range(n):
        sig = signals[i]
        model = np.zeros_like(sig)
        for a, m, s in zip(
            data["amplitudes_fit"][i],
            data["means_fit"][i],
            data["stddevs_fit"][i],
        ):
            model += a * np.exp(-0.5 * ((x - m) / s) ** 2)
        rms[i] = np.sqrt(np.mean((sig - model) ** 2))
        ncomp[i] = len(data["amplitudes_fit"][i])
    return rms, ncomp


def _annotate_panel(ax: Axes, rms: np.ndarray, ncomp: np.ndarray) -> None:
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
    ph_rms: np.ndarray,
    ph_ncomp: np.ndarray,
    gp_rms: np.ndarray,
    gp_ncomp: np.ndarray,
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
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
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
    help="Directory containing spectra.npz, results.json, and phspectra_results.json.",
)
def ncomp_rms_plot(data_dir: str) -> None:
    """Generate N-components vs residual RMS scatter from saved comparison data."""
    for name in ("spectra.npz", "phspectra_results.json", "results.json"):
        if not os.path.exists(os.path.join(data_dir, name)):
            err_console.print(f"ERROR: {name} not found in {data_dir}")
            raise SystemExit(1)

    console.print("Loading comparison data...", style="bold cyan")
    signals = np.load(os.path.join(data_dir, "spectra.npz"))["signals"]
    with open(os.path.join(data_dir, "phspectra_results.json"), encoding="utf-8") as f:
        ph_data = json.load(f)
    with open(os.path.join(data_dir, "results.json"), encoding="utf-8") as f:
        gp_data = json.load(f)

    ph_rms, ph_ncomp = _compute_rms_and_ncomp(signals, ph_data)
    gp_rms, gp_ncomp = _compute_rms_and_ncomp(signals, gp_data)

    console.print(f"  {len(signals)} spectra loaded from [blue]{data_dir}[/blue]")
    _build_ncomp_rms(ph_rms, ph_ncomp, gp_rms, gp_ncomp)
    console.print("Done.", style="bold green")
