"""``benchmarks inspect`` -- inspect a single GRS pixel interactively.

Decomposes a single GRS pixel with phspectra at several (beta, mf_snr_min)
combinations and overlays the results against the GaussPy+ Docker
decomposition.  Useful for diagnosing why the two methods disagree on a
particular spectrum.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import click
import numpy as np
from benchmarks._console import console, err_console
from benchmarks._constants import CACHE_DIR
from benchmarks._data import ensure_fits
from benchmarks._database import load_components, load_pixels
from benchmarks._gaussian import gaussian_model
from benchmarks._plotting import configure_axes
from benchmarks._types import Component
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError
from rich.table import Table

from phspectra import fit_gaussians


def _normalize_axes_grid(axes: Any, n_rows: int, n_cols: int) -> Any:
    """Ensure *axes* is always indexable as ``axes[i][j]``."""
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    if n_rows == 1:
        return np.array([axes])
    if n_cols == 1:
        return np.array([[ax] for ax in axes])
    return axes


def _compute_zoom(
    all_comps: list[list[Component]],
    n_channels: int,
) -> tuple[int, int]:
    """Return (lo, hi) channel range enclosing all components."""
    all_means = [c.mean for comps in all_comps for c in comps]
    all_stds = [c.stddev for comps in all_comps for c in comps]
    if not all_means:
        return 0, n_channels
    lo = max(0, int(min(all_means) - 4 * max(all_stds + [5])) - 10)
    hi = min(n_channels, int(max(all_means) + 4 * max(all_stds + [5])) + 10)
    return lo, hi


@click.command("inspect")
@click.argument("px", type=int)
@click.argument("py", type=int)
@click.option(
    "--data-dir",
    default=os.path.join(CACHE_DIR, "compare-docker"),
    show_default=True,
)
@click.option(
    "--betas",
    default="3.35,3.50,3.65",
    show_default=True,
    help="Comma-separated beta values.",
)
@click.option(
    "--mf-snr-mins",
    default="4.0,5.0,6.0",
    show_default=True,
    help="Comma-separated matched-filter SNR thresholds.",
)
def inspect_pixel(
    px: int,
    py: int,
    data_dir: str,
    betas: str,
    mf_snr_mins: str,
) -> None:
    """Inspect pixel (PX, PY): spectrum + GP+ + phspectra at multiple betas."""
    beta_list = [float(b) for b in betas.split(",")]
    mf_snr_min_list = [float(s) for s in mf_snr_mins.split(",")]

    db_path = os.path.join(data_dir, "pre-compute.db")
    if not os.path.exists(db_path):
        err_console.print("ERROR: pre-compute.db not found, run ``benchmarks pre-compute`` first")
        sys.exit(1)

    fits_path = os.path.join(data_dir, "..", "grs-test-field.fits")

    # Load data
    _, cube = ensure_fits(path=fits_path)
    n_channels = cube.shape[0]

    # Read pixel list from SQLite
    ph_pixel_rows = load_pixels(db_path, "phspectra")
    selected = [(r["xpos"], r["ypos"]) for r in ph_pixel_rows]

    try:
        spec_idx = selected.index((px, py))
    except ValueError:
        err_console.print(f"ERROR: pixel ({px}, {py}) not in the {len(selected)} selected spectra.")
        nearby = [(x, y) for x, y in selected if abs(x - px) <= 5 and abs(y - py) <= 5]
        if nearby:
            err_console.print(f"Nearby pixels: {nearby}")
        sys.exit(1)

    signal = np.nan_to_num(cube[:, py, px].astype(np.float64), nan=0.0)
    x = np.arange(n_channels, dtype=np.float64)

    # GaussPy+ results from SQLite
    gp_comp_map = load_components(db_path, "gausspyplus")
    gp_raw = gp_comp_map.get((px, py), [])
    gp_comps = [Component(a, m, s) for a, m, s in gp_raw]
    gp_model = gaussian_model(x, gp_comps)
    gp_rms = float(np.sqrt(np.mean((signal - gp_model) ** 2)))

    console.print(f"Pixel ({px}, {py}), index {spec_idx}", style="bold cyan")
    console.print(f"GaussPy+: {len(gp_comps)} components, RMS={gp_rms:.4f}")

    # phspectra at multiple (beta, mf_snr_min)
    ph_results: dict[tuple[float, float], list[Component]] = {}
    for beta in beta_list:
        for mf_snr_min in mf_snr_min_list:
            try:
                comps = fit_gaussians(signal, beta=beta, mf_snr_min=mf_snr_min)
            except (LinAlgError, ValueError):
                comps = []
            ph_results[(beta, mf_snr_min)] = [Component(c.amplitude, c.mean, c.stddev) for c in comps]

    table = Table(title="phspectra decompositions")
    table.add_column("beta", justify="right")
    table.add_column("mf_snr_min", justify="right")
    table.add_column("components", justify="right")
    for beta in beta_list:
        for mf_snr_min in mf_snr_min_list:
            n = len(ph_results[(beta, mf_snr_min)])
            table.add_row(f"{beta:.2f}", f"{mf_snr_min:.1f}", str(n))
    console.print(table)

    # Zoom range
    all_comps = [gp_comps] + list(ph_results.values())
    lo, hi = _compute_zoom(all_comps, n_channels)

    # Plot
    n_rows, n_cols = len(beta_list), len(mf_snr_min_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.07, top=0.93, wspace=0.05, hspace=0.05)
    axes = _normalize_axes_grid(axes, n_rows, n_cols)

    for i, beta in enumerate(beta_list):
        for j, mf_snr_min in enumerate(mf_snr_min_list):
            ax = axes[i][j]
            ax.step(x, signal, where="mid", color="0.6", linewidth=1.0, alpha=0.7, label="Data")
            ax.plot(
                x,
                gp_model,
                color="k",
                linewidth=1.5,
                label=f"GP+ ({len(gp_comps)} comp, RMS={gp_rms:.3f})",
            )

            ph_comps = ph_results[(beta, mf_snr_min)]
            ph_model = gaussian_model(x, ph_comps)
            rms = float(np.sqrt(np.mean((signal - ph_model) ** 2)))
            ax.plot(
                x,
                ph_model,
                color="#1f77b4",
                linewidth=1.0,
                linestyle="--",
                label=f"PHS ({len(ph_comps)} comp, RMS={rms:.3f})",
            )

            ax.set_xlim(lo, hi)
            ax.text(
                0.03,
                0.05,
                f"$\\beta$={beta}, MF$_{{\\min}}$={mf_snr_min}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
            )
            ax.legend(loc="upper right", frameon=False)
            configure_axes(ax)

            if i == n_rows - 1:
                ax.set_xlabel("Channel")
            if j == 0:
                ax.set_ylabel(r"$T$ (K)")

    fig.suptitle(f"Pixel ({px}, {py})")
    plt.show()
