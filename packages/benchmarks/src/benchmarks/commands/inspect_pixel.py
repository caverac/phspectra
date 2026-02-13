"""``benchmarks inspect`` -- inspect a single GRS pixel interactively.

Decomposes a single GRS pixel with phspectra at several (beta, sig_min)
combinations and overlays the results against the GaussPy+ Docker
decomposition.  Useful for diagnosing why the two methods disagree on a
particular spectrum.
"""

from __future__ import annotations

import json
import os
import sys

import click
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError
from rich.table import Table

from benchmarks._console import console, err_console
from benchmarks._constants import CACHE_DIR
from benchmarks._data import ensure_fits
from benchmarks._gaussian import gaussian_model
from benchmarks._plotting import configure_axes
from benchmarks._types import Component

from phspectra import fit_gaussians


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
    default="3.50,3.75,4.0",
    show_default=True,
    help="Comma-separated beta values.",
)
@click.option(
    "--sig-mins",
    default="4.0,4.5,5.0",
    show_default=True,
    help="Comma-separated sig_min values.",
)
def inspect_pixel(
    px: int,
    py: int,
    data_dir: str,
    betas: str,
    sig_mins: str,
) -> None:
    """Inspect pixel (PX, PY): spectrum + GP+ + phspectra at multiple betas."""
    beta_list = [float(b) for b in betas.split(",")]
    sig_min_list = [float(s) for s in sig_mins.split(",")]

    fits_path = os.path.join(data_dir, "..", "grs-test-field.fits")
    docker_results_path = os.path.join(data_dir, "results.json")

    # Load data
    _, cube = ensure_fits(path=fits_path)
    n_channels = cube.shape[0]

    if not os.path.exists(docker_results_path):
        err_console.print("ERROR: results.json not found, run ``benchmarks compare`` first")
        sys.exit(1)

    # Read pixel list saved by ``benchmarks compare``
    ph_results_path = os.path.join(data_dir, "phspectra_results.json")
    if not os.path.exists(ph_results_path):
        err_console.print(
            "ERROR: phspectra_results.json not found, run ``benchmarks compare`` first"
        )
        sys.exit(1)
    with open(ph_results_path, encoding="utf-8") as f:
        ph_saved = json.load(f)
    selected = [tuple(p) for p in ph_saved["pixels"]]

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

    # GaussPy+ results
    with open(docker_results_path, encoding="utf-8") as f:
        gp = json.load(f)
    gp_comps = [
        Component(a, m, s)
        for a, m, s in zip(
            gp["amplitudes_fit"][spec_idx],
            gp["means_fit"][spec_idx],
            gp["stddevs_fit"][spec_idx],
        )
    ]
    gp_model = gaussian_model(x, gp_comps)
    gp_rms = float(np.sqrt(np.mean((signal - gp_model) ** 2)))

    console.print(f"Pixel ({px}, {py}), index {spec_idx}", style="bold cyan")
    console.print(f"GaussPy+: {len(gp_comps)} components, RMS={gp_rms:.4f}")

    # phspectra at multiple (beta, sig_min)
    ph_results: dict[tuple[float, float], list[Component]] = {}
    for beta in beta_list:
        for sig_min in sig_min_list:
            try:
                comps = fit_gaussians(signal, beta=beta, max_components=10, sig_min=sig_min)
            except (LinAlgError, ValueError):
                comps = []
            ph_results[(beta, sig_min)] = [Component(c.amplitude, c.mean, c.stddev) for c in comps]

    table = Table(title="phspectra decompositions")
    table.add_column("beta", justify="right")
    table.add_column("sig_min", justify="right")
    table.add_column("components", justify="right")
    for beta in beta_list:
        for sig_min in sig_min_list:
            n = len(ph_results[(beta, sig_min)])
            table.add_row(f"{beta:.2f}", f"{sig_min:.2f}", str(n))
    console.print(table)

    # Zoom range
    all_comps = [gp_comps] + list(ph_results.values())
    all_means = [c.mean for comps in all_comps for c in comps]
    all_stds = [c.stddev for comps in all_comps for c in comps]
    if all_means:
        lo = max(0, int(min(all_means) - 4 * max(all_stds + [5])) - 10)
        hi = min(n_channels, int(max(all_means) + 4 * max(all_stds + [5])) + 10)
    else:
        lo, hi = 0, n_channels

    # Plot
    n_rows, n_cols = len(beta_list), len(sig_min_list)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.0 * n_rows), sharex=True, sharey=True
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.07, top=0.93, wspace=0.05, hspace=0.05)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, beta in enumerate(beta_list):
        for j, sig_min in enumerate(sig_min_list):
            ax = axes[i][j]

            # Data
            ax.step(x, signal, where="mid", color="0.6", linewidth=1.0, alpha=0.7, label="Data")

            # GaussPy+ model
            ax.plot(
                x,
                gp_model,
                color="k",
                linewidth=1.5,
                label=f"GP+ ({len(gp_comps)} comp, RMS={gp_rms:.3f})",
            )

            # phspectra model
            comps = ph_results[(beta, sig_min)]
            ph_model = gaussian_model(x, comps)
            rms = float(np.sqrt(np.mean((signal - ph_model) ** 2)))
            ax.plot(
                x,
                ph_model,
                color="#1f77b4",
                linewidth=1.0,
                linestyle="--",
                label=f"PHS ({len(comps)} comp, RMS={rms:.3f})",
            )

            ax.set_xlim(lo, hi)
            ax.text(
                0.03,
                0.05,
                f"$\\beta$={beta}, $\\sigma_{{\\min}}$={sig_min}",
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
