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

from benchmarks._console import console, err_console
from benchmarks._constants import CACHE_DIR
from benchmarks._data import ensure_fits, match_catalog_pixels
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
    default="5.0,3.0,2.0",
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
    catalog_path = os.path.join(data_dir, "..", "gausspy-catalog.votable")
    docker_results_path = os.path.join(data_dir, "results.json")

    # Load data
    header, cube = ensure_fits(path=fits_path)
    n_channels = cube.shape[0]

    if not os.path.exists(docker_results_path):
        err_console.print("ERROR: results.json not found, run ``benchmarks compare`` first")
        sys.exit(1)

    # Rebuild pixel selection to find spec_idx
    from astropy.table import Table

    catalog = Table.read(catalog_path, format="votable")
    pixel_counts = match_catalog_pixels(catalog, header)
    eligible = {k: v for k, v in pixel_counts.items() if 1 <= v <= 8}
    eligible_keys = list(eligible.keys())

    seed = 2026_02_12
    n_spectra = 400
    rng = np.random.default_rng(seed)
    n_select = min(n_spectra, len(eligible_keys))
    idx = rng.choice(len(eligible_keys), size=n_select, replace=False)
    selected = [eligible_keys[i] for i in idx]

    try:
        spec_idx = selected.index((px, py))
    except ValueError:
        err_console.print(f"ERROR: pixel ({px}, {py}) not in the {n_select} selected spectra.")
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
            console.print(f"  beta={beta}, sig_min={sig_min}: {len(comps)} components")

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
        n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows), sharex=True, sharey=True
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.07, top=0.93, wspace=0.08, hspace=0.18)
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
                linewidth=2.0,
                linestyle="--",
                label=f"GP+ ({len(gp_comps)} comp, RMS={gp_rms:.3f})",
            )

            # phspectra model
            comps = ph_results[(beta, sig_min)]
            ph_model = gaussian_model(x, comps)
            rms = float(np.sqrt(np.mean((signal - ph_model) ** 2)))
            ax.plot(
                x,
                ph_model,
                color="k",
                linewidth=2.0,
                label=f"PH ({len(comps)} comp, RMS={rms:.3f})",
            )

            ax.set_xlim(lo, hi)
            ax.text(
                0.03,
                0.05,
                f"$\\beta$={beta}, sig_min={sig_min}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
            )
            ax.legend(loc="upper right", frameon=False, fontsize=7)
            configure_axes(ax)

            if i == n_rows - 1:
                ax.set_xlabel("Channel")
            if j == 0:
                ax.set_ylabel("T (K)")

    fig.suptitle(f"Pixel ({px}, {py})")
    plt.show()
