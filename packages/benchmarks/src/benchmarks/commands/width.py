"""``benchmarks width`` — compare fitted widths: phspectra vs GaussPy+ (Docker)."""

from __future__ import annotations

import json
import os
import sys

import click
import numpy as np
from benchmarks._console import console, err_console
from benchmarks._constants import CACHE_DIR, DEFAULT_BETA, DEFAULT_MAX_COMPONENTS
from numpy.linalg import LinAlgError
from scipy.optimize import linear_sum_assignment

from phspectra import fit_gaussians


@click.command()
@click.option(
    "--data-dir",
    default=os.path.join(CACHE_DIR, "compare-docker"),
    show_default=True,
)
@click.option("--beta", default=DEFAULT_BETA, show_default=True)
@click.option("--output-dir", default=None, help="Output directory (defaults to data-dir).")
def width(data_dir: str, beta: float, output_dir: str | None) -> None:
    """Compare fitted widths: phspectra vs GaussPy+ (Docker)."""
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = data_dir

    spectra_path = os.path.join(data_dir, "spectra.npz")
    results_path = os.path.join(data_dir, "results.json")
    if not os.path.exists(spectra_path) or not os.path.exists(results_path):
        err_console.print(
            f"ERROR: {spectra_path} or {results_path} not found.\n"
            "Run ``benchmarks compare`` first."
        )
        sys.exit(1)

    console.print("Loading spectra ...", style="bold cyan")
    signals = np.load(spectra_path)["signals"]
    n_spectra, n_channels = signals.shape
    console.print(f"  {n_spectra} spectra, {n_channels} channels")

    console.print("Loading GaussPy+ Docker results ...", style="bold cyan")
    with open(results_path, encoding="utf-8") as f:
        gp = json.load(f)

    console.print(f"Running phspectra (beta={beta}) ...", style="bold cyan")
    ph_all: list[list[tuple[float, float, float]]] = []
    for i in range(n_spectra):
        try:
            comps = fit_gaussians(
                signals[i], beta=beta, max_components=DEFAULT_MAX_COMPONENTS, sig_min=3.0
            )
        except (LinAlgError, ValueError):
            comps = []
        ph_all.append([(c.amplitude, c.mean, c.stddev) for c in comps])
        if (i + 1) % 100 == 0:
            console.print(f"  {i + 1}/{n_spectra}", style="dim")

    # Match components
    console.print("Matching components ...", style="bold cyan")
    gp_widths: list[float] = []
    ph_widths: list[float] = []
    gp_amps_matched: list[float] = []

    pos_tol_sigma = 2.0
    for i in range(n_spectra):
        gp_a = gp["amplitudes_fit"][i]
        gp_m = gp["means_fit"][i]
        gp_s = gp["stddevs_fit"][i]
        ph = ph_all[i]
        if not gp_a or not ph:
            continue
        n_gp, n_ph = len(gp_a), len(ph)
        cost = np.full((n_gp, n_ph), 1e9)
        for gi in range(n_gp):
            for pi in range(n_ph):
                cost[gi, pi] = abs(ph[pi][1] - gp_m[gi])
        row_ind, col_ind = linear_sum_assignment(cost)
        for gi, pi in zip(row_ind, col_ind):
            pos_diff = abs(ph[pi][1] - gp_m[gi])
            if pos_diff < pos_tol_sigma * gp_s[gi] and gp_s[gi] > 0.1:
                gp_widths.append(gp_s[gi])
                ph_widths.append(ph[pi][2])
                gp_amps_matched.append(gp_a[gi])

    gp_w = np.array(gp_widths)
    ph_w = np.array(ph_widths)
    gp_a_arr = np.array(gp_amps_matched)
    ratios = gp_w / np.maximum(ph_w, 0.1)

    n_gp_wider = int(np.sum(gp_w > ph_w))
    console.print(f"  {len(gp_w)} matched pairs", style="green")
    console.print(f"  GP+ wider: {n_gp_wider} ({n_gp_wider / len(gp_w) * 100:.0f}%)", style="green")
    console.print(f"  Median ratio (GP+/PH): {np.median(ratios):.2f}", style="green")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sc = axes[0].scatter(ph_w, gp_w, s=10, alpha=0.4, c=gp_a_arr, cmap="viridis", edgecolors="none")
    lim = max(ph_w.max(), gp_w.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.6, label="1:1")
    axes[0].set_xlabel(r"phspectra $\sigma$ (channels)")
    axes[0].set_ylabel(r"GaussPy+ $\sigma$ (channels)")
    axes[0].set_title("Matched component widths")
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)
    fig.colorbar(sc, ax=axes[0], shrink=0.8).set_label("GP+ amplitude (K)", fontsize=8)

    bins = np.linspace(0, min(np.percentile(ratios, 99), 10), 50)
    axes[1].hist(ratios, bins=bins, alpha=0.8, color="C0", edgecolor="white")
    axes[1].axvline(1.0, color="k", linestyle="--", linewidth=0.8, label="ratio = 1")
    axes[1].axvline(
        np.median(ratios),
        color="C3",
        linestyle="-",
        linewidth=1.5,
        label=f"median = {np.median(ratios):.2f}",
    )
    axes[1].set_xlabel(r"Width ratio ($\sigma_\mathrm{GP+} / \sigma_\mathrm{PH}$)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Width ratio distribution (N={len(ratios)})")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    diff = gp_w - ph_w
    axes[2].scatter(gp_a_arr, diff, s=10, alpha=0.4, color="C2", edgecolors="none")
    axes[2].axhline(0, color="k", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("GP+ amplitude (K)")
    axes[2].set_ylabel(r"$\sigma_\mathrm{GP+} - \sigma_\mathrm{PH}$ (channels)")
    axes[2].set_title("Width difference vs amplitude")
    axes[2].grid(True, alpha=0.2)

    pct = n_gp_wider / len(gp_w) * 100
    fig.suptitle(
        f"GaussPy+ fits wider profiles than phspectra: "
        f"{n_gp_wider}/{len(gp_w)} pairs ({pct:.0f}%), "
        f"median ratio = {np.median(ratios):.2f}",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = os.path.join(output_dir, "width-comparison.png")
    fig.savefig(out_path, dpi=150)
    console.print(f"\nPlot: [blue]{out_path}[/blue]")
    plt.close(fig)
