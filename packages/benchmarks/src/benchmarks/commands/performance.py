"""``benchmarks performance`` — timing benchmark: phspectra vs GaussPy+ (Docker)."""

from __future__ import annotations

import json
import os
import sys
import time

import click
import numpy as np
from benchmarks._console import console, err_console
from benchmarks._constants import (
    CACHE_DIR,
    DEFAULT_BETA,
    DEFAULT_SEED,
)
from benchmarks._data import (
    ensure_catalog,
    ensure_fits,
    fits_bounds,
    select_spectra,
)
from benchmarks._docker import build_image, run_gausspyplus
from numpy.linalg import LinAlgError

from phspectra import fit_gaussians


@click.command()
@click.option("--n-spectra", default=200, show_default=True)
@click.option("--beta", default=DEFAULT_BETA, show_default=True)
@click.option("--seed", default=DEFAULT_SEED, show_default=True)
def performance(n_spectra: int, beta: float, seed: int) -> None:
    """Run timing benchmark: phspectra vs GaussPy+ (Docker)."""
    import matplotlib.pyplot as plt

    output_dir = os.path.join(CACHE_DIR, "performance-benchmark")
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    console.print("Step 1: GRS test field FITS", style="bold cyan")
    header, cube = ensure_fits()
    console.print(f"  Cube: {cube.shape}")

    console.print("\nStep 2: GaussPy+ catalog", style="bold cyan")
    bounds = fits_bounds(header)
    catalog = ensure_catalog(*bounds)
    if len(catalog) == 0:
        err_console.print("ERROR: no catalog entries")
        sys.exit(1)

    console.print("\nStep 3: Select spectra", style="bold cyan")
    _, signals = select_spectra(cube, header, catalog, n_spectra, seed)

    npz_path = os.path.join(output_dir, "spectra.npz")
    np.savez(npz_path, signals=signals)

    # phspectra benchmark
    console.print(f"\nStep 4: phspectra benchmark (beta={beta})", style="bold cyan")
    ph_times: list[float] = []
    ph_n_comps: list[int] = []
    t_ph_start = time.perf_counter()
    for i, _ in enumerate(signals):
        t0 = time.perf_counter()
        try:
            comps = fit_gaussians(signals[i], beta=beta, max_components=8)
        except (LinAlgError, ValueError):
            comps = []
        ph_times.append(time.perf_counter() - t0)
        ph_n_comps.append(len(comps))
        if (i + 1) % 50 == 0:
            console.print(f"  {i + 1}/{len(signals)}", style="dim")
    t_ph_total = time.perf_counter() - t_ph_start
    console.print(
        f"  phspectra: {t_ph_total:.1f}s total, "
        f"{t_ph_total / len(signals) * 1000:.1f}ms/spectrum",
        style="green",
    )

    # GaussPy+ Docker benchmark
    console.print("\nStep 5: GaussPy+ benchmark (Docker)", style="bold cyan")
    build_image()
    gp_results = run_gausspyplus(os.path.abspath(output_dir))
    gp_total = gp_results["total_time_s"]
    gp_times = gp_results["times"]

    # Save results
    speedup = gp_total / t_ph_total if t_ph_total > 0 else 0
    combined = {
        "phspectra": {
            "total_time_s": round(t_ph_total, 3),
            "mean_time_per_spectrum_s": round(t_ph_total / len(signals), 6),
            "mean_n_components": round(float(np.mean(ph_n_comps)), 2),
            "times": [round(t, 6) for t in ph_times],
        },
        "gausspyplus": gp_results,
        "speedup": round(speedup, 2),
    }
    json_path = os.path.join(output_dir, "performance_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    console.print(f"  JSON: [blue]{json_path}[/blue]")

    # Plot
    ph_arr = np.array(ph_times) * 1000
    gp_arr = np.array(gp_times) * 1000
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    bins = np.linspace(0, max(np.percentile(ph_arr, 99), np.percentile(gp_arr, 99)), 40)
    axes[0].hist(ph_arr, bins=bins, alpha=0.7, label="phspectra", color="C0")
    axes[0].hist(gp_arr, bins=bins, alpha=0.7, label="GaussPy+", color="C3")
    axes[0].set_xlabel("Time per spectrum (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Timing distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    bp = axes[1].boxplot([ph_arr, gp_arr], tick_labels=["phspectra", "GaussPy+"], patch_artist=True)
    bp["boxes"][0].set_facecolor("C0")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("C3")
    bp["boxes"][1].set_alpha(0.7)
    axes[1].set_ylabel("Time per spectrum (ms)")
    axes[1].set_title("Timing box plot")
    axes[1].grid(True, alpha=0.3, axis="y")

    rects = axes[2].bar(
        ["phspectra", "GaussPy+"], [t_ph_total, gp_total], color=["C0", "C3"], alpha=0.8
    )
    axes[2].set_ylabel("Total time (s)")
    axes[2].set_title(f"Total wall time ({len(ph_times)} spectra)\nSpeedup: {speedup:.1f}x")
    axes[2].grid(True, alpha=0.3, axis="y")
    for rect, val in zip(rects, [t_ph_total, gp_total]):
        axes[2].text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle(
        f"Performance: phspectra vs GaussPy+ — {len(ph_times)} real GRS spectra",
        fontsize=13,
    )
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "performance-benchmark.png")
    fig.savefig(plot_path, dpi=150)
    console.print(f"  Plot: [blue]{plot_path}[/blue]")
    plt.close(fig)

    console.print(
        f"\nSpeedup: [bold yellow]{speedup:.1f}x[/bold yellow]",
    )
    console.print("Done.", style="bold green")
