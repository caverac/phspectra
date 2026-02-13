"""``benchmarks performance`` — timing benchmark: phspectra vs GaussPy+ (Docker)."""

from __future__ import annotations

import json
import os
import sys
import time

import click
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from benchmarks._console import console, err_console
from benchmarks._constants import (
    CACHE_DIR,
    DEFAULT_BETA,
    DEFAULT_SEED,
)
from benchmarks._plotting import docs_figure
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
    _plot_timing(np.array(ph_times) * 1000, np.array(gp_times) * 1000)

    console.print(
        f"\nSpeedup: [bold yellow]{speedup:.1f}x[/bold yellow]",
    )
    console.print("Done.", style="bold green")


def _configure_axes(ax: Axes) -> None:
    """Apply the shared tick/grid style."""
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="minor", length=3, color="gray", direction="in")
    ax.tick_params(which="major", length=6, direction="in")
    ax.tick_params(top=True, right=True, which="both")


@docs_figure("performance-benchmark.png")
def _plot_timing(
    ph_ms: np.ndarray,
    gp_ms: np.ndarray,
) -> Figure:
    """Single-frame timing distribution histogram."""
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

    clip = max(np.percentile(ph_ms, 99), np.percentile(gp_ms, 99))
    bins = np.linspace(0, clip, 40)
    ax.hist(ph_ms, bins=bins, alpha=0.7, color="k", label="phspectra")
    ax.hist(gp_ms, bins=bins, alpha=0.2, color="k", label="GaussPy+",
            edgecolor="k")

    ax.set_xlabel("Time per spectrum (ms)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", frameon=False)
    _configure_axes(ax)
    return fig


@click.command("performance-plot")
@click.option(
    "--data-dir",
    default=os.path.join(CACHE_DIR, "compare-docker"),
    show_default=True,
    help="Directory containing phspectra_results.json and results.json.",
)
def performance_plot(data_dir: str) -> None:
    """Generate performance plot from saved comparison data."""
    ph_path = os.path.join(data_dir, "phspectra_results.json")
    gp_path = os.path.join(data_dir, "results.json")
    for path, label in [(ph_path, "phspectra_results.json"), (gp_path, "results.json")]:
        if not os.path.exists(path):
            err_console.print(f"ERROR: {label} not found in {data_dir}.\nRun ``benchmarks compare`` first.")
            sys.exit(1)

    console.print("Loading saved comparison data ...", style="bold cyan")
    with open(ph_path, encoding="utf-8") as f:
        ph_data = json.load(f)
    with open(gp_path, encoding="utf-8") as f:
        gp_data = json.load(f)

    ph_ms = np.array(ph_data["times"]) * 1000
    gp_ms = np.array(gp_data["times"]) * 1000
    speedup = sum(gp_data["times"]) / max(sum(ph_data["times"]), 1e-9)
    console.print(f"  {len(ph_ms)} spectra, speedup={speedup:.1f}x")

    _plot_timing(ph_ms, gp_ms)
    console.print("Done.", style="bold green")
