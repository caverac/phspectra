"""``benchmarks compare`` -- phspectra vs GaussPy+ (Docker) on real GRS spectra.

Selects random spectra from the GRS test field, decomposes them with
both phspectra and the GaussPy+ Docker container, and saves component
parameters, RMS values, and timing to JSON files for downstream
plotting with ``benchmarks compare-plot``.
"""

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
    DEFAULT_N_SPECTRA,
    DEFAULT_SEED,
)
from benchmarks._data import (
    ensure_catalog,
    ensure_fits,
    fits_bounds,
    select_spectra,
)
from benchmarks._docker import build_image, run_gausspyplus
from benchmarks._gaussian import residual_rms
from benchmarks._types import Component
from numpy.linalg import LinAlgError

from phspectra import fit_gaussians
from phspectra._types import GaussianComponent


@click.command()
@click.option("--n-spectra", default=DEFAULT_N_SPECTRA, show_default=True)
@click.option("--beta", default=DEFAULT_BETA, show_default=True)
@click.option("--seed", default=DEFAULT_SEED, show_default=True)
@click.option("--output-dir", default=os.path.join(CACHE_DIR, "compare-docker"), show_default=True)
@click.option(
    "--extra-pixels",
    default="",
    show_default=False,
    help='Force extra pixel coordinates, e.g. "10,20;30,40;50,60".',
)
def compare(n_spectra: int, beta: float, seed: int, output_dir: str, extra_pixels: str) -> None:
    """Run phspectra vs GaussPy+ (Docker) on real GRS spectra and save results."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load FITS
    console.print("Step 1: GRS test field FITS", style="bold cyan")
    header, cube = ensure_fits()
    n_channels, ny, nx = cube.shape
    console.print(f"  Cube: {cube.shape} ({nx * ny} spectra, {n_channels} channels)")

    # Step 2: Catalog
    console.print("\nStep 2: GaussPy+ catalog (pixel selection)", style="bold cyan")
    bounds = fits_bounds(header)
    catalog = ensure_catalog(*bounds)
    if len(catalog) == 0:
        err_console.print("ERROR: no catalog entries")
        sys.exit(1)

    # Step 3: Select spectra
    console.print("\nStep 3: Select spectra", style="bold cyan")
    selected, signals = select_spectra(cube, header, catalog, n_spectra, seed)

    # Merge extra pixels
    if extra_pixels:
        existing = set(selected)
        added = 0
        for pair in extra_pixels.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            parts = pair.split(",")
            px, py = int(parts[0]), int(parts[1])
            if (px, py) in existing:
                console.print(f"  Extra pixel ({px},{py}) already selected, skipping", style="dim")
                continue
            if not (0 <= px < nx and 0 <= py < ny):
                console.print(f"  Extra pixel ({px},{py}) out of bounds, skipping", style="dim")
                continue
            sig = np.nan_to_num(cube[:, py, px].astype(np.float64), nan=0.0)
            selected.append((px, py))
            signals = np.vstack([signals, sig[np.newaxis, :]])
            existing.add((px, py))
            added += 1
        if added:
            console.print(f"  Added {added} extra pixel(s)", style="yellow")

    n_select = len(selected)

    npz_path = os.path.join(output_dir, "spectra.npz")
    np.savez(npz_path, signals=signals)
    console.print(f"  Saved to [blue]{npz_path}[/blue]")

    # Step 4: phspectra
    console.print(f"\nStep 4: phspectra (beta={beta})", style="bold cyan")
    ph_comps_all: list[list[GaussianComponent]] = []
    ph_times: list[float] = []
    for i in range(n_select):
        t0 = time.perf_counter()
        try:
            comps = fit_gaussians(signals[i], beta=beta, max_components=8, mf_snr_min=5.0)
        except (LinAlgError, ValueError):
            comps = []
        ph_times.append(time.perf_counter() - t0)
        ph_comps_all.append(comps)
        if (i + 1) % 100 == 0:
            console.print(f"  {i + 1}/{n_select}", style="dim")

    ph_total = sum(ph_times)
    ph_mean_n = float(np.mean([len(c) for c in ph_comps_all]))
    console.print(
        f"  Done: {ph_total:.1f}s total, "
        f"{ph_total / n_select * 1000:.1f}ms/spectrum, "
        f"mean {ph_mean_n:.1f} components",
        style="green",
    )

    # Save phspectra results
    ph_pixels = [list(p) for p in selected]
    ph_results_dict = {
        "beta": beta,
        "pixels": ph_pixels,
        "amplitudes_fit": [[c.amplitude for c in comps] for comps in ph_comps_all],
        "means_fit": [[c.mean for c in comps] for comps in ph_comps_all],
        "stddevs_fit": [[c.stddev for c in comps] for comps in ph_comps_all],
        "times": ph_times,
        "total_time_s": ph_total,
        "mean_n_components": ph_mean_n,
    }
    ph_json_path = os.path.join(output_dir, "phspectra_results.json")
    with open(ph_json_path, "w", encoding="utf-8") as f:
        json.dump(ph_results_dict, f, indent=2)
    console.print(f"  Saved to [blue]{ph_json_path}[/blue]")

    # Step 5: GaussPy+ (Docker)
    console.print("\nStep 5: GaussPy+ (Docker)", style="bold cyan")
    build_image()
    gp_results = run_gausspyplus(os.path.abspath(output_dir))
    gp_total = gp_results["total_time_s"]
    console.print(
        f"  Done: {gp_total:.1f}s total, "
        f"{gp_total / n_select * 1000:.1f}ms/spectrum, "
        f"mean {gp_results['mean_n_components']:.1f} components",
        style="green",
    )

    # Step 6: Build comparison results and save summary
    console.print("\nStep 6: Build comparison summary", style="bold cyan")
    ph_rms_list: list[float] = []
    gp_rms_list: list[float] = []
    for i in range(n_select):
        signal = signals[i]
        ph_comp_list = [Component(c.amplitude, c.mean, c.stddev) for c in ph_comps_all[i]]
        gp_amps = gp_results["amplitudes_fit"][i]
        gp_means = gp_results["means_fit"][i]
        gp_stds = gp_results["stddevs_fit"][i]
        gp_comp_list = [Component(a, m, s) for a, m, s in zip(gp_amps, gp_means, gp_stds)]
        ph_rms_list.append(residual_rms(signal, ph_comp_list))
        gp_rms_list.append(residual_rms(signal, gp_comp_list))

    n_ph_wins = sum(1 for pr, gr in zip(ph_rms_list, gp_rms_list) if pr < gr)
    console.print(
        f"  RMS wins: phspectra {n_ph_wins}/{n_select}, " f"GP+ {n_select - n_ph_wins}/{n_select}",
        style="green",
    )

    speedup = gp_total / ph_total if ph_total > 0 else 0
    summary = {
        "n_spectra": n_select,
        "phspectra": {
            "total_time_s": round(ph_total, 3),
            "mean_time_ms": round(ph_total / n_select * 1000, 1),
            "mean_rms": round(float(np.mean(ph_rms_list)), 6),
            "mean_n_components": round(ph_mean_n, 2),
        },
        "gausspyplus": {
            "total_time_s": round(gp_total, 3),
            "mean_time_ms": round(gp_total / n_select * 1000, 1),
            "mean_rms": round(float(np.mean(gp_rms_list)), 6),
            "mean_n_components": round(gp_results["mean_n_components"], 2),
        },
        "speedup": round(speedup, 2),
        "rms_wins_phspectra": n_ph_wins,
        "rms_wins_gausspyplus": n_select - n_ph_wins,
    }
    json_path = os.path.join(output_dir, "comparison_docker.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    console.print(f"\n  JSON: [blue]{json_path}[/blue]")
    console.print("\nDone.", style="bold green")
