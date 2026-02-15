"""``benchmarks pre-compute`` -- phspectra vs GaussPy+ (Docker) on real GRS spectra.

Selects spectra from the GRS test field, decomposes them with both
phspectra and the GaussPy+ Docker container, and saves component
parameters, RMS values, and timing to a SQLite database for downstream
plotting commands.
"""

from __future__ import annotations

import json
import os
import time

import click
import numpy as np
import numpy.typing as npt
from benchmarks._console import console
from benchmarks._constants import (
    CACHE_DIR,
    DEFAULT_BETA,
    DEFAULT_SEED,
)
from benchmarks._data import ensure_fits
from benchmarks._database import (
    create_db,
    insert_components,
    insert_gausspyplus_run,
    insert_phspectra_run,
    insert_pixels,
)
from benchmarks._docker import build_image, run_gausspyplus
from benchmarks._gaussian import residual_rms
from benchmarks._types import Component
from numpy.linalg import LinAlgError

from phspectra import fit_gaussians
from phspectra._types import GaussianComponent


def _merge_extra_pixels(
    extra_pixels: str,
    selected: list[tuple[int, int]],
    signals: npt.NDArray[np.float64],
    cube: npt.NDArray[np.float64],
    nx: int,
    ny: int,
) -> tuple[list[tuple[int, int]], npt.NDArray[np.float64]]:
    """Parse ``--extra-pixels`` and append any new ones to *selected*."""
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
    return selected, signals


@click.command("pre-compute")
@click.option("--n-spectra", default=None, type=int, help="Number of spectra (default: all eligible).")
@click.option("--beta", default=DEFAULT_BETA, show_default=True)
@click.option("--seed", default=DEFAULT_SEED, show_default=True)
@click.option("--output-dir", default=os.path.join(CACHE_DIR, "compare-docker"), show_default=True)
@click.option(
    "--extra-pixels",
    default="",
    show_default=False,
    help='Force extra pixel coordinates, e.g. "10,20;30,40;50,60".',
)
def pre_compute(
    n_spectra: int | None,
    beta: float,
    seed: int,
    output_dir: str,
    extra_pixels: str,
) -> None:
    """Run phspectra vs GaussPy+ (Docker) on real GRS spectra and save results."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load FITS
    console.print("Step 1: GRS test field FITS", style="bold cyan")
    _, cube = ensure_fits()
    n_channels, ny, nx = cube.shape
    console.print(f"  Cube: {cube.shape} ({nx * ny} spectra, {n_channels} channels)")

    # Step 2: Select spectra
    console.print("\nStep 2: Select spectra", style="bold cyan")
    all_pixels = [(px, py) for py in range(ny) for px in range(nx)]

    if n_spectra is None:
        selected = all_pixels
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(all_pixels), size=min(n_spectra, len(all_pixels)), replace=False)
        selected = [all_pixels[i] for i in sorted(indices)]

    signals_list = []
    for px, py in selected:
        signals_list.append(np.nan_to_num(cube[:, py, px].astype(np.float64), nan=0.0))
    signals = np.array(signals_list)
    console.print(f"  Using {len(selected)}/{nx * ny} pixels")

    if extra_pixels:
        selected, signals = _merge_extra_pixels(
            extra_pixels,
            selected,
            signals,
            cube,
            nx,
            ny,
        )

    n_select = len(selected)

    npz_path = os.path.join(output_dir, "spectra.npz")
    np.savez(npz_path, signals=signals)
    console.print(f"  Saved to [blue]{npz_path}[/blue]")

    # Step 3: phspectra
    console.print(f"\nStep 3: phspectra (beta={beta})", style="bold cyan")
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

    # Step 4: GaussPy+ (Docker)
    console.print("\nStep 4: GaussPy+ (Docker)", style="bold cyan")
    build_image()
    gp_results = run_gausspyplus(os.path.abspath(output_dir))
    gp_total = gp_results["total_time_s"]
    console.print(
        f"  Done: {gp_total:.1f}s total, "
        f"{gp_total / n_select * 1000:.1f}ms/spectrum, "
        f"mean {gp_results['mean_n_components']:.1f} components",
        style="green",
    )

    # Step 5: Build comparison and write SQLite
    console.print("\nStep 5: Write results to SQLite", style="bold cyan")
    db_path = os.path.join(output_dir, "pre-compute.db")
    conn = create_db(db_path)

    # phspectra run
    ph_run_id = insert_phspectra_run(conn, beta, n_select, ph_total)

    ph_rms_list: list[float] = []
    ph_pixel_rows: list[tuple[int, int, int, float, float]] = []
    for i in range(n_select):
        signal = signals[i]
        px, py = selected[i]
        ph_comp_list = [Component(c.amplitude, c.mean, c.stddev) for c in ph_comps_all[i]]
        rms = residual_rms(signal, ph_comp_list)
        ph_rms_list.append(rms)
        insert_components(
            conn,
            "phspectra_components",
            ph_run_id,
            px,
            py,
            [(c.amplitude, c.mean, c.stddev) for c in ph_comps_all[i]],
        )
        ph_pixel_rows.append((px, py, len(ph_comps_all[i]), rms, ph_times[i]))

    insert_pixels(conn, "phspectra_pixels", ph_run_id, ph_pixel_rows)

    # gausspyplus run
    gp_run_id = insert_gausspyplus_run(
        conn,
        alpha1=gp_results.get("alpha1", 2.89),
        alpha2=gp_results.get("alpha2", 6.65),
        phase=gp_results.get("phase", "two"),
        n_spectra=n_select,
        total_time_s=gp_total,
    )

    gp_rms_list: list[float] = []
    gp_pixel_rows: list[tuple[int, int, int, float, float]] = []
    for i in range(n_select):
        signal = signals[i]
        px, py = selected[i]
        gp_amps = gp_results["amplitudes_fit"][i]
        gp_means = gp_results["means_fit"][i]
        gp_stds = gp_results["stddevs_fit"][i]
        gp_comp_list = [Component(a, m, s) for a, m, s in zip(gp_amps, gp_means, gp_stds)]
        rms = residual_rms(signal, gp_comp_list)
        gp_rms_list.append(rms)
        insert_components(
            conn,
            "gausspyplus_components",
            gp_run_id,
            px,
            py,
            list(zip(gp_amps, gp_means, gp_stds)),
        )
        gp_time = gp_results["times"][i] if i < len(gp_results["times"]) else 0.0
        gp_pixel_rows.append((px, py, len(gp_amps), rms, gp_time))

    insert_pixels(conn, "gausspyplus_pixels", gp_run_id, gp_pixel_rows)
    conn.commit()
    conn.close()
    console.print(f"  SQLite: [blue]{db_path}[/blue]")

    # Summary stats
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
    console.print(f"  JSON: [blue]{json_path}[/blue]")
    console.print("\nDone.", style="bold green")
