"""``benchmarks compare`` — phspectra vs GaussPy+ (Docker) on real GRS spectra."""

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
from benchmarks._matching import match_pairs
from benchmarks._plotting import plot_panel
from benchmarks._types import ComparisonResult, Component
from numpy.linalg import LinAlgError

from phspectra import fit_gaussians
from phspectra._types import GaussianComponent


def _select_disagreement_cases(
    results: list[ComparisonResult],
) -> list[tuple[str, ComparisonResult]]:
    """Pick up to 6 interesting disagreement cases."""
    cases: list[tuple[str, ComparisonResult]] = []
    used: set[tuple[int, int]] = set()

    def _pick(label: str, candidates: list[ComparisonResult]) -> bool:
        for r in candidates:
            if r.pixel not in used:
                cases.append((label, r))
                used.add(r.pixel)
                return True
        return False

    fewer = [r for r in results if len(r.ph_comps) < len(r.gp_comps)]
    fewer.sort(key=lambda r: len(r.gp_comps) - len(r.ph_comps), reverse=True)
    _pick("PH fewer components", fewer)

    more = [r for r in results if len(r.ph_comps) > len(r.gp_comps)]
    more.sort(key=lambda r: len(r.ph_comps) - len(r.gp_comps), reverse=True)
    _pick("PH more components", more)

    better_rms = [r for r in results if r.ph_rms < r.gp_rms * 0.9]
    better_rms.sort(key=lambda r: r.gp_rms - r.ph_rms, reverse=True)
    _pick("PH lower RMS", better_rms)

    worse_rms = [r for r in results if r.gp_rms < r.ph_rms * 0.9]
    worse_rms.sort(key=lambda r: r.ph_rms - r.gp_rms, reverse=True)
    _pick("GP+ lower RMS", worse_rms)

    same_n = [r for r in results if len(r.ph_comps) == len(r.gp_comps) and len(r.ph_comps) > 1]
    same_n_diff_pos = []
    for r in same_n:
        pairs = match_pairs(r.gp_comps, r.ph_comps)
        if pairs:
            max_pos_diff = max(abs(gc.mean - pc.mean) for gc, pc in pairs)
            if max_pos_diff > 2.0:
                same_n_diff_pos.append((max_pos_diff, r))
    same_n_diff_pos.sort(key=lambda x: x[0], reverse=True)
    _pick("Same N, different positions", [x[1] for x in same_n_diff_pos])

    width_diff_cases = []
    for r in results:
        pairs = match_pairs(r.gp_comps, r.ph_comps)
        if pairs:
            max_ratio = max(gc.stddev / max(pc.stddev, 0.1) for gc, pc in pairs)
            if max_ratio > 1.5:
                width_diff_cases.append((max_ratio, r))
    width_diff_cases.sort(key=lambda x: x[0], reverse=True)
    _pick("Different widths", [x[1] for x in width_diff_cases])

    if len(cases) < 6:
        rms_diff = [(abs(r.ph_rms - r.gp_rms), r) for r in results]
        rms_diff.sort(key=lambda x: x[0], reverse=True)
        for _, r in rms_diff:
            if r.pixel not in used:
                cases.append(("Notable difference", r))
                used.add(r.pixel)
                if len(cases) >= 6:
                    break

    return cases[:6]


def _select_narrower_width_cases(
    results: list[ComparisonResult],
    n: int = 6,
) -> list[tuple[float, ComparisonResult]]:
    """Select up to *n* cases where GP+ fits wider than phspectra."""
    scored: list[tuple[float, ComparisonResult]] = []
    for r in results:
        pairs = match_pairs(r.gp_comps, r.ph_comps)
        if not pairs:
            continue
        ratios = [gc.stddev / max(pc.stddev, 0.1) for gc, pc in pairs]
        avg_ratio = float(np.mean(ratios))
        if avg_ratio > 1.3:
            scored.append((avg_ratio, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:n]


@click.command()
@click.option("--n-spectra", default=DEFAULT_N_SPECTRA, show_default=True)
@click.option("--beta", default=DEFAULT_BETA, show_default=True)
@click.option("--seed", default=DEFAULT_SEED, show_default=True)
@click.option("--output-dir", default=os.path.join(CACHE_DIR, "compare-docker"), show_default=True)
def compare(n_spectra: int, beta: float, seed: int, output_dir: str) -> None:
    """Compare phspectra vs GaussPy+ (Docker) on real GRS spectra."""
    import matplotlib.pyplot as plt

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
            comps = fit_gaussians(signals[i], beta=beta, max_components=8, sig_min=4.0)
        except (LinAlgError, ValueError):
            comps = []
        ph_times.append(time.perf_counter() - t0)
        ph_comps_all.append(comps)
        if (i + 1) % 100 == 0:
            console.print(f"  {i + 1}/{n_select}", style="dim")

    ph_total = sum(ph_times)
    console.print(
        f"  Done: {ph_total:.1f}s total, "
        f"{ph_total / n_select * 1000:.1f}ms/spectrum, "
        f"mean {np.mean([len(c) for c in ph_comps_all]):.1f} components",
        style="green",
    )

    # Step 5: GaussPy+ (Docker)
    console.print("\nStep 5: GaussPy+ (Docker)", style="bold cyan")
    build_image()
    gp_results = run_gausspyplus(os.path.abspath(output_dir))
    gp_times = gp_results["times"]
    gp_total = gp_results["total_time_s"]
    console.print(
        f"  Done: {gp_total:.1f}s total, "
        f"{gp_total / n_select * 1000:.1f}ms/spectrum, "
        f"mean {gp_results['mean_n_components']:.1f} components",
        style="green",
    )

    # Step 6: Build comparison results
    console.print("\nStep 6: Build comparison results", style="bold cyan")
    results: list[ComparisonResult] = []

    for i in range(n_select):
        signal = signals[i]
        pixel = selected[i]
        ph_comp_list = [Component(c.amplitude, c.mean, c.stddev) for c in ph_comps_all[i]]
        gp_amps = gp_results["amplitudes_fit"][i]
        gp_means = gp_results["means_fit"][i]
        gp_stds = gp_results["stddevs_fit"][i]
        gp_comp_list = [Component(a, m, s) for a, m, s in zip(gp_amps, gp_means, gp_stds)]
        ph_rms = residual_rms(signal, ph_comp_list)
        gp_rms = residual_rms(signal, gp_comp_list)
        results.append(
            ComparisonResult(
                pixel=pixel,
                signal=signal,
                gp_comps=gp_comp_list,
                ph_comps=ph_comp_list,
                ph_rms=ph_rms,
                gp_rms=gp_rms,
                ph_time=ph_times[i],
                gp_time=gp_times[i] if i < len(gp_times) else 0.0,
            )
        )

    # Summary
    n_ph_wins = sum(1 for r in results if r.ph_rms < r.gp_rms)
    console.print(
        f"  RMS wins: phspectra {n_ph_wins}/{len(results)}, "
        f"GP+ {len(results) - n_ph_wins}/{len(results)}",
        style="green",
    )

    # Step 7: Plots
    console.print("\nStep 7: Generate plots", style="bold cyan")

    # Disagreements
    disagreements = _select_disagreement_cases(results)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, (label, r) in enumerate(disagreements):
        plot_panel(axes.ravel()[i], r, f"{label} — px({r.pixel[0]},{r.pixel[1]})")
    for i in range(len(disagreements), 6):
        axes.ravel()[i].set_visible(False)
    fig.suptitle(
        f"phspectra vs GaussPy+ (Docker) — disagreement cases (N={n_select})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "compare-disagreements-docker.png"), dpi=150)
    plt.close(fig)

    # Narrower widths
    narrower = _select_narrower_width_cases(results)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, (ratio, r) in enumerate(narrower):
        plot_panel(
            axes.ravel()[i],
            r,
            f"GP+/PH width ratio={ratio:.1f}x — px({r.pixel[0]},{r.pixel[1]})",
        )
    for i in range(len(narrower), 6):
        axes.ravel()[i].set_visible(False)
    fig.suptitle(
        f"phspectra predicts narrower widths than GaussPy+ (Docker, N={n_select})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "compare-narrower-widths-docker.png"), dpi=150)
    plt.close(fig)

    # RMS histogram
    ph_rms_arr = np.array([r.ph_rms for r in results])
    gp_rms_arr = np.array([r.gp_rms for r in results])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bins = np.linspace(0, max(np.percentile(ph_rms_arr, 99), np.percentile(gp_rms_arr, 99)), 40)
    axes[0].hist(ph_rms_arr, bins=bins, alpha=0.7, label="phspectra", color="C0")
    axes[0].hist(gp_rms_arr, bins=bins, alpha=0.7, label="GaussPy+", color="C3")
    axes[0].set_xlabel("RMS (K)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("RMS distribution")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(gp_rms_arr, ph_rms_arr, s=8, alpha=0.5, color="0.3")
    lim = max(ph_rms_arr.max(), gp_rms_arr.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("GaussPy+ RMS (K)")
    axes[1].set_ylabel("phspectra RMS (K)")
    axes[1].set_title(f"PH lower RMS: {n_ph_wins}/{len(results)}")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)
    diff = gp_rms_arr - ph_rms_arr
    axes[2].hist(diff, bins=40, alpha=0.8, color="C2", edgecolor="white")
    axes[2].axvline(0, color="k", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("RMS(GP+) - RMS(PH) (K)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("RMS difference (positive = phspectra wins)")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(
        f"RMS comparison: phspectra vs GaussPy+ (Docker) — {len(results)} spectra",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "compare-rms-docker.png"), dpi=150)
    plt.close(fig)

    # Save JSON summary
    speedup = gp_total / ph_total if ph_total > 0 else 0
    summary = {
        "n_spectra": n_select,
        "phspectra": {
            "total_time_s": round(ph_total, 3),
            "mean_time_ms": round(ph_total / n_select * 1000, 1),
            "mean_rms": round(float(np.mean(ph_rms_arr)), 6),
            "mean_n_components": round(float(np.mean([len(r.ph_comps) for r in results])), 2),
        },
        "gausspyplus": {
            "total_time_s": round(gp_total, 3),
            "mean_time_ms": round(gp_total / n_select * 1000, 1),
            "mean_rms": round(float(np.mean(gp_rms_arr)), 6),
            "mean_n_components": round(gp_results["mean_n_components"], 2),
        },
        "speedup": round(speedup, 2),
        "rms_wins_phspectra": n_ph_wins,
        "rms_wins_gausspyplus": len(results) - n_ph_wins,
    }
    json_path = os.path.join(output_dir, "comparison_docker.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    console.print(f"\n  JSON: [blue]{json_path}[/blue]")
    console.print("\nDone.", style="bold green")
