"""``benchmarks compare-plot`` — generate plots from saved comparison data."""

from __future__ import annotations

import json
import os

import click
import numpy as np
from matplotlib import pyplot as plt

from benchmarks._console import console, err_console
from benchmarks._constants import CACHE_DIR
from benchmarks._gaussian import residual_rms
from benchmarks._matching import match_pairs
from benchmarks._plotting import plot_panel
from benchmarks._types import ComparisonResult, Component


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


def _load_results(
    data_dir: str,
) -> tuple[list[ComparisonResult], dict[str, object]]:
    """Load saved comparison data and reconstruct ComparisonResult objects."""
    spectra_path = os.path.join(data_dir, "spectra.npz")
    ph_path = os.path.join(data_dir, "phspectra_results.json")
    gp_path = os.path.join(data_dir, "results.json")

    for path, label in [
        (spectra_path, "spectra.npz"),
        (ph_path, "phspectra_results.json"),
        (gp_path, "results.json"),
    ]:
        if not os.path.exists(path):
            err_console.print(f"ERROR: missing {label} in {data_dir}")
            raise SystemExit(1)

    signals = np.load(spectra_path)["signals"]

    with open(ph_path, encoding="utf-8") as f:
        ph_data = json.load(f)
    with open(gp_path, encoding="utf-8") as f:
        gp_data = json.load(f)

    n_spectra = len(signals)
    results: list[ComparisonResult] = []
    for i in range(n_spectra):
        signal = signals[i]
        pixel = tuple(ph_data["pixels"][i])

        ph_comps = [
            Component(a, m, s)
            for a, m, s in zip(
                ph_data["amplitudes_fit"][i],
                ph_data["means_fit"][i],
                ph_data["stddevs_fit"][i],
            )
        ]
        gp_comps = [
            Component(a, m, s)
            for a, m, s in zip(
                gp_data["amplitudes_fit"][i],
                gp_data["means_fit"][i],
                gp_data["stddevs_fit"][i],
            )
        ]

        ph_rms = residual_rms(signal, ph_comps)
        gp_rms = residual_rms(signal, gp_comps)
        ph_time = ph_data["times"][i]
        gp_time = gp_data["times"][i] if i < len(gp_data["times"]) else 0.0

        results.append(
            ComparisonResult(
                pixel=pixel,
                signal=signal,
                gp_comps=gp_comps,
                ph_comps=ph_comps,
                ph_rms=ph_rms,
                gp_rms=gp_rms,
                ph_time=ph_time,
                gp_time=gp_time,
            )
        )

    summary = {
        "ph_total_time": ph_data["total_time_s"],
        "gp_total_time": gp_data["total_time_s"],
        "ph_mean_n_components": ph_data["mean_n_components"],
        "gp_mean_n_components": gp_data["mean_n_components"],
    }
    return results, summary


@click.command("compare-plot")
@click.option(
    "--data-dir",
    default=os.path.join(CACHE_DIR, "compare-docker"),
    show_default=True,
    help="Directory containing spectra.npz, results.json, and phspectra_results.json.",
)
@click.option(
    "--output-dir",
    default=None,
    show_default=True,
    help="Directory for plot PNGs (defaults to --data-dir).",
)
def compare_plot(data_dir: str, output_dir: str | None) -> None:
    """Generate comparison plots from saved phspectra vs GaussPy+ data."""
    if output_dir is None:
        output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)

    console.print("Loading comparison data...", style="bold cyan")
    results, summary = _load_results(data_dir)
    n_select = len(results)
    console.print(f"  Loaded {n_select} spectra from [blue]{data_dir}[/blue]")

    # Summary stats
    n_ph_wins = sum(1 for r in results if r.ph_rms < r.gp_rms)
    ph_total = summary["ph_total_time"]
    gp_total = summary["gp_total_time"]
    speedup = gp_total / ph_total if ph_total > 0 else 0
    console.print(
        f"  RMS wins: phspectra {n_ph_wins}/{n_select}, " f"GP+ {n_select - n_ph_wins}/{n_select}",
        style="green",
    )
    console.print(
        f"  Timing: phspectra {ph_total:.1f}s, GP+ {gp_total:.1f}s ({speedup:.1f}x speedup)",
        style="green",
    )

    # Disagreements
    console.print("\nPlot 1: Disagreement cases", style="bold cyan")
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
    path = os.path.join(output_dir, "compare-disagreements-docker.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    console.print(f"  Saved [blue]{path}[/blue]")

    # Narrower widths
    console.print("Plot 2: Narrower width cases", style="bold cyan")
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
    path = os.path.join(output_dir, "compare-narrower-widths-docker.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    console.print(f"  Saved [blue]{path}[/blue]")

    # RMS histogram
    console.print("Plot 3: RMS comparison", style="bold cyan")
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
    axes[1].set_title(f"PH lower RMS: {n_ph_wins}/{n_select}")
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
        f"RMS comparison: phspectra vs GaussPy+ (Docker) — {n_select} spectra",
        fontsize=13,
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "compare-rms-docker.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    console.print(f"  Saved [blue]{path}[/blue]")

    console.print("\nDone.", style="bold green")
