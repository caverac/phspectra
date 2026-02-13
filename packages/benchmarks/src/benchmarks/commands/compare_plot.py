"""``benchmarks compare-plot`` — generate plots from saved comparison data."""

from __future__ import annotations

import json
import os

import click
import numpy as np
from benchmarks._console import console, err_console
from benchmarks._constants import CACHE_DIR
from benchmarks._gaussian import residual_rms
from benchmarks._matching import match_pairs
from benchmarks._plotting import docs_figure, plot_panel
from benchmarks._types import ComparisonResult, Component
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator


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


def _collect_matched_widths(
    results: list[ComparisonResult],
) -> tuple[list[float], list[float]]:
    """Collect matched (phspectra, GaussPy+) width pairs across all results."""
    ph_widths: list[float] = []
    gp_widths: list[float] = []
    for r in results:
        pairs = match_pairs(r.gp_comps, r.ph_comps)
        for gc, pc in pairs:
            gp_widths.append(gc.stddev)
            ph_widths.append(pc.stddev)
    return ph_widths, gp_widths


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


def _style_ax(ax: plt.Axes) -> None:
    """Apply the shared tick/grid style."""
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="minor", length=3, color="gray", direction="in")
    ax.tick_params(which="major", length=6, direction="in")
    ax.tick_params(top=True, right=True, which="both")


@docs_figure("rms-distribution.png")
def _build_rms_hist(results: list[ComparisonResult]) -> Figure:
    """Build the RMS distribution histogram."""
    ph_rms_arr = np.array([r.ph_rms for r in results])
    gp_rms_arr = np.array([r.gp_rms for r in results])

    fig: Figure
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.95)

    upper = max(np.percentile(ph_rms_arr, 99), np.percentile(gp_rms_arr, 99))
    bins = np.linspace(0, upper, 40)
    ax.hist(ph_rms_arr, bins=bins, alpha=0.7, label="phspectra", color="C0")
    ax.hist(gp_rms_arr, bins=bins, alpha=0.7, label="GaussPy+", color="C3")
    ax.set_xlabel("RMS (K)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", frameon=False)
    _style_ax(ax)

    return fig


@docs_figure("rms-scatter.png")
def _build_rms_scatter(results: list[ComparisonResult]) -> Figure:
    """Build the phspectra vs GaussPy+ RMS scatter plot."""
    ph_rms_arr = np.array([r.ph_rms for r in results])
    gp_rms_arr = np.array([r.gp_rms for r in results])
    n_ph_wins = int(np.sum(ph_rms_arr < gp_rms_arr))
    n_select = len(results)

    fig: Figure
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.95)

    ax.scatter(gp_rms_arr, ph_rms_arr, s=8, alpha=0.5, color="0.3")
    lim = max(ph_rms_arr.max(), gp_rms_arr.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("GaussPy+ RMS (K)")
    ax.set_ylabel("phspectra RMS (K)")
    ax.set_aspect("equal")
    ax.legend(
        [f"PH lower: {n_ph_wins}/{n_select}"],
        loc="upper left",
        frameon=False,
    )
    _style_ax(ax)

    return fig


@docs_figure("compare-disagreements.png")
def _build_disagreements_figure(
    disagreements: list[tuple[str, ComparisonResult]],
) -> Figure:
    """Build the 6-panel disagreement cases figure."""
    fig: Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.07, top=0.95, wspace=0.08, hspace=0.18)

    for i, (label, r) in enumerate(disagreements):
        plot_panel(axes.ravel()[i], r, f"{label} — px({r.pixel[0]},{r.pixel[1]})")
        _style_ax(axes.ravel()[i])
    for i in range(len(disagreements), 6):
        axes.ravel()[i].set_visible(False)

    # Only label outer axes
    for ax in axes[:, 1:].ravel():
        ax.set_ylabel("")
    for ax in axes[0, :].ravel():
        ax.set_xlabel("")

    return fig


@docs_figure("width-comparison.png")
def _build_width_hist(
    ph_widths: list[float],
    gp_widths: list[float],
) -> Figure:
    """Build the log-width-ratio histogram."""
    ph_w = np.array(ph_widths)
    gp_w = np.array(gp_widths)
    log_ratios = np.log(ph_w / np.maximum(gp_w, 0.1))

    fig: Figure
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.95)

    bins = np.linspace(
        float(np.percentile(log_ratios, 1)),
        float(np.percentile(log_ratios, 99)),
        50,
    )
    ax.hist(log_ratios, bins=bins, alpha=0.7, color="C0", edgecolor="white")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$\ln(\sigma_{\mathrm{phspectra}}\;/\;\sigma_{\mathrm{GaussPy+}})$")
    ax.set_ylabel("Count")
    _style_ax(ax)

    return fig


@click.command("compare-plot")
@click.option(
    "--data-dir",
    default=os.path.join(CACHE_DIR, "compare-docker"),
    show_default=True,
    help="Directory containing spectra.npz, results.json, and phspectra_results.json.",
)
def compare_plot(data_dir: str) -> None:
    """Generate comparison plots from saved phspectra vs GaussPy+ data."""
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

    # Disagreements (saved to docs via decorator)
    console.print("\nPlot 1: Disagreement cases", style="bold cyan")
    disagreements = _select_disagreement_cases(results)
    _build_disagreements_figure(disagreements)

    # Width comparison
    console.print("\nPlot 2: Width comparison", style="bold cyan")
    ph_widths, gp_widths = _collect_matched_widths(results)
    console.print(f"  {len(ph_widths)} matched component pairs", style="green")
    _build_width_hist(ph_widths, gp_widths)

    # RMS comparison (saved to docs via decorator)
    console.print("\nPlot 3: RMS distribution", style="bold cyan")
    _build_rms_hist(results)
    console.print("Plot 4: RMS scatter", style="bold cyan")
    _build_rms_scatter(results)

    console.print("\nDone.", style="bold green")
