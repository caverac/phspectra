"""``benchmarks compare-plot`` -- generate plots from saved comparison data.

Reads the SQLite/NPZ outputs of ``benchmarks pre-compute`` and produces four
figures: RMS distribution histogram, RMS scatter, six-panel disagreement
cases, and matched-width ratio histogram.  All are saved to the docs
static image directory via the ``@docs_figure`` decorator.
"""

from __future__ import annotations

import os

import click
import numpy as np
from benchmarks._console import console, err_console
from benchmarks._constants import CACHE_DIR
from benchmarks._database import load_components, load_pixels, load_run
from benchmarks._gaussian import residual_rms
from benchmarks._matching import match_pairs
from benchmarks._plotting import configure_axes, docs_figure, plot_panel
from benchmarks._types import ComparisonResult, ComparisonSummary, Component
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch


def _select_disagreement_cases(
    results: list[ComparisonResult],
) -> list[tuple[str, ComparisonResult]]:
    """Select up to six spectra where phspectra and GaussPy+ disagree.

    Cases are chosen to cover different disagreement types: fewer/more
    components, lower/higher RMS, different positions, different widths.

    Parameters
    ----------
    results : list[ComparisonResult]
        All comparison results.

    Returns
    -------
    list[tuple[str, ComparisonResult]]
        At most six ``(label, result)`` pairs.
    """
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
    """Collect matched (phspectra, GaussPy+) width pairs across all results.

    Components are matched using position tolerance from
    ``match_pairs``.

    Parameters
    ----------
    results : list[ComparisonResult]
        All comparison results.

    Returns
    -------
    tuple[list[float], list[float]]
        ``(ph_widths, gp_widths)`` parallel lists of matched stddev
        values.
    """
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
) -> tuple[list[ComparisonResult], ComparisonSummary]:
    """Load saved comparison data from SQLite and NPZ.

    Parameters
    ----------
    data_dir : str
        Directory containing the saved benchmark files.

    Returns
    -------
    tuple[list[ComparisonResult], ComparisonSummary]
        ``(results, summary)``
    """
    spectra_path = os.path.join(data_dir, "spectra.npz")
    db_path = os.path.join(data_dir, "pre-compute.db")

    for path, label in [
        (spectra_path, "spectra.npz"),
        (db_path, "pre-compute.db"),
    ]:
        if not os.path.exists(path):
            err_console.print(f"ERROR: missing {label} in {data_dir}")
            raise SystemExit(1)

    npz = np.load(spectra_path)
    signals = npz["signals"]
    npz_pixels = npz["pixels"]  # (N, 2) array saved by pre-compute
    pixel_to_signal_idx = {(int(r[0]), int(r[1])): i for i, r in enumerate(npz_pixels)}

    ph_run = load_run(db_path, "phspectra")
    gp_run = load_run(db_path, "gausspyplus")

    ph_comp_map = load_components(db_path, "phspectra")
    gp_comp_map = load_components(db_path, "gausspyplus")

    ph_pixel_rows = load_pixels(db_path, "phspectra")
    gp_pixel_rows = load_pixels(db_path, "gausspyplus")

    pixel_order = [(r["xpos"], r["ypos"]) for r in ph_pixel_rows]
    ph_time_map = {(r["xpos"], r["ypos"]): r["time_s"] for r in ph_pixel_rows}
    gp_time_map = {(r["xpos"], r["ypos"]): r["time_s"] for r in gp_pixel_rows}

    results: list[ComparisonResult] = []
    for px, py in pixel_order:
        signal = signals[pixel_to_signal_idx[(px, py)]]

        ph_comps = [Component(a, m, s) for a, m, s in ph_comp_map.get((px, py), [])]
        gp_comps = [Component(a, m, s) for a, m, s in gp_comp_map.get((px, py), [])]

        ph_rms = residual_rms(signal, ph_comps)
        gp_rms = residual_rms(signal, gp_comps)
        ph_time = ph_time_map.get((px, py), 0.0)
        gp_time = gp_time_map.get((px, py), 0.0)

        results.append(
            ComparisonResult(
                pixel=(px, py),
                signal=signal,
                gp_comps=gp_comps,
                ph_comps=ph_comps,
                ph_rms=ph_rms,
                gp_rms=gp_rms,
                ph_time=ph_time,
                gp_time=gp_time,
            )
        )

    summary = ComparisonSummary(
        ph_total_time=ph_run["total_time_s"],
        gp_total_time=gp_run["total_time_s"],
        ph_mean_n_components=float(np.mean([len(r.ph_comps) for r in results])),
        gp_mean_n_components=float(np.mean([len(r.gp_comps) for r in results])),
    )
    return results, summary


@docs_figure("rms-distribution.png")
def _build_rms_hist(results: list[ComparisonResult]) -> Figure:
    """Build overlaid histograms of per-spectrum RMS for both methods.

    Parameters
    ----------
    results : list[ComparisonResult]
        All comparison results.

    Returns
    -------
    Figure
        Single-axes matplotlib figure.
    """
    ph_rms_arr = np.array([r.ph_rms for r in results])
    gp_rms_arr = np.array([r.gp_rms for r in results])

    fig: Figure
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.95)

    upper = max(np.percentile(ph_rms_arr, 99), np.percentile(gp_rms_arr, 99))
    bins_ph = np.linspace(0, upper, 50)
    bins_gp = np.linspace(0, upper, 51)
    ph_counts, ph_edges = np.histogram(ph_rms_arr, bins=bins_ph)
    gp_counts, gp_edges = np.histogram(gp_rms_arr, bins=bins_gp)
    ax.stairs(ph_counts, ph_edges, color="k", linewidth=1.2, linestyle="-")
    ax.stairs(gp_counts, gp_edges, color="k", linewidth=1.2, linestyle="--")

    ax.set_xlabel("RMS (K)")
    ax.set_ylabel("Count")
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor="k", linewidth=1.2, linestyle="-", label="PHSpectra"),
            Patch(facecolor="none", edgecolor="k", linewidth=1.2, linestyle="--", label="GaussPy+"),
        ],
        loc="upper right",
        frameon=False,
    )
    configure_axes(ax)

    return fig


@docs_figure("rms-scatter.png")
def _build_rms_scatter(results: list[ComparisonResult]) -> Figure:
    """Build a phspectra-vs-GaussPy+ RMS scatter plot.

    Points below the diagonal indicate phspectra achieves lower RMS.

    Parameters
    ----------
    results : list[ComparisonResult]
        All comparison results.

    Returns
    -------
    Figure
        Single-axes matplotlib figure.
    """
    ph_rms_arr = np.array([r.ph_rms for r in results])
    gp_rms_arr = np.array([r.gp_rms for r in results])

    fig: Figure
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.95)

    ax.scatter(gp_rms_arr, ph_rms_arr, s=8, alpha=0.5, color="0.3")
    lim = max(ph_rms_arr.max(), gp_rms_arr.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("GaussPy+ RMS (K)")
    ax.set_ylabel("PHSpectra RMS (K)")
    ax.set_aspect("equal")
    configure_axes(ax)

    return fig


@docs_figure("compare-disagreements.png")
def _build_disagreements_figure(
    disagreements: list[tuple[str, ComparisonResult]],
) -> Figure:
    """Build a 2x3 grid of spectra where the two methods disagree.

    Each panel shows the raw data, phspectra model, and GaussPy+ model
    with RMS and component counts in the legend.

    Parameters
    ----------
    disagreements : list[tuple[str, ComparisonResult]]
        Up to six ``(label, result)`` pairs from
        ``_select_disagreement_cases``.

    Returns
    -------
    Figure
        Full-width 2x3 matplotlib figure.
    """
    fig: Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.07, top=0.95, wspace=0.08, hspace=0.08)

    for i, (label, r) in enumerate(disagreements):
        plot_panel(axes.ravel()[i], r, f"{label} - px({r.pixel[0]},{r.pixel[1]})")
        configure_axes(axes.ravel()[i])
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
    """Build a histogram of log(sigma_ph / sigma_gp) for matched components.

    A distribution centred on zero indicates no systematic width bias
    between the two methods.

    Parameters
    ----------
    ph_widths : list[float]
        Matched phspectra component standard deviations.
    gp_widths : list[float]
        Matched GaussPy+ component standard deviations.

    Returns
    -------
    Figure
        Single-axes matplotlib figure.
    """
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
    ax.hist(
        log_ratios,
        bins=bins,  # type: ignore[arg-type]
        alpha=0.7,
        color="k",
        edgecolor="white",
    )
    ax.axvline(
        0.0,
        color="k",
        linestyle="--",
        linewidth=0.8,
    )
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"$\ln(\sigma_{\mathrm{PHSpectra}}\;/\;\sigma_{\mathrm{GaussPy+}})$")
    ax.set_ylabel("Count")
    configure_axes(ax)

    return fig


@click.command("compare-plot")
@click.option(
    "--data-dir",
    default=os.path.join(CACHE_DIR, "compare-docker"),
    show_default=True,
    help="Directory containing spectra.npz and pre-compute.db.",
)
def compare_plot(data_dir: str) -> None:
    """Generate comparison plots from saved phspectra vs GaussPy+ data."""
    console.print("Loading comparison data...", style="bold cyan")
    results, summary = _load_results(data_dir)
    n_select = len(results)
    console.print(f"  Loaded {n_select} spectra from [blue]{data_dir}[/blue]")

    # Summary stats
    ph_rms_arr = np.array([r.ph_rms for r in results])
    gp_rms_arr = np.array([r.gp_rms for r in results])
    n_ph_wins = int(np.sum(ph_rms_arr < gp_rms_arr))
    ph_total = summary.ph_total_time
    gp_total = summary.gp_total_time
    speedup = gp_total / ph_total if ph_total > 0 else 0
    console.print(
        f"  Mean RMS: phspectra {ph_rms_arr.mean():.4f} K, GP+ {gp_rms_arr.mean():.4f} K",
        style="green",
    )
    console.print(
        f"  RMS wins: phspectra {n_ph_wins}/{n_select} "
        f"({100 * n_ph_wins / n_select:.0f}%), "
        f"GP+ {n_select - n_ph_wins}/{n_select} "
        f"({100 * (n_select - n_ph_wins) / n_select:.0f}%)",
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
    n_pairs = len(ph_widths)
    ph_w = np.array(ph_widths)
    gp_w = np.array(gp_widths)
    n_ph_wider = int(np.sum(ph_w > gp_w))
    n_gp_wider = int(np.sum(gp_w > ph_w))
    median_log_ratio = float(np.median(np.log(ph_w / np.maximum(gp_w, 0.1))))
    console.print(f"  {n_pairs} matched component pairs", style="green")
    console.print(
        f"  PH wider: {n_ph_wider}/{n_pairs} ({100 * n_ph_wider / n_pairs:.0f}%), "
        f"GP+ wider: {n_gp_wider}/{n_pairs} ({100 * n_gp_wider / n_pairs:.0f}%)",
        style="green",
    )
    console.print(f"  Median ln(sigma_PH / sigma_GP+): {median_log_ratio:.3f}", style="green")
    _build_width_hist(ph_widths, gp_widths)

    # RMS comparison (saved to docs via decorator)
    console.print("\nPlot 3: RMS distribution", style="bold cyan")
    _build_rms_hist(results)
    console.print("Plot 4: RMS scatter", style="bold cyan")
    _build_rms_scatter(results)

    console.print("\nDone.", style="bold green")
