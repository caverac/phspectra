"""``benchmarks synthetic`` -- synthetic benchmark with known ground truth.

Generates seven categories of synthetic spectra (single bright/faint/
narrow/broad, multi separated/blended, crowded), decomposes each with
phspectra over a grid of beta values, and reports F1, precision, recall,
and parameter-recovery errors.

Outputs are written to ``/tmp/phspectra/synthetic-benchmark/`` (CSV,
JSON, and two PNG figures) and to the docs static image directory.
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor  # pylint: disable=no-name-in-module
from dataclasses import asdict, dataclass, fields

import click
import numpy as np
import numpy.typing as npt
from benchmarks._console import console
from rich.table import Table
from benchmarks._constants import MEAN_MARGIN, N_CHANNELS, NOISE_SIGMA
from benchmarks._gaussian import gaussian, gaussian_model
from benchmarks._matching import count_correct_matches, f1_score, match_pairs
from benchmarks._plotting import AxesGrid1D, configure_axes, docs_figure
from benchmarks._types import Component, SyntheticSpectrum
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.linalg import LinAlgError

from phspectra import fit_gaussians
from phspectra.quality import aicc


@dataclass
class _WorkItem:
    """Input bundle for a single (spectrum, beta) evaluation."""

    signal: npt.NDArray[np.float64]
    category: str
    spec_idx: int
    true_comps_raw: list[dict[str, float]]
    beta: float


@dataclass
class _EvalResult:
    """Output of a single (spectrum, beta) evaluation."""

    category: str
    spectrum_idx: int
    beta: float
    f1: float
    precision: float
    recall: float
    n_correct: int
    n_true: int
    n_detected: int
    n_guessed: int
    count_error: int
    residual_rms: float
    aicc: float | None
    amp_log_ratio_mean: float | None
    pos_log_ratio_mean: float | None
    width_log_ratio_mean: float | None


CATEGORY_LABELS: dict[str, str] = {
    "single_bright": "SB",
    "single_faint": "SF",
    "single_narrow": "SN",
    "single_broad": "SBd",
    "multi_separated": "MS",
    "multi_blended": "MB",
    "crowded": "C",
}


def _make_spectrum(
    rng: np.random.Generator,
    components: list[Component],
) -> npt.NDArray[np.float64]:
    """Build a noisy signal from ground-truth Gaussian components.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for additive noise.
    components : list[Component]
        Ground-truth Gaussian parameters (amplitude, mean, stddev).

    Returns
    -------
    npt.NDArray[np.float64]
        Synthetic spectrum of length ``N_CHANNELS`` with Gaussian noise
        of standard deviation ``NOISE_SIGMA``.
    """
    x = np.arange(N_CHANNELS, dtype=np.float64)
    signal = np.zeros(N_CHANNELS, dtype=np.float64)
    for c in components:
        signal += gaussian(x, c.amplitude, c.mean, c.stddev)
    signal += rng.normal(0.0, NOISE_SIGMA, N_CHANNELS)
    return signal


def _rand_mean(rng: np.random.Generator) -> float:
    return float(rng.uniform(MEAN_MARGIN, N_CHANNELS - MEAN_MARGIN))


def _gen_single_bright(rng: np.random.Generator, n: int) -> list[SyntheticSpectrum]:
    spectra = []
    for i in range(n):
        amp = rng.uniform(1.0, 5.0)
        stddev = rng.uniform(3.0, 10.0)
        while amp / NOISE_SIGMA < 7.0:
            amp = rng.uniform(1.0, 5.0)
        comp = Component(amplitude=float(amp), mean=_rand_mean(rng), stddev=float(stddev))
        spectra.append(SyntheticSpectrum("single_bright", i, _make_spectrum(rng, [comp]), [comp]))
    return spectra


def _gen_single_faint(rng: np.random.Generator, n: int) -> list[SyntheticSpectrum]:
    spectra = []
    for i in range(n):
        snr_target = rng.uniform(2.0, 6.0)
        amp = float(np.clip(snr_target * NOISE_SIGMA, 0.3, 0.8))
        stddev = rng.uniform(3.0, 10.0)
        comp = Component(amplitude=amp, mean=_rand_mean(rng), stddev=float(stddev))
        spectra.append(SyntheticSpectrum("single_faint", i, _make_spectrum(rng, [comp]), [comp]))
    return spectra


def _gen_single_narrow(rng: np.random.Generator, n: int) -> list[SyntheticSpectrum]:
    spectra = []
    for i in range(n):
        amp = rng.uniform(1.0, 5.0)
        stddev = rng.uniform(1.0, 2.5)
        comp = Component(amplitude=float(amp), mean=_rand_mean(rng), stddev=float(stddev))
        spectra.append(SyntheticSpectrum("single_narrow", i, _make_spectrum(rng, [comp]), [comp]))
    return spectra


def _gen_single_broad(rng: np.random.Generator, n: int) -> list[SyntheticSpectrum]:
    spectra = []
    for i in range(n):
        amp = rng.uniform(0.5, 3.0)
        stddev = rng.uniform(10.0, 20.0)
        comp = Component(amplitude=float(amp), mean=_rand_mean(rng), stddev=float(stddev))
        spectra.append(SyntheticSpectrum("single_broad", i, _make_spectrum(rng, [comp]), [comp]))
    return spectra


def _gen_multi_separated(rng: np.random.Generator, n: int) -> list[SyntheticSpectrum]:
    spectra = []
    for i in range(n):
        n_comp = int(rng.integers(2, 4))
        comps: list[Component] = []
        for _ in range(n_comp):
            amp = rng.uniform(0.5, 4.0)
            stddev = rng.uniform(2.0, 8.0)
            mean = 0.0
            for _attempt in range(200):
                mean = _rand_mean(rng)
                if all(abs(mean - c.mean) > 4.0 * max(stddev, c.stddev) for c in comps):
                    break
            comps.append(Component(float(amp), mean, float(stddev)))
        spectra.append(SyntheticSpectrum("multi_separated", i, _make_spectrum(rng, comps), comps))
    return spectra


def _gen_multi_blended(rng: np.random.Generator, n: int) -> list[SyntheticSpectrum]:
    spectra = []
    for i in range(n):
        n_comp = int(rng.integers(2, 4))
        comps: list[Component] = []
        for k in range(n_comp):
            amp = rng.uniform(0.5, 4.0)
            stddev = rng.uniform(3.0, 8.0)
            if k == 0:
                mean = _rand_mean(rng)
            else:
                ref = comps[-1]
                sep_sigma = rng.uniform(1.5, 3.0)
                offset = sep_sigma * max(stddev, ref.stddev)
                direction = rng.choice([-1.0, 1.0])
                mean = float(
                    np.clip(ref.mean + direction * offset, MEAN_MARGIN, N_CHANNELS - MEAN_MARGIN)
                )
            comps.append(Component(float(amp), mean, float(stddev)))
        spectra.append(SyntheticSpectrum("multi_blended", i, _make_spectrum(rng, comps), comps))
    return spectra


def _gen_crowded(rng: np.random.Generator, n: int) -> list[SyntheticSpectrum]:
    spectra = []
    for i in range(n):
        n_comp = int(rng.integers(4, 6))
        comps: list[Component] = []
        for _ in range(n_comp):
            amp = rng.uniform(0.3, 3.0)
            stddev = rng.uniform(2.0, 6.0)
            comps.append(Component(float(amp), _rand_mean(rng), float(stddev)))
        spectra.append(SyntheticSpectrum("crowded", i, _make_spectrum(rng, comps), comps))
    return spectra


GENERATORS = {
    "single_bright": _gen_single_bright,
    "single_faint": _gen_single_faint,
    "single_narrow": _gen_single_narrow,
    "single_broad": _gen_single_broad,
    "multi_separated": _gen_multi_separated,
    "multi_blended": _gen_multi_blended,
    "crowded": _gen_crowded,
}


# Parallel worker


def _evaluate_one(args: _WorkItem) -> _EvalResult:
    """Evaluate a single (spectrum, beta) pair.

    Designed for use with ``ProcessPoolExecutor``.

    Parameters
    ----------
    args : _WorkItem
        Input bundle containing the signal, category metadata,
        serialised ground-truth components, and beta value.

    Returns
    -------
    _EvalResult
        Evaluation metrics including F1, precision, recall, component
        counts, residual RMS, AICc, and per-parameter mean errors.
    """
    true_comps = [Component(**c) for c in args.true_comps_raw]
    signal = args.signal
    x = np.arange(len(signal), dtype=np.float64)

    try:
        detected_raw = fit_gaussians(signal, beta=args.beta, max_components=8)
    except (LinAlgError, ValueError):
        detected_raw = []

    detected = [Component(c.amplitude, c.mean, c.stddev) for c in detected_raw]
    n_true = len(true_comps)
    n_detected = len(detected)
    n_correct = count_correct_matches(true_comps, detected)
    prec, rec, f1 = f1_score(n_correct, n_true, n_detected)

    model = gaussian_model(x, detected) if detected else np.zeros_like(signal)
    residual = signal - model
    rms = float(np.sqrt(np.mean(residual**2)))
    n_params = 3 * n_detected
    aic = aicc(residual, n_params)

    amp_log_ratios: list[float] = []
    pos_log_ratios: list[float] = []
    width_log_ratios: list[float] = []

    pairs = match_pairs(true_comps, detected, pos_tol_sigma=1.0)
    for tc, dc in pairs:
        if tc.amplitude > 0:
            amp_log_ratios.append(float(np.log(dc.amplitude / tc.amplitude)))
        if tc.mean > 0:
            pos_log_ratios.append(float(np.log(dc.mean / tc.mean)))
        if tc.stddev > 0:
            width_log_ratios.append(float(np.log(dc.stddev / tc.stddev)))

    return _EvalResult(
        category=args.category,
        spectrum_idx=args.spec_idx,
        beta=args.beta,
        f1=round(f1, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        n_correct=n_correct,
        n_true=n_true,
        n_detected=n_detected,
        n_guessed=n_detected,
        count_error=n_detected - n_true,
        residual_rms=round(rms, 6),
        aicc=round(aic, 4) if np.isfinite(aic) else None,
        amp_log_ratio_mean=round(float(np.mean(amp_log_ratios)), 4) if amp_log_ratios else None,
        pos_log_ratio_mean=round(float(np.mean(pos_log_ratios)), 4) if pos_log_ratios else None,
        width_log_ratio_mean=(
            round(float(np.mean(width_log_ratios)), 4) if width_log_ratios else None
        ),
    )


@click.command()
@click.option("--n-per-category", default=50, show_default=True)
@click.option("--beta-min", default=3.8, show_default=True)
@click.option("--beta-max", default=4.5, show_default=True)
@click.option("--beta-steps", default=7, show_default=True)
@click.option("--seed", default=2026_02_12, show_default=True)
def synthetic(
    n_per_category: int,
    beta_min: float,
    beta_max: float,
    beta_steps: int,
    seed: int,
) -> None:
    """Synthetic benchmark with known ground-truth components."""
    output_dir = os.path.join("/tmp/phspectra", "synthetic-benchmark")
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    beta_grid = np.round(np.linspace(beta_min, beta_max, beta_steps), 1).tolist()

    # Generate spectra
    console.print("Step 1: Generate synthetic spectra", style="bold cyan")
    all_spectra: list[SyntheticSpectrum] = []
    categories = list(GENERATORS.keys())
    table = Table(title="Synthetic spectra")
    table.add_column("Category")
    table.add_column("Spectra", justify="right")
    table.add_column("Components", justify="right")
    for cat, gen_fn in GENERATORS.items():
        batch = gen_fn(rng, n_per_category)
        all_spectra.extend(batch)
        n_comps = sum(len(s.components) for s in batch)
        table.add_row(cat, str(len(batch)), str(n_comps))
    n_total = len(all_spectra)
    table.add_section()
    table.add_row("Total", str(n_total), str(sum(len(s.components) for s in all_spectra)))
    console.print(table)

    # Beta sweep
    n_fits = n_total * len(beta_grid)
    n_workers = min(os.cpu_count() or 4, 8)
    console.print(
        f"\nStep 2: Sweep {len(beta_grid)} beta values "
        f"({n_fits} total fits, {n_workers} workers)",
        style="bold cyan",
    )
    work_items: list[_WorkItem] = []
    for beta in beta_grid:
        for spec in all_spectra:
            comps_raw = [asdict(c) for c in spec.components]
            work_items.append(_WorkItem(spec.signal, spec.category, spec.index, comps_raw, beta))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        all_results = list(pool.map(_evaluate_one, work_items, chunksize=4))
    total_elapsed = time.perf_counter() - t0
    csv_rows: list[_EvalResult] = all_results

    for beta in beta_grid:
        subset = [r for r in csv_rows if r.beta == beta]
        tot_correct = sum(r.n_correct for r in subset)
        tot_true = sum(r.n_true for r in subset)
        tot_guessed = sum(r.n_guessed for r in subset)
        _, _, overall_f1 = f1_score(tot_correct, tot_true, tot_guessed)
        console.print(
            f"  beta={beta:>4.1f}  F1={overall_f1:.3f}  "
            f"correct={tot_correct:>4d}/{tot_true}  guessed={tot_guessed:>4d}"
        )
    console.print(f"  Total: {total_elapsed:.1f}s", style="green")

    # Find optimal beta
    console.print("\nStep 3: Aggregate results", style="bold cyan")
    beta_scores: dict[float, float] = {}
    for beta in beta_grid:
        subset = [r for r in csv_rows if r.beta == beta]
        tot_c = sum(r.n_correct for r in subset)
        tot_t = sum(r.n_true for r in subset)
        tot_g = sum(r.n_guessed for r in subset)
        _, _, f1 = f1_score(tot_c, tot_t, tot_g)
        beta_scores[beta] = f1
    optimal_beta = max(beta_scores, key=beta_scores.get)  # type: ignore[arg-type]
    console.print(
        f"  Optimal beta = [bold yellow]{optimal_beta:.1f}[/bold yellow]"
        f"  (F1 = {beta_scores[optimal_beta]:.4f})"
    )

    # Save outputs
    csv_path = os.path.join(output_dir, "synthetic_benchmark.csv")
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [fld.name for fld in fields(_EvalResult)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(r) for r in csv_rows)
    console.print(f"  CSV: [blue]{csv_path}[/blue]")

    json_data: dict[str, object] = {
        "config": {
            "n_channels": N_CHANNELS,
            "noise_sigma": NOISE_SIGMA,
            "n_per_category": n_per_category,
            "seed": seed,
            "beta_grid": beta_grid,
        },
        "optimal_beta": optimal_beta,
        "optimal_f1": round(beta_scores[optimal_beta], 4),
    }
    json_path = os.path.join(output_dir, "synthetic_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    console.print(f"  JSON: [blue]{json_path}[/blue]")

    _plot_f1_vs_beta(csv_rows, categories)
    _plot_error_boxplots(csv_rows, categories, optimal_beta)
    console.print("\nDone.", style="bold green")


def _agg_f1(csv_rows: list[_EvalResult], beta: float, cat: str | None = None) -> float:
    """Compute micro-averaged F1 score for a given beta.

    Parameters
    ----------
    csv_rows : list[_EvalResult]
        Full benchmark result rows from ``_evaluate_one``.
    beta : float
        Beta value to filter on.
    cat : str or None, optional
        If given, restrict to this category.

    Returns
    -------
    float
        Micro-averaged F1 score.
    """
    subset = [r for r in csv_rows if r.beta == beta]
    if cat:
        subset = [r for r in subset if r.category == cat]
    if not subset:
        return 0.0
    tc = sum(r.n_correct for r in subset)
    tt = sum(r.n_true for r in subset)
    tg = sum(r.n_guessed for r in subset)
    _, _, f1_val = f1_score(tc, tt, tg)
    return f1_val


@docs_figure("synthetic-f1.png")
def _plot_f1_vs_beta(
    csv_rows: list[_EvalResult],
    categories: list[str],
) -> Figure:
    """Build the F1 vs beta figure for all categories.

    One line per category plus a bold overall line.

    Parameters
    ----------
    csv_rows : list[_EvalResult]
        Full benchmark result rows.
    categories : list[str]
        Ordered list of category keys.

    Returns
    -------
    Figure
        The completed matplotlib figure.
    """
    betas = sorted(set(r.beta for r in csv_rows))

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.95)

    for cat in categories:
        ax.plot(
            betas,
            [_agg_f1(csv_rows, b, cat) for b in betas],
            "o-",
            label=CATEGORY_LABELS[cat],
            markersize=4,
        )
    ax.plot(
        betas,
        [_agg_f1(csv_rows, b) for b in betas],
        "ko-",
        linewidth=2.5,
        markersize=6,
        label="Overall",
    )
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("$F_1$")
    ax.legend(loc="lower left", frameon=False)
    ax.set_ylim(-0.05, 1.05)
    configure_axes(ax)

    return fig


@docs_figure("synthetic-errors.png")
def _plot_error_boxplots(
    csv_rows: list[_EvalResult],
    categories: list[str],
    optimal_beta: float,
) -> Figure:
    """Build box-whisker panels for parameter-recovery errors.

    Three vertically stacked panels (amplitude relative error, position
    error in channels, width relative error) at the optimal beta, one
    box per category.

    Parameters
    ----------
    csv_rows : list[_EvalResult]
        Full benchmark result rows.
    categories : list[str]
        Ordered list of category keys.
    optimal_beta : float
        Beta value with the highest overall F1.

    Returns
    -------
    Figure
        The completed matplotlib figure.
    """
    cat_labels = [CATEGORY_LABELS[c] for c in categories]
    opt_rows = [r for r in csv_rows if r.beta == optimal_beta]

    err_panels = [
        ("amp_log_ratio_mean", r"$\ln(A_{\mathrm{fit}} / A_{\mathrm{true}})$"),
        ("pos_log_ratio_mean", r"$\ln(\mu_{\mathrm{fit}} / \mu_{\mathrm{true}})$"),
        ("width_log_ratio_mean", r"$\ln(\sigma_{\mathrm{fit}} / \sigma_{\mathrm{true}})$"),
    ]

    fig: Figure
    axes: AxesGrid1D
    fig, axes = plt.subplots(
        len(err_panels),
        1,
        figsize=(6, 6),
        sharex=True,
    )
    fig.subplots_adjust(left=0.14, right=0.95, bottom=0.08, top=0.97, hspace=0.05)

    for ax, (key, ylabel) in zip(axes, err_panels):
        box_data = []
        for cat in categories:
            vals = [
                getattr(r, key)
                for r in opt_rows
                if r.category == cat and getattr(r, key) is not None
            ]
            box_data.append(vals)
        bp = ax.boxplot(
            box_data,
            labels=cat_labels,  # type: ignore[call-arg]
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "k", "linewidth": 1.5},
            showfliers=False,
        )
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(ylabel)
        configure_axes(ax)

    axes[-1].set_xlabel("Category")

    return fig
