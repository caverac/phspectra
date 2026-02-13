"""``benchmarks train-beta`` -- sweep beta on real GRS data.

Loads spectra and GaussPy+ decompositions from ``benchmarks compare``
output, then evaluates phspectra at each beta in a user-defined grid.
Reports F1, precision, and recall per beta and produces a line plot
saved to both the benchmark output directory and the docs image
directory.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import sys
import time

import click
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.linalg import LinAlgError

from benchmarks._console import console, err_console
from benchmarks._constants import CACHE_DIR
from benchmarks._gaussian import gaussian_model
from benchmarks._matching import count_correct_matches, f1_score
from benchmarks._plotting import SAVEFIG_DEFAULTS, configure_axes, docs_figure
from benchmarks._types import BetaSweepResult, Component
from phspectra import fit_gaussians


@click.command("train-beta")
@click.option(
    "--data-dir",
    default=os.path.join(CACHE_DIR, "compare-docker"),
    show_default=True,
    help="Directory with spectra.npz and results.json from ``compare``.",
)
@click.option("--beta-min", default=3.0, show_default=True)
@click.option("--beta-max", default=6.0, show_default=True)
@click.option("--beta-steps", default=13, show_default=True)
def train_beta(data_dir: str, beta_min: float, beta_max: float, beta_steps: int) -> None:
    """Sweep beta values against GaussPy+ Docker decompositions."""
    output_dir = os.path.join(data_dir, "training-output")
    os.makedirs(output_dir, exist_ok=True)

    beta_grid = np.round(np.linspace(beta_min, beta_max, beta_steps), 2).tolist()

    # Load spectra
    spectra_path = os.path.join(data_dir, "spectra.npz")
    results_path = os.path.join(data_dir, "results.json")
    if not os.path.exists(spectra_path) or not os.path.exists(results_path):
        err_console.print(
            f"ERROR: {spectra_path} or {results_path} not found.\n"
            "Run ``benchmarks compare`` first."
        )
        sys.exit(1)

    console.print("Step 1: Load spectra", style="bold cyan")
    signals = np.load(spectra_path)["signals"]
    n_spectra, n_channels = signals.shape
    console.print(f"  {n_spectra} spectra, {n_channels} channels each")

    # Load GaussPy+ Docker decompositions
    console.print("\nStep 2: Load GaussPy+ Docker decompositions", style="bold cyan")
    with open(results_path, encoding="utf-8") as f:
        gp_results = json.load(f)
    gp_amps = gp_results["amplitudes_fit"]
    gp_means = gp_results["means_fit"]
    gp_stds = gp_results["stddevs_fit"]

    # Build training set (skip spectra with no GP+ detections)
    training: list[tuple[np.ndarray, list[Component]]] = []
    for i in range(n_spectra):
        if not gp_amps[i]:
            continue
        ref = [
            Component(amplitude=a, mean=m, stddev=s)
            for a, m, s in zip(gp_amps[i], gp_means[i], gp_stds[i])
        ]
        training.append((signals[i], ref))
    n_train = len(training)
    console.print(f"  {n_train} spectra with GaussPy+ components")

    # Beta sweep
    console.print(
        f"\nStep 3: Sweep {len(beta_grid)} beta values ({n_train * len(beta_grid)} fits)",
        style="bold cyan",
    )
    results: list[BetaSweepResult] = []
    for beta in beta_grid:
        tot_correct = 0
        tot_true = 0
        tot_guessed = 0
        ph_rms_list: list[float] = []
        gp_rms_list: list[float] = []

        t0 = time.perf_counter()
        for signal, ref in training:
            x = np.arange(len(signal), dtype=np.float64)
            try:
                guessed_raw = fit_gaussians(signal, beta=beta, max_components=16, sig_min=3.0)
            except (LinAlgError, ValueError):
                guessed_raw = []
            guessed = [Component(c.amplitude, c.mean, c.stddev) for c in guessed_raw]
            tot_correct += count_correct_matches(ref, guessed)
            tot_true += len(ref)
            tot_guessed += len(guessed)
            ph_model = gaussian_model(x, guessed) if guessed else np.zeros_like(x)
            ph_rms_list.append(float(np.sqrt(np.mean((signal - ph_model) ** 2))))
            gp_model = gaussian_model(x, ref)
            gp_rms_list.append(float(np.sqrt(np.mean((signal - gp_model) ** 2))))
        elapsed = time.perf_counter() - t0

        prec, rec, f1 = f1_score(tot_correct, tot_true, tot_guessed)
        results.append(
            BetaSweepResult(
                beta=beta,
                f1=round(f1, 4),
                precision=round(prec, 4),
                recall=round(rec, 4),
                n_correct=tot_correct,
                n_true=tot_true,
                n_guessed=tot_guessed,
                time_s=round(elapsed, 2),
                mean_ph_rms=round(float(np.mean(ph_rms_list)), 6),
                mean_gp_rms=round(float(np.mean(gp_rms_list)), 6),
                n_ph_wins=sum(1 for p, g in zip(ph_rms_list, gp_rms_list) if p < g),
            )
        )
        console.print(
            f"  beta={beta:>5.2f}  F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  "
            f"guessed={tot_guessed:>4d}  correct={tot_correct:>3d}/{tot_true}  "
            f"({elapsed:.1f}s)"
        )

    best = max(results, key=lambda r: r.f1)
    console.print(
        f"\nOptimal beta = [bold yellow]{best.beta:.2f}[/bold yellow]" f"  (F1 = {best.f1:.4f})"
    )

    # Save CSV
    csv_path = os.path.join(output_dir, "f1-beta-sweep.csv")
    fieldnames = [f.name for f in dataclasses.fields(BetaSweepResult)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    console.print(f"  CSV: [blue]{csv_path}[/blue]")

    # Save JSON
    json_path = os.path.join(output_dir, "f1-beta-sweep.json")
    payload = {
        "n_spectra": n_train,
        "beta_grid": beta_grid,
        "reference": "GaussPy+ Docker (alpha1=2.89, alpha2=6.65, two-phase)",
        "optimal_beta": best.beta,
        "optimal_f1": best.f1,
        "results": [dataclasses.asdict(r) for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    console.print(f"  JSON: [blue]{json_path}[/blue]")

    # Plot
    _plot_beta_sweep(results, output_dir)

    console.print("\nDone.", style="bold green")


@docs_figure("f1-beta-sweep.png")
def _plot_beta_sweep(results: list[BetaSweepResult], output_dir: str) -> Figure:
    """Build the F1/precision/recall vs beta figure.

    Parameters
    ----------
    results : list[BetaSweepResult]
        One entry per beta value, carrying F1, precision, and recall.
    output_dir : str
        Directory for the local copy of the figure.

    Returns
    -------
    Figure
        The completed matplotlib figure (saved by the ``@docs_figure``
        decorator).
    """
    betas = [r.beta for r in results]

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

    ax.plot(betas, [r.f1 for r in results], "-k", lw=1.5, label="$F_1$")
    ax.plot(betas, [r.precision for r in results], "--k", lw=1.5, label="Precision")
    ax.plot(betas, [r.recall for r in results], "-.k", lw=1.5, label="Recall")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower left", frameon=False)
    configure_axes(ax)

    # Local copy for the benchmark output directory.
    fig.savefig(os.path.join(output_dir, "f1-beta-sweep.png"), **SAVEFIG_DEFAULTS)
    return fig
