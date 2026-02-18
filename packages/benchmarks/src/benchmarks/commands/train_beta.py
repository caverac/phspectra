"""``benchmarks train`` -- sweep beta on real GRS data.

Loads spectra and a curated training set from ``benchmarks pre-compute``
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
from typing import Any

import click
import numpy as np
import numpy.typing as npt
from benchmarks._console import console, err_console
from benchmarks._constants import CACHE_DIR
from benchmarks._gaussian import gaussian_model
from benchmarks._matching import count_correct_matches, f1_score
from benchmarks._plotting import configure_axes, docs_figure
from benchmarks._types import BetaSweepResult, Component
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.linalg import LinAlgError

from phspectra import fit_gaussians


def _load_training_set(
    training_set: str,
    pixel_to_idx: dict[tuple[int, int], int],
    signals: npt.NDArray[np.float64],
) -> tuple[list[tuple[npt.NDArray[np.float64], list[Component]]], str]:
    """Load a curated training set and resolve pixel-to-signal indices.

    Returns the training pairs and a human-readable label.
    """
    with open(training_set, encoding="utf-8") as f:
        ts_entries: list[dict[str, Any]] = json.load(f)

    training: list[tuple[npt.NDArray[np.float64], list[Component]]] = []
    n_skipped = 0
    for entry in ts_entries:
        pixel = (entry["pixel"][0], entry["pixel"][1])
        comps = entry.get("components", [])
        if not comps:
            continue
        if pixel not in pixel_to_idx:
            n_skipped += 1
            continue
        idx = pixel_to_idx[pixel]
        ref = [Component(c["amplitude"], c["mean"], c["stddev"]) for c in comps]
        training.append((signals[idx], ref))

    ref_label = f"Curated training set ({training_set})"
    n_train = len(training)
    n_comps = sum(len(r) for _, r in training)
    console.print(f"  {n_train} curated spectra, {n_comps} reference components")
    if n_skipped:
        console.print(f"  {n_skipped} pixel(s) not in pre-compute data, skipped", style="yellow")
    return training, ref_label


@click.command("train")
@click.option(
    "--data-dir",
    default=os.path.join(CACHE_DIR, "compare-docker"),
    show_default=True,
    help="Directory with spectra.npz from ``pre-compute``.",
)
@click.option("--beta-min", default=3.8, show_default=True)
@click.option("--beta-max", default=4.5, show_default=True)
@click.option("--beta-steps", default=16, show_default=True)
@click.option(
    "--training-set",
    required=True,
    type=click.Path(exists=True),
    help="Path to curated training_set.json from train-gui.",
)
def train_beta(
    data_dir: str,
    beta_min: float,
    beta_max: float,
    beta_steps: int,
    training_set: str,
) -> None:
    """Sweep beta values against a curated training set."""
    output_dir = os.path.join(data_dir, "training-output")
    os.makedirs(output_dir, exist_ok=True)

    beta_grid = np.round(np.linspace(beta_min, beta_max, beta_steps), 2).tolist()

    # Load spectra --------------------------------------------------------
    spectra_path = os.path.join(data_dir, "spectra.npz")
    if not os.path.exists(spectra_path):
        err_console.print(f"ERROR: {spectra_path} not found.\nRun ``benchmarks pre-compute`` first.")
        sys.exit(1)

    console.print("Step 1: Load spectra", style="bold cyan")
    npz = np.load(spectra_path)
    signals = npz["signals"]
    if "pixels" not in npz:
        err_console.print(
            "ERROR: spectra.npz missing 'pixels' array.\nRe-run ``benchmarks pre-compute`` to regenerate it."
        )
        sys.exit(1)
    npz_pixels = npz["pixels"]
    pixel_to_idx = {(int(r[0]), int(r[1])): i for i, r in enumerate(npz_pixels)}
    n_spectra, n_channels = signals.shape
    console.print(f"  {n_spectra} spectra, {n_channels} channels each")

    # Build reference training set ----------------------------------------
    console.print("\nStep 2: Load curated training set", style="bold cyan")
    training, ref_label = _load_training_set(
        training_set,
        pixel_to_idx,
        signals,
    )
    if not training:
        err_console.print("ERROR: no training spectra found.")
        sys.exit(1)

    # Beta sweep
    console.print(
        f"\nStep 3: Sweep {len(beta_grid)} beta values ({len(training) * len(beta_grid)} fits)",
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
                guessed_raw = fit_gaussians(signal, beta=beta, mf_snr_min=3.5)
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
    f1_values = [r.f1 for r in results]
    f1_variation = max(f1_values) - min(f1_values)
    console.print(
        f"\nOptimal beta = [bold yellow]{best.beta:.2f}[/bold yellow]"
        f"  (F1 = {best.f1:.4f}, P = {best.precision:.4f}, R = {best.recall:.4f})"
    )
    console.print(
        f"  F1 variation across sweep: {f1_variation:.3f} "
        f"(min {min(f1_values):.3f} at beta={results[f1_values.index(min(f1_values))].beta:.2f}, "
        f"max {max(f1_values):.3f} at beta={best.beta:.2f})"
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
        "n_spectra": len(training),
        "beta_grid": beta_grid,
        "reference": ref_label,
        "optimal_beta": best.beta,
        "optimal_f1": best.f1,
        "results": [dataclasses.asdict(r) for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    console.print(f"  JSON: [blue]{json_path}[/blue]")

    # Plot
    _plot_beta_sweep(results)

    console.print("\nDone.", style="bold green")


@docs_figure("f1-beta-sweep.png")
def _plot_beta_sweep(results: list[BetaSweepResult]) -> Figure:
    """Build the F1/precision/recall vs beta figure.

    Parameters
    ----------
    results : list[BetaSweepResult]
        One entry per beta value, carrying F1, precision, and recall.

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

    return fig
