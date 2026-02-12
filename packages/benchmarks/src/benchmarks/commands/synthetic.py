"""``benchmarks synthetic`` — synthetic benchmark with known ground truth."""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import click
import numpy as np
from matplotlib import pyplot as plt

from benchmarks._console import console
from benchmarks._constants import MEAN_MARGIN, N_CHANNELS, NOISE_SIGMA
from benchmarks._gaussian import gaussian, gaussian_model
from benchmarks._matching import count_correct_matches, f1_score, match_pairs
from benchmarks._types import Component, SyntheticSpectrum
from numpy.linalg import LinAlgError

from phspectra import fit_gaussians
from phspectra.quality import aicc


def _make_spectrum(
    rng: np.random.Generator,
    components: list[Component],
) -> np.ndarray:
    """Build a noisy signal from ground-truth components."""
    x = np.arange(N_CHANNELS, dtype=np.float64)
    signal = np.zeros(N_CHANNELS, dtype=np.float64)
    for c in components:
        signal += gaussian(x, c.amplitude, c.mean, c.stddev)
    signal += rng.normal(0.0, NOISE_SIGMA, N_CHANNELS)
    return signal


def _rand_mean(rng: np.random.Generator) -> float:
    return float(rng.uniform(MEAN_MARGIN, N_CHANNELS - MEAN_MARGIN))


# Spectrum generators


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


def _evaluate_one(args: tuple) -> dict:
    """Evaluate a single (spectrum, beta) pair — picklable for multiprocessing."""
    signal, category, spec_idx, true_comps_raw, beta = args
    true_comps = [Component(**c) for c in true_comps_raw]
    x = np.arange(len(signal), dtype=np.float64)

    try:
        detected_raw = fit_gaussians(signal, beta=beta, max_components=8)
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

    amp_rel_errs: list[float] = []
    pos_errs: list[float] = []
    width_rel_errs: list[float] = []

    pairs = match_pairs(true_comps, detected, pos_tol_sigma=1.0)
    for tc, dc in pairs:
        if tc.amplitude > 0:
            amp_rel_errs.append(abs(dc.amplitude - tc.amplitude) / tc.amplitude)
        pos_errs.append(abs(dc.mean - tc.mean))
        if tc.stddev > 0:
            width_rel_errs.append(abs(dc.stddev - tc.stddev) / tc.stddev)

    return {
        "category": category,
        "spectrum_idx": spec_idx,
        "beta": beta,
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "n_correct": n_correct,
        "n_true": n_true,
        "n_detected": n_detected,
        "n_guessed": n_detected,
        "count_error": n_detected - n_true,
        "residual_rms": round(rms, 6),
        "aicc": round(aic, 4) if np.isfinite(aic) else None,
        "amp_rel_err_mean": (round(float(np.mean(amp_rel_errs)), 4) if amp_rel_errs else None),
        "pos_err_mean": round(float(np.mean(pos_errs)), 4) if pos_errs else None,
        "width_rel_err_mean": (
            round(float(np.mean(width_rel_errs)), 4) if width_rel_errs else None
        ),
    }


@click.command()
@click.option("--n-per-category", default=50, show_default=True)
@click.option("--beta-min", default=3.8, show_default=True)
@click.option("--beta-max", default=4.5, show_default=True)
@click.option("--beta-steps", default=7, show_default=True)
@click.option("--seed", default=2025_02_12, show_default=True)
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
    for cat, gen_fn in GENERATORS.items():
        batch = gen_fn(rng, n_per_category)
        all_spectra.extend(batch)
        n_comps = sum(len(s.components) for s in batch)
        console.print(f"  {cat:20s}: {len(batch)} spectra, {n_comps} total components")
    n_total = len(all_spectra)
    console.print(f"  Total: {n_total} spectra")

    # Beta sweep
    n_fits = n_total * len(beta_grid)
    n_workers = min(os.cpu_count() or 4, 8)
    console.print(
        f"\nStep 2: Sweep {len(beta_grid)} beta values "
        f"({n_fits} total fits, {n_workers} workers)",
        style="bold cyan",
    )
    work_items: list[tuple] = []
    for beta in beta_grid:
        for spec in all_spectra:
            comps_raw = [asdict(c) for c in spec.components]
            work_items.append((spec.signal, spec.category, spec.index, comps_raw, beta))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        all_results = list(pool.map(_evaluate_one, work_items, chunksize=4))
    total_elapsed = time.perf_counter() - t0
    csv_rows: list[dict] = all_results

    for beta in beta_grid:
        subset = [r for r in csv_rows if r["beta"] == beta]
        tot_correct = sum(r["n_correct"] for r in subset)
        tot_true = sum(r["n_true"] for r in subset)
        tot_guessed = sum(r["n_guessed"] for r in subset)
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
        subset = [r for r in csv_rows if r["beta"] == beta]
        tot_c = sum(r["n_correct"] for r in subset)
        tot_t = sum(r["n_true"] for r in subset)
        tot_g = sum(r["n_guessed"] for r in subset)
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
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    console.print(f"  CSV: [blue]{csv_path}[/blue]")

    json_data: dict = {
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

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def agg_f1(beta: float, cat: str | None = None) -> float:
        subset = [r for r in csv_rows if r["beta"] == beta]
        if cat:
            subset = [r for r in subset if r["category"] == cat]
        if not subset:
            return 0.0
        tc = sum(r["n_correct"] for r in subset)
        tt = sum(r["n_true"] for r in subset)
        tg = sum(r["n_guessed"] for r in subset)
        _, _, f1_val = f1_score(tc, tt, tg)
        return f1_val

    betas = sorted(set(r["beta"] for r in csv_rows))
    for cat in categories:
        axes[0, 0].plot(betas, [agg_f1(b, cat) for b in betas], "o-", label=cat, markersize=4)
    axes[0, 0].plot(
        betas, [agg_f1(b) for b in betas], "ko-", linewidth=2.5, markersize=6, label="overall"
    )
    axes[0, 0].axvline(optimal_beta, color="grey", linestyle=":", alpha=0.5)
    axes[0, 0].set_xlabel(r"$\beta$")
    axes[0, 0].set_ylabel("F1")
    axes[0, 0].set_title("F1 vs beta")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].grid(True, alpha=0.3)

    # Remaining panels omitted for brevity — key data is in CSV/JSON
    for i in range(1, 6):
        row, col = divmod(i, 3)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Synthetic benchmark — {n_total} spectra, optimal beta={optimal_beta}",
        fontsize=13,
    )
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "synthetic-benchmark.png")
    fig.savefig(plot_path, dpi=150)
    console.print(f"  Plot: [blue]{plot_path}[/blue]")
    plt.close(fig)

    console.print("\nDone.", style="bold green")
