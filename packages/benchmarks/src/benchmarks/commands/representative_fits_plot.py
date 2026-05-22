"""``benchmarks representative-fits-plot`` -- representative-fits figure.

Generates ``representative-fits.png`` for the manuscript, addressing R4
in the 2026-05-08 reviewer plan: a 2x3 panel showing individual
phspectra decompositions across the benchmark difficulty spectrum.

Five panels are drawn from the synthetic benchmark with known
ground-truth components (single bright, well-separated multi, heavily
blended, crowded, plus a worst-F1 failure case from a small MB/C scan).
The sixth panel is a real noisy GRS sightline drawn from the
``grs-test-field.fits`` cube downloaded by ``benchmarks download``.

Each synthetic panel overlays the true Gaussian components beneath the
recovered fit so the reader can compare fit-versus-truth directly; the
GRS panel shows data plus fit only, since no ground truth is available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import click
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from benchmarks._console import console, err_console
from benchmarks._constants import DEFAULT_SEED, FITS_CACHE
from benchmarks._gaussian import gaussian_model, residual_rms
from benchmarks._matching import count_correct_matches, f1_score
from benchmarks._plotting import AxesGrid2D, configure_axes, docs_figure, zoom_range
from benchmarks._types import Component, SyntheticSpectrum
from benchmarks.commands.train_synthetic import (
    _gen_crowded,
    _gen_multi_blended,
    _gen_multi_separated,
    _gen_single_bright,
)
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.linalg import LinAlgError

from phspectra import fit_gaussians

_DEFAULT_BETA = 3.5
_SUCCESS_SCAN = 8  # per-category candidates scanned to pick the best-F1 fit
_FAILURE_SCAN = 50  # MB + C spectra scanned to pick the worst-F1 case
_NOISY_RMS_THRESHOLD = 0.2  # K -- matches the "noisy" regime in §3.2


@dataclass(frozen=True)
class _FitPanel:
    """One panel of the representative-fits figure."""

    signal: npt.NDArray[np.float64]
    fitted: list[Component]
    truth: list[Component] | None
    label: str
    rms: float
    f1: float | None  # None when ground truth is unavailable


def _fit_signal(signal: npt.NDArray[np.float64], beta: float) -> list[Component]:
    """Run ``fit_gaussians`` on *signal*, returning a list of components.

    Wraps the low-level solver to suppress the rare ``LinAlgError``/
    ``ValueError`` cases that the benchmark already handles elsewhere.
    """
    try:
        raw = fit_gaussians(signal, beta=beta)
    except (LinAlgError, ValueError):  # pragma: no cover - defensive
        return []
    return [Component(c.amplitude, c.mean, c.stddev) for c in raw]


def _make_synthetic_panel(
    spec: SyntheticSpectrum,
    beta: float,
    label: str,
) -> _FitPanel:
    """Fit a single synthetic spectrum and bundle the panel inputs."""
    fitted = _fit_signal(spec.signal, beta)
    n_correct = count_correct_matches(spec.components, fitted)
    _, _, f1 = f1_score(n_correct, len(spec.components), len(fitted))
    return _FitPanel(
        signal=spec.signal,
        fitted=fitted,
        truth=list(spec.components),
        label=label,
        rms=residual_rms(spec.signal, fitted),
        f1=f1,
    )


def _select_best_panel(
    candidates: Sequence[SyntheticSpectrum],
    beta: float,
    label: str,
) -> _FitPanel:
    """Return the highest-F1 fit from *candidates* for a success-case panel."""
    best: _FitPanel | None = None
    for spec in candidates:
        panel = _make_synthetic_panel(spec, beta, label)
        if best is None or (panel.f1 is not None and panel.f1 > (best.f1 or 0.0)):
            best = panel
    assert best is not None, "candidate list must be non-empty"
    return best


def _select_failure_panel(
    candidates: Sequence[SyntheticSpectrum],
    beta: float,
) -> _FitPanel:
    """Return the spectrum from *candidates* whose fit has the lowest F1.

    Used to surface a representative failure mode (under- or over-fit) for
    the manuscript's "challenging cases" panel.
    """
    worst: _FitPanel | None = None
    for spec in candidates:
        panel = _make_synthetic_panel(spec, beta, label="Failure")
        if worst is None or (panel.f1 is not None and panel.f1 < (worst.f1 or 1.0)):
            worst = panel
    assert worst is not None, "candidate list must be non-empty"
    return worst


def _load_grs_noisy_panel(fits_path: str, beta: float) -> _FitPanel:
    """Load the GRS test field and fit a representative noisy pixel.

    Per-pixel noise is estimated robustly with the median absolute
    deviation of channel values.  Among pixels above the noisy-regime
    threshold (~0.2 K, the cut used in §3.2 of the manuscript), the one
    whose noise level sits closest to the median of that subset is
    chosen, yielding a representative -- not extreme -- example.
    """
    cube = np.asarray(fits.getdata(fits_path), dtype=np.float64)

    med = np.median(cube, axis=0, keepdims=True)
    sigma_map = 1.4826 * np.median(np.abs(cube - med), axis=0)

    noisy = sigma_map > _NOISY_RMS_THRESHOLD
    if np.any(noisy):
        target = float(np.median(sigma_map[noisy]))
        diffs = np.abs(sigma_map - target)
        diffs[~noisy] = np.inf
        flat = int(np.argmin(diffs))
    else:
        flat = int(np.argmax(sigma_map))

    indices = np.unravel_index(flat, sigma_map.shape)
    yi, xi = int(indices[0]), int(indices[1])
    signal = cube[:, yi, xi]
    fitted = _fit_signal(signal, beta)
    return _FitPanel(
        signal=signal,
        fitted=fitted,
        truth=None,
        label=f"Noisy GRS pixel ({int(xi)}, {int(yi)})",
        rms=residual_rms(signal, fitted),
        f1=None,
    )


def _draw_panel(ax: Axes, panel: _FitPanel) -> None:
    """Render one ``_FitPanel`` onto *ax* with consistent styling."""
    x = np.arange(len(panel.signal), dtype=np.float64)

    if panel.truth is not None:
        data_label = f"Data ($N_{{\\mathrm{{components}}}} = {len(panel.truth)}$)"
    else:
        data_label = "Data"
    ax.step(x, panel.signal, where="mid", color="0.55", linewidth=1.0, alpha=0.85, label=data_label)

    if panel.fitted:
        ax.plot(
            x,
            gaussian_model(x, panel.fitted),
            color="black",
            linewidth=1.6,
            label=f"PHS ($N = {len(panel.fitted)}$, RMS$={panel.rms:.3f}$)",
        )
    else:
        ax.plot(
            x,
            np.zeros_like(x),
            color="black",
            linewidth=1.6,
            label="PHS (no components)",
        )

    if panel.fitted or panel.truth:
        comps_for_zoom = [list(panel.fitted)]
        if panel.truth is not None:
            comps_for_zoom.append(list(panel.truth))
        lo, hi = zoom_range(comps_for_zoom, len(panel.signal))
        ax.set_xlim(lo, hi)

    title = panel.label
    if panel.f1 is not None:
        title = f"{title}  ($F_1 = {panel.f1:.2f}$)"
    ax.text(
        0.03,
        0.05,
        title,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)


@docs_figure("representative-fits.png")
def _build_representative_fits_figure(panels: Sequence[_FitPanel]) -> Figure:
    """Assemble the 2x3 representative-fits figure."""
    n_rows, n_cols = 2, 3
    fig: Figure
    axes: AxesGrid2D
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 6))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=0.97, wspace=0.16, hspace=0.18)

    for ax, panel in zip(axes.ravel(), panels):
        _draw_panel(ax, panel)
        configure_axes(ax)

    for j in range(n_cols):
        axes[n_rows - 1, j].set_xlabel("Channel")
    for i in range(n_rows):
        axes[i, 0].set_ylabel("$T$ (K)")

    return fig


def _build_panels(
    fits_path: str,
    beta: float,
    seed: int,
) -> list[_FitPanel]:
    """Build the six panel inputs for the representative-fits figure."""
    rng = np.random.default_rng(seed)

    sb_pool = _gen_single_bright(rng, _SUCCESS_SCAN)
    ms_pool = _gen_multi_separated(rng, _SUCCESS_SCAN)
    mb_pool = _gen_multi_blended(rng, _SUCCESS_SCAN)
    cr_pool = _gen_crowded(rng, _SUCCESS_SCAN)
    failure_pool: list[SyntheticSpectrum] = []
    failure_pool.extend(_gen_multi_blended(rng, _FAILURE_SCAN // 2))
    failure_pool.extend(_gen_crowded(rng, _FAILURE_SCAN // 2))

    panels: list[_FitPanel] = [
        _select_best_panel(sb_pool, beta, "Single Bright"),
        _select_best_panel(ms_pool, beta, "Multi Separated"),
        _select_best_panel(mb_pool, beta, "Multi Blended"),
        _select_best_panel(cr_pool, beta, "Crowded"),
        _select_failure_panel(failure_pool, beta),
        _load_grs_noisy_panel(fits_path, beta),
    ]
    return panels


@click.command("representative-fits-plot")
@click.option(
    "--fits-path",
    default=FITS_CACHE,
    show_default=True,
    type=click.Path(),
    help="Path to grs-test-field.fits (run `benchmarks download` to populate).",
)
@click.option("--beta", default=_DEFAULT_BETA, show_default=True, type=float)
@click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)
def representative_fits_plot(fits_path: str, beta: float, seed: int) -> None:
    """Generate the representative-fits figure (R4 of the review)."""
    if not os.path.exists(fits_path):
        err_console.print(
            f"ERROR: missing GRS test field at [bold]{fits_path}[/bold].\n" "Run `benchmarks download` first.",
        )
        raise SystemExit(1)

    console.print("Building representative-fits panels ...", style="bold cyan")
    panels = _build_panels(fits_path, beta, seed)
    for p in panels:
        f1_txt = "n/a" if p.f1 is None else f"{p.f1:.2f}"
        truth_n = "n/a" if p.truth is None else str(len(p.truth))
        console.print(
            f"  {p.label:<22s}  true N={truth_n:>3s}  " f"fit N={len(p.fitted):>2d}  RMS={p.rms:.3f}  F1={f1_txt}",
            style="green",
        )

    _build_representative_fits_figure(panels)
    console.print("\nDone.", style="bold green")
