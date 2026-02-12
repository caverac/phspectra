"""Shared matplotlib style and panel helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from benchmarks._gaussian import gaussian_model
from benchmarks._types import ComparisonResult, Component


def plot_panel(
    ax: object,
    result: ComparisonResult,
    title: str,
) -> None:
    """Draw a single comparison panel on *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    result : ComparisonResult
        Comparison data.
    title : str
        Panel title.
    """
    n_ch = len(result.signal)
    x = np.arange(n_ch, dtype=np.float64)

    ax.plot(  # type: ignore[attr-defined]
        x, result.signal, color="0.5", linewidth=0.5, alpha=0.7, label="Data"
    )

    ph_model = gaussian_model(x, result.ph_comps) if result.ph_comps else np.zeros(n_ch)
    gp_model = gaussian_model(x, result.gp_comps) if result.gp_comps else np.zeros(n_ch)

    ax.plot(  # type: ignore[attr-defined]
        x,
        ph_model,
        color="C0",
        linewidth=1.5,
        label=f"phspectra ({len(result.ph_comps)} comp, RMS={result.ph_rms:.3f})",
    )
    ax.plot(  # type: ignore[attr-defined]
        x,
        gp_model,
        color="C3",
        linewidth=1.5,
        linestyle="--",
        label=f"GP+ ({len(result.gp_comps)} comp, RMS={result.gp_rms:.3f})",
    )

    all_means = [c.mean for c in result.ph_comps] + [c.mean for c in result.gp_comps]
    all_stds = [c.stddev for c in result.ph_comps] + [c.stddev for c in result.gp_comps]
    if all_means:
        lo = max(0, int(min(all_means) - 4 * max(all_stds)) - 10)
        hi = min(n_ch, int(max(all_means) + 4 * max(all_stds)) + 10)
        ax.set_xlim(lo, hi)  # type: ignore[attr-defined]

    ax.set_title(title, fontsize=9)  # type: ignore[attr-defined]
    ax.legend(fontsize=6, loc="upper right")  # type: ignore[attr-defined]
    ax.set_xlabel("Channel")  # type: ignore[attr-defined]
    ax.set_ylabel("T (K)")  # type: ignore[attr-defined]
    ax.grid(True, alpha=0.2)  # type: ignore[attr-defined]


def zoom_range(
    comps_list: Sequence[Sequence[Component]],
    n_channels: int,
    padding: int = 10,
) -> tuple[int, int]:
    """Compute a zoom range around all components.

    Parameters
    ----------
    comps_list : Sequence[Sequence[Component]]
        One or more lists of components.
    n_channels : int
        Total number of channels.
    padding : int
        Extra channels on each side.

    Returns
    -------
    tuple[int, int]
        (lo, hi) channel range.
    """
    all_means: list[float] = []
    all_stds: list[float] = []
    for comps in comps_list:
        for c in comps:
            all_means.append(c.mean)
            all_stds.append(c.stddev)
    if not all_means:
        return 0, n_channels
    lo = max(0, int(min(all_means) - 4 * max(all_stds)) - padding)
    hi = min(n_channels, int(max(all_means) + 4 * max(all_stds)) + padding)
    return lo, hi
