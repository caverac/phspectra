"""Shared matplotlib style and panel helpers."""

from __future__ import annotations

import functools
import io
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
from benchmarks._console import console
from benchmarks._constants import DOCS_IMG_DIR
from benchmarks._gaussian import gaussian_model
from benchmarks._types import ComparisonResult, Component
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

F = TypeVar("F", bound=Callable[..., Figure])

SAVEFIG_DEFAULTS: dict[str, int | str] = {
    "dpi": 300,
    "bbox_inches": "tight",
    "facecolor": "white",
    "edgecolor": "none",
}


def docs_figure(filename: str, **extra_kwargs: Any) -> Callable[[F], F]:
    """Save the returned Figure to DOCS_IMG_DIR if changed.

    The decorated function must return a ``matplotlib.figure.Figure``.
    The decorator handles saving, change detection, console output, and
    closing the figure.

    Parameters
    ----------
    filename:
        Output filename (e.g. ``"my-plot.png"``), saved under *DOCS_IMG_DIR*.
    **extra_kwargs:
        Extra arguments forwarded to ``fig.savefig()``, merged on top of
        ``SAVEFIG_DEFAULTS``.
    """
    savefig_kwargs = {**SAVEFIG_DEFAULTS, **extra_kwargs}

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Figure:
            fig = func(*args, **kwargs)
            path = Path(DOCS_IMG_DIR) / filename
            if save_figure_if_changed(fig, path, **savefig_kwargs):
                console.print(f"  Saved [blue]{path}[/blue]")
            else:
                console.print(f"  Unchanged [dim]{path}[/dim]")
            plt.close(fig)
            return fig

        return wrapper  # type: ignore[return-value]

    return decorator


def save_figure_if_changed(
    fig: Figure,
    path: Path,
    **savefig_kwargs: Any,
) -> bool:
    """Save a figure only if its pixel content differs from the existing file.

    Prevents unnecessary file changes that would bloat git history
    when the visual content hasn't actually changed.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    path:
        Output path for the figure.
    **savefig_kwargs:
        Arguments passed to ``fig.savefig()`` (e.g. *dpi*, *bbox_inches*).

    Returns
    -------
    bool
        True if the file was written, False if skipped (unchanged).
    """
    if path.exists():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", **savefig_kwargs)
        buf.seek(0)
        new_img = plt.imread(buf)
        existing_img = plt.imread(path)
        if new_img.shape == existing_img.shape and bool(np.array_equal(new_img, existing_img)):
            return False

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **savefig_kwargs)
    return True


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

    ax.step(  # type: ignore[attr-defined]
        x, result.signal, where="mid", color="0.6", linewidth=1.0, alpha=0.7, label="Data"
    )

    ph_model = gaussian_model(x, result.ph_comps) if result.ph_comps else np.zeros(n_ch)
    gp_model = gaussian_model(x, result.gp_comps) if result.gp_comps else np.zeros(n_ch)

    ax.plot(  # type: ignore[attr-defined]
        x,
        ph_model,
        color="k",
        linewidth=2.0,
        label=f"phspectra ({len(result.ph_comps)} comp, RMS={result.ph_rms:.3f})",
    )
    ax.plot(  # type: ignore[attr-defined]
        x,
        gp_model,
        color="k",
        linewidth=2.0,
        linestyle="--",
        label=f"GP+ ({len(result.gp_comps)} comp, RMS={result.gp_rms:.3f})",
    )

    all_means = [c.mean for c in result.ph_comps] + [c.mean for c in result.gp_comps]
    all_stds = [c.stddev for c in result.ph_comps] + [c.stddev for c in result.gp_comps]
    if all_means:
        lo = max(0, int(min(all_means) - 4 * max(all_stds)) - 10)
        hi = min(n_ch, int(max(all_means) + 4 * max(all_stds)) + 10)
        ax.set_xlim(lo, hi)  # type: ignore[attr-defined]

    ax.text(  # type: ignore[attr-defined]
        0.03,
        0.05,
        title,
        transform=ax.transAxes,  # type: ignore[attr-defined]
        va="bottom",
        ha="left",
    )
    ax.legend(loc="upper right", frameon=False)  # type: ignore[attr-defined]
    ax.set_xlabel("Channel")  # type: ignore[attr-defined]
    ax.set_ylabel("T (K)")  # type: ignore[attr-defined]


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
