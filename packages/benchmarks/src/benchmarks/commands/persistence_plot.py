"""``benchmarks persistence-plot`` -- generate persistence homology figures.

Produces two figures for the documentation site:

1. **water-level-stages.png** -- four-panel illustration of the descending
   threshold on a synthetic three-peak spectrum.
2. **persistence-diagram.png** -- birth-death scatter plot with the
   persistence threshold line.

Both are saved to the docs static image directory via the
``@docs_figure`` decorator.
"""

from __future__ import annotations

from dataclasses import dataclass

import click
import numpy as np
import numpy.typing as npt
from benchmarks._console import console
from benchmarks._plotting import AxesGrid2D, configure_axes, docs_figure
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True)
class PersistenceEvent:
    """A single birth-death event from the persistence filtration."""

    index: int
    birth: float
    death: float
    persistence: float


def _synthetic_signal(n: int = 200) -> npt.NDArray[np.float64]:
    """Build a synthetic nested-peak signal with additive Gaussian noise.

    The signal contains a broad parent Gaussian (amplitude 2.5, sigma 12
    at channel 100) and two narrow children (amplitudes 1.5 and 1.0,
    sigma 2.5) sitting on its flanks at channels 82 and 118.  Both
    children survive the default persistence cut and illustrate the
    embedded / broad-plus-narrow superposition cases.  Additive
    i.i.d. Gaussian noise has sigma = 0.08.

    Parameters
    ----------
    n : int, optional
        Number of channels, by default 200.

    Returns
    -------
    npt.NDArray[np.float64]
        Noisy 1-D spectrum of length *n*.
    """
    x = np.arange(n, dtype=np.float64)
    signal = (
        2.5 * np.exp(-0.5 * ((x - 100.0) / 12.0) ** 2)
        + 1.5 * np.exp(-0.5 * ((x - 82.0) / 2.5) ** 2)
        + 1.0 * np.exp(-0.5 * ((x - 118.0) / 2.5) ** 2)
    )
    rng = np.random.default_rng(42)
    return signal + rng.normal(0, 0.08, n)


def _run_persistence(signal: npt.NDArray[np.float64]) -> list[PersistenceEvent]:
    """Compute 0-dimensional persistent homology of a 1-D signal.

    Processes channels in decreasing signal order, tracking connected
    components with union-find.  Every merge produces a birth-death
    pair; the global maximum is appended with death = 0.

    Parameters
    ----------
    signal : npt.NDArray[np.float64]
        Input 1-D spectrum.

    Returns
    -------
    list[PersistenceEvent]
        All birth-death events, sorted by decreasing persistence.
    """
    n = len(signal)
    parent = np.arange(n)
    rank = np.zeros(n, dtype=np.intp)
    rep = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> int:
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1
        if signal[rep[b]] > signal[rep[a]]:
            rep[a] = rep[b]
        return a

    order = np.argsort(-signal)
    visited = np.zeros(n, dtype=bool)
    events: list[PersistenceEvent] = []

    for idx in order:
        visited[idx] = True
        comp = idx
        for neighbor in (idx - 1, idx + 1):
            if neighbor < 0 or neighbor >= n or not visited[neighbor]:
                continue
            neighbor_root = find(neighbor)
            idx_root = find(comp)
            if idx_root == neighbor_root:  # pragma: no cover
                continue
            rep_ir = rep[idx_root]
            rep_nr = rep[neighbor_root]
            younger_rep = rep_ir if signal[rep_ir] < signal[rep_nr] else rep_nr
            death = signal[idx]
            birth = signal[younger_rep]
            events.append(
                PersistenceEvent(
                    index=int(younger_rep),
                    birth=float(birth),
                    death=float(death),
                    persistence=float(birth - death),
                )
            )
            merged = union(find(comp), find(neighbor))
            comp = merged

    global_max_idx = int(order[0])
    events.append(
        PersistenceEvent(
            index=global_max_idx,
            birth=float(signal[global_max_idx]),
            death=0.0,
            persistence=float(signal[global_max_idx]),
        )
    )
    events.sort(key=lambda e: e.persistence, reverse=True)
    return events


@docs_figure("water-level-stages.png")
def _plot_water_levels(
    x: npt.NDArray[np.float64],
    signal: npt.NDArray[np.float64],
    top_peaks: list[PersistenceEvent],
) -> Figure:
    """Build the four-panel descending water-level illustration.

    The signal is assumed to be a broad parent with two embedded narrow
    children.  Panels show the descending water level at four
    thresholds: (1) above all peaks, (2) the parent born as the global
    maximum, (3) both children born as sub-islands inside the parent's
    island, (4) the first child merging back into the parent.

    Parameters
    ----------
    x : npt.NDArray[np.float64]
        Channel indices.
    signal : npt.NDArray[np.float64]
        1-D spectrum values.
    top_peaks : list[PersistenceEvent]
        The three most persistent events, sorted by **persistence**
        descending.  Element 0 is the parent (global maximum) and
        elements 1 and 2 are the two children.

    Returns
    -------
    Figure
        A 2x2 matplotlib figure.
    """
    parent = top_peaks[0]
    children = top_peaks[1:]
    # The child with the higher saddle (death) merges first as the water
    # descends, so its merge is the event shown in panel 4.
    first_to_merge = max(children, key=lambda e: e.death)
    lower_child_birth = min(c.birth for c in children)

    levels = [
        parent.birth + 0.15,
        parent.birth - 0.05,
        lower_child_birth - 0.05,
        first_to_merge.death - 0.10,
    ]
    stage_labels = [
        "Water above all peaks",
        f"Parent born (channel {parent.index})",
        "Both children born inside parent",
        f"Child at channel {first_to_merge.index} merges into parent",
    ]

    fig: Figure
    axes: AxesGrid2D
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=True)

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.09, top=0.96, wspace=0.05, hspace=0.12)

    for ax, level, label in zip(axes.ravel(), levels, stage_labels):
        ax.plot(x, signal, "-k", lw=1.0, alpha=0.6, label="Signal")
        ax.fill_between(
            x,
            level,
            np.max(signal) + 0.5,
            color="steelblue",
            alpha=0.25,
            label=f"Water ($t = {level:.2f}$)",
        )
        ax.axhline(level, color="steelblue", lw=1.0, ls="--", alpha=0.7)

        for pk in top_peaks:
            if pk.birth >= level:
                ax.plot(pk.index, signal[pk.index], "o", color="C3", ms=7, zorder=5)
                ax.annotate(
                    f"$\\pi = {pk.persistence:.2f}$",
                    xy=(pk.index, signal[pk.index]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                )

        ax.set_title(label)
        ax.legend(loc="upper right", frameon=False)
        configure_axes(ax)

    fig.supxlabel("Channel")
    fig.supylabel("Signal value")
    return fig


_NOISE_SIGMA = 0.08  # must match _synthetic_signal()
_BETA = 4.0


@docs_figure("persistence-diagram.png")
def _plot_persistence_diagram(events: list[PersistenceEvent]) -> Figure:
    """Build a birth-death persistence diagram.

    Each event is plotted as a point ``(birth, death)``.  Points are
    coloured by whether their persistence exceeds the threshold
    ``pi_min = beta * sigma_rms``.  The diagonal and the threshold
    line are drawn for reference.

    Parameters
    ----------
    events : list[PersistenceEvent]
        All birth-death events from ``_run_persistence``.

    Returns
    -------
    Figure
        Single-axes matplotlib figure.
    """
    births = np.array([e.birth for e in events])
    deaths = np.array([e.death for e in events])
    persistences = births - deaths

    pi_min = _BETA * _NOISE_SIGMA
    sig_mask = persistences > pi_min
    noise_mask = ~sig_mask

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

    if np.any(noise_mask):
        ax.scatter(
            births[noise_mask],
            deaths[noise_mask],
            s=10,
            color="#5D5D5D",
            alpha=0.5,
            label="Noise",
            zorder=3,
        )
    ax.scatter(
        births[sig_mask],
        deaths[sig_mask],
        s=60,
        color="#FF6F61",
        edgecolors="#FF6F61",
        linewidths=0.5,
        label="Significant peaks",
        zorder=4,
    )

    diag_max = max(births.max(), deaths.max()) * 1.1
    ax.plot([0, diag_max], [0, diag_max], "k--", lw=0.8, alpha=0.4)

    # Threshold line: death = birth - pi_min, parallel to the diagonal.
    # Points below this line have persistence > pi_min (signal);
    # points above have persistence < pi_min (noise).
    ax.plot(
        [pi_min, diag_max],
        [0, diag_max - pi_min],
        color="0.4",
        lw=1.2,
        ls="--",
        zorder=2,
        label=r"$\pi_{\min} = \beta \sigma_{\rm rms}$",
    )

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_xlim(-0.1, diag_max)
    ax.set_ylim(-0.1, diag_max)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=False)
    configure_axes(ax)
    return fig


@click.command("persistence-plot")
def persistence_plot() -> None:
    """Generate figures illustrating the descending water level algorithm."""
    signal = _synthetic_signal()
    x = np.arange(len(signal), dtype=np.float64)
    events = _run_persistence(signal)

    top_peaks = events[:3]

    console.print(
        "Synthetic signal: broad parent + 2 narrow children + noise (sigma=0.08)",
        style="bold cyan",
    )
    console.print(f"  {len(events)} persistence events, top 3:")
    for e in top_peaks:
        console.print(
            f"    ch={e.index:>3d}  birth={e.birth:.2f}  death={e.death:.2f}  " f"persistence={e.birth - e.death:.2f}"
        )

    _plot_water_levels(x, signal, top_peaks)
    _plot_persistence_diagram(events)

    console.print("\nDone.", style="bold green")
