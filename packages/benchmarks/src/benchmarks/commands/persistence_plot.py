"""Illustrate the descending water level algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import click

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import numpy.typing as npt

from benchmarks._console import console
from benchmarks._plotting import AxesGrid2D, docs_figure


@dataclass(frozen=True, slots=True)
class PersistenceEvent:
    """A single birth-death event from the persistence filtration."""

    index: int
    birth: float
    death: float
    persistence: float


def _synthetic_signal(n: int = 200) -> npt.NDArray[np.float64]:
    """Build a clean multi-peak signal for illustration."""
    x = np.arange(n, dtype=np.float64)
    signal = (
        2.5 * np.exp(-0.5 * ((x - 60.0) / 8.0) ** 2)
        + 1.4 * np.exp(-0.5 * ((x - 100.0) / 5.0) ** 2)
        + 0.7 * np.exp(-0.5 * ((x - 140.0) / 10.0) ** 2)
    )
    rng = np.random.default_rng(42)
    return signal + rng.normal(0, 0.08, n)


def _run_persistence(signal: npt.NDArray[np.float64]) -> list[PersistenceEvent]:
    """Run the union-find algorithm, recording every birth/death event."""
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
            if idx_root == neighbor_root:
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


def _configure_axes(ax: Axes) -> None:
    """Apply the shared tick/grid style."""
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="minor", length=3, color="gray", direction="in")
    ax.tick_params(which="major", length=6, direction="in")
    ax.tick_params(top=True, right=True, which="both")


@docs_figure("water-level-stages.png")
def _plot_water_levels(
    x: npt.NDArray[np.float64],
    signal: npt.NDArray[np.float64],
    top_peaks: list[PersistenceEvent],
) -> Figure:
    """Build the four-panel water level figure."""
    levels = [
        top_peaks[0].birth + 0.15,
        top_peaks[0].birth - 0.05,
        top_peaks[1].birth - 0.05,
        top_peaks[2].death - 0.10,
    ]
    stage_labels = [
        "Water above all peaks",
        f"Peak A born (channel {top_peaks[0].index})",
        f"Peak B born (channel {top_peaks[1].index})",
        "Peak C merges into A — C dies",
    ]

    fig: Figure
    axes: AxesGrid2D
    fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.09, top=0.96, wspace=0.06, hspace=0.12)

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
        _configure_axes(ax)

    fig.supxlabel("Channel")
    fig.supylabel("Signal value")
    return fig


_NOISE_SIGMA = 0.08  # must match _synthetic_signal()
_BETA = 4.0


@docs_figure("persistence-diagram.png")
def _plot_persistence_diagram(events: list[PersistenceEvent]) -> Figure:
    """Build the persistence diagram figure."""
    births = np.array([e.birth for e in events])
    deaths = np.array([e.death for e in events])
    persistences = births - deaths

    pi_min = _BETA * _NOISE_SIGMA
    sig_mask = persistences > pi_min
    noise_mask = ~sig_mask

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
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

    # Threshold line: death = birth - π_min, parallel to the diagonal.
    # Points below this line have persistence > π_min (signal);
    # points above have persistence < π_min (noise).
    ax.plot(
        [pi_min, diag_max],
        [0, diag_max - pi_min],
        color="0.4",
        lw=1.2,
        ls="--",
        zorder=2,
        label=r"$\pi_{\min} = \beta \times \sigma_{\rm rms}$",
    )

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_xlim(-0.1, diag_max)
    ax.set_ylim(-0.1, diag_max)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=False)
    _configure_axes(ax)
    return fig


@click.command("persistence-plot")
def persistence_plot() -> None:
    """Generate figures illustrating the descending water level algorithm."""
    signal = _synthetic_signal()
    x = np.arange(len(signal), dtype=np.float64)
    events = _run_persistence(signal)

    top_peaks = sorted(events[:3], key=lambda e: e.index)

    _plot_water_levels(x, signal, top_peaks)
    _plot_persistence_diagram(events)

    console.print("\nDone.", style="bold green")
