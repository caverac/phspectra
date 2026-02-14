"""Tests for benchmarks.commands.persistence_plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from benchmarks.cli import main
from benchmarks.commands.persistence_plot import (
    PersistenceEvent,
    _plot_persistence_diagram,
    _plot_water_levels,
    _run_persistence,
    _synthetic_signal,
)
from click.testing import CliRunner
from matplotlib import pyplot as plt


def test_synthetic_signal() -> None:
    """_synthetic_signal should return a 1-D array of the requested length."""
    sig = _synthetic_signal(100)
    assert sig.shape == (100,)
    assert sig.dtype == np.float64


def test_run_persistence() -> None:
    """_run_persistence should return events sorted by persistence."""
    sig = _synthetic_signal()
    events = _run_persistence(sig)
    assert len(events) > 0
    assert events[0].persistence >= events[-1].persistence
    assert all(isinstance(e, PersistenceEvent) for e in events)



def test_persistence_event() -> None:
    """PersistenceEvent should be a frozen dataclass."""
    e = PersistenceEvent(index=0, birth=2.0, death=1.0, persistence=1.0)
    assert e.persistence == 1.0


def test_plot_water_levels(docs_img_dir: Path) -> None:
    """_plot_water_levels should produce a 2x2 figure."""
    sig = _synthetic_signal()
    events = _run_persistence(sig)
    top = sorted(events[:3], key=lambda e: e.index)
    x = np.arange(len(sig), dtype=np.float64)
    fig = _plot_water_levels(x, sig, top)
    assert fig is not None
    plt.close(fig)


def test_plot_persistence_diagram(docs_img_dir: Path) -> None:
    """_plot_persistence_diagram should produce a scatter figure."""
    sig = _synthetic_signal()
    events = _run_persistence(sig)
    fig = _plot_persistence_diagram(events)
    assert fig is not None
    plt.close(fig)


def test_persistence_plot_cli(docs_img_dir: Path) -> None:
    """CLI should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["persistence-plot"])
    assert result.exit_code == 0, result.output
