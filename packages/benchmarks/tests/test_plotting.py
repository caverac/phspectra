"""Tests for benchmarks._plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from benchmarks._plotting import (
    configure_axes,
    docs_figure,
    plot_panel,
    save_figure_if_changed,
    zoom_range,
)
from benchmarks._types import ComparisonResult, Component
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def test_configure_axes() -> None:
    """configure_axes should set tick params without error."""
    fig, ax = plt.subplots()
    configure_axes(ax)
    assert ax.xaxis.get_minor_locator() is not None
    plt.close(fig)


def test_save_figure_if_changed_new(tmp_path: Path) -> None:
    """save_figure_if_changed should write a new file."""
    fig, _ = plt.subplots()
    path = tmp_path / "new.png"
    assert save_figure_if_changed(fig, path, dpi=50) is True
    assert path.exists()
    plt.close(fig)


def test_save_figure_if_changed_unchanged(tmp_path: Path) -> None:
    """save_figure_if_changed should skip identical file."""
    fig, _ = plt.subplots()
    path = tmp_path / "same.png"
    save_figure_if_changed(fig, path, dpi=50)
    assert save_figure_if_changed(fig, path, dpi=50) is False
    plt.close(fig)


def test_save_figure_if_changed_changed(tmp_path: Path) -> None:
    """save_figure_if_changed should overwrite changed file."""
    fig1, ax1 = plt.subplots()
    ax1.plot([0, 1], [0, 1])
    path = tmp_path / "changed.png"
    save_figure_if_changed(fig1, path, dpi=50)
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [1, 0])
    assert save_figure_if_changed(fig2, path, dpi=50) is True
    plt.close(fig2)


def test_docs_figure_decorator(docs_img_dir: Path) -> None:
    """@docs_figure should save the figure and return it."""

    @docs_figure("test-out.png")
    def make_fig() -> Figure:
        fig, _ = plt.subplots()
        return fig

    fig = make_fig()
    assert isinstance(fig, Figure)
    assert (docs_img_dir / "test-out.png").exists()


@pytest.mark.usefixtures("docs_img_dir")
def test_docs_figure_unchanged() -> None:
    """@docs_figure should print 'Unchanged' when figure is identical."""

    @docs_figure("unchanged.png")
    def make_fig() -> Figure:
        fig, _ = plt.subplots()
        return fig

    make_fig()
    fig2 = make_fig()
    assert isinstance(fig2, Figure)


def test_plot_panel_with_components() -> None:
    """plot_panel should plot data and models."""
    signal = np.random.default_rng(0).normal(0, 0.1, 50)
    comps = [Component(1.0, 25.0, 3.0)]
    result = ComparisonResult(
        pixel=(0, 0),
        signal=signal,
        ph_comps=comps,
        gp_comps=comps,
        ph_rms=0.1,
        gp_rms=0.1,
        ph_time=0.01,
        gp_time=0.5,
    )
    fig, ax = plt.subplots()
    plot_panel(ax, result, "test")
    assert len(ax.lines) > 0
    plt.close(fig)


def test_plot_panel_no_components() -> None:
    """plot_panel should handle empty component lists."""
    signal = np.zeros(50)
    result = ComparisonResult(
        pixel=(0, 0),
        signal=signal,
        ph_comps=[],
        gp_comps=[],
        ph_rms=0.0,
        gp_rms=0.0,
        ph_time=0.01,
        gp_time=0.5,
    )
    fig, ax = plt.subplots()
    plot_panel(ax, result, "empty")
    plt.close(fig)


def test_zoom_range_with_components() -> None:
    """zoom_range should return a range around components."""
    comps = [Component(1.0, 100.0, 5.0)]
    lo, hi = zoom_range([comps], 424)
    assert lo < 100
    assert hi > 100
    assert lo >= 0
    assert hi <= 424


def test_zoom_range_empty() -> None:
    """zoom_range should return full range for empty components."""
    lo, hi = zoom_range([[]], 424)
    assert lo == 0
    assert hi == 424
