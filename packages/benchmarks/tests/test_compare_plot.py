"""Tests for benchmarks.commands.compare_plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from benchmarks._types import ComparisonResult, Component
from benchmarks.cli import main
from benchmarks.commands.compare_plot import (
    _build_disagreements_figure,
    _build_rms_hist,
    _build_rms_scatter,
    _build_width_hist,
    _collect_matched_widths,
    _load_results,
    _select_disagreement_cases,
)
from click.testing import CliRunner
from matplotlib import pyplot as plt


def _make_result(
    px: int = 0,
    py: int = 0,
    ph_mean: float = 25.0,
    gp_mean: float = 25.0,
    ph_amp: float = 1.0,
    gp_amp: float = 1.0,
    ph_std: float = 3.0,
    gp_std: float = 3.0,
    n_ph: int = 1,
    n_gp: int = 1,
) -> ComparisonResult:
    signal = np.random.default_rng(px).normal(0, 0.1, 50)
    ph = [Component(ph_amp, ph_mean + i * 20, ph_std) for i in range(n_ph)]
    gp = [Component(gp_amp, gp_mean + i * 20, gp_std) for i in range(n_gp)]
    return ComparisonResult(
        pixel=(px, py),
        signal=signal,
        ph_comps=ph,
        gp_comps=gp,
        ph_rms=0.1 * (px + 1),
        gp_rms=0.15 * (px + 1),
        ph_time=0.01,
        gp_time=0.5,
    )


def test_load_results(comparison_data_dir: Path) -> None:
    """_load_results should reconstruct ComparisonResult from files."""
    results, summary = _load_results(str(comparison_data_dir))
    assert len(results) == 2
    assert summary.ph_total_time == 0.03


def test_load_results_missing_file(tmp_path: Path) -> None:
    """_load_results should exit when files are missing."""
    with pytest.raises(SystemExit):
        _load_results(str(tmp_path))


def test_select_disagreement_cases() -> None:
    """_select_disagreement_cases should pick up to 6 cases."""
    # "Same N, different positions" branch: 2 comps each, with position diff > 2.0
    # but within match_pairs tolerance (2.0 * stddev = 2.0 * 10.0 = 20.0)
    signal_pos = np.random.default_rng(10).normal(0, 0.1, 50)
    pos_result = ComparisonResult(
        pixel=(10, 10),
        signal=signal_pos,
        ph_comps=[Component(1.0, 10.0, 10.0), Component(1.0, 40.0, 10.0)],
        gp_comps=[Component(1.0, 15.0, 10.0), Component(1.0, 45.0, 10.0)],
        ph_rms=0.1,
        gp_rms=0.15,
        ph_time=0.01,
        gp_time=0.5,
    )
    results = [
        _make_result(0, n_ph=1, n_gp=3),
        _make_result(1, n_ph=3, n_gp=1),
        _make_result(2, ph_std=1.0, gp_std=5.0),
        _make_result(3),
        pos_result,
        _make_result(5),
        _make_result(6),
    ]
    cases = _select_disagreement_cases(results)
    assert len(cases) <= 6
    assert len(cases) >= 1


def test_collect_matched_widths() -> None:
    """_collect_matched_widths should return parallel width lists."""
    results = [_make_result(0), _make_result(1)]
    ph_w, gp_w = _collect_matched_widths(results)
    assert len(ph_w) == len(gp_w)
    assert len(ph_w) >= 1


@pytest.mark.usefixtures("docs_img_dir")
def test_build_rms_hist() -> None:
    """_build_rms_hist should produce a figure."""
    results = [_make_result(i) for i in range(5)]
    fig = _build_rms_hist(results)
    assert fig is not None
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_build_rms_scatter() -> None:
    """_build_rms_scatter should produce a figure."""
    results = [_make_result(i) for i in range(5)]
    fig = _build_rms_scatter(results)
    assert fig is not None
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_build_disagreements_figure() -> None:
    """_build_disagreements_figure should produce a 2x3 figure."""
    cases = [("label", _make_result(i)) for i in range(3)]
    fig = _build_disagreements_figure(cases)
    assert fig is not None
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_build_width_hist() -> None:
    """_build_width_hist should produce a figure."""
    fig = _build_width_hist([3.0, 4.0, 5.0], [3.1, 4.1, 5.1])
    assert fig is not None
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_compare_plot_cli(comparison_data_dir: Path) -> None:
    """CLI should succeed with valid data dir."""
    runner = CliRunner()
    result = runner.invoke(main, ["compare-plot", "--data-dir", str(comparison_data_dir)])
    assert result.exit_code == 0, result.output


def test_compare_plot_cli_missing(tmp_path: Path) -> None:
    """CLI should fail when files are missing."""
    runner = CliRunner()
    result = runner.invoke(main, ["compare-plot", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
