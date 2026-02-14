"""Tests for benchmarks.commands.ncomp_rms_plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from benchmarks.cli import main
from benchmarks.commands.ncomp_rms_plot import (
    _annotate_panel,
    _build_ncomp_rms,
    _compute_rms_and_ncomp,
)
from click.testing import CliRunner
from matplotlib import pyplot as plt


def test_compute_rms_and_ncomp() -> None:
    """_compute_rms_and_ncomp should return rms and ncomp arrays."""
    signals = np.zeros((2, 50))
    data = {
        "amplitudes_fit": [[1.0], [2.0]],
        "means_fit": [[25.0], [25.0]],
        "stddevs_fit": [[3.0], [4.0]],
    }
    rms, ncomp = _compute_rms_and_ncomp(signals, data)
    assert len(rms) == 2
    assert len(ncomp) == 2
    assert all(ncomp == 1)


def test_annotate_panel() -> None:
    """_annotate_panel should add text annotations."""
    fig, ax = plt.subplots()
    rms = np.array([0.1, 0.3])
    ncomp = np.array([1, 3])
    _annotate_panel(ax, rms, ncomp)
    assert len(ax.texts) == 2
    plt.close(fig)


def test_build_ncomp_rms(docs_img_dir: Path) -> None:
    """_build_ncomp_rms should produce a two-panel figure."""
    rms = np.array([0.1, 0.15, 0.25, 0.3])
    ncomp = np.array([1, 2, 3, 4])
    fig = _build_ncomp_rms(rms, ncomp, rms, ncomp)
    assert fig is not None
    plt.close(fig)


def test_ncomp_rms_plot_cli(comparison_data_dir: Path, docs_img_dir: Path) -> None:
    """CLI should succeed with valid data dir."""
    runner = CliRunner()
    result = runner.invoke(main, ["ncomp-rms-plot", "--data-dir", str(comparison_data_dir)])
    assert result.exit_code == 0, result.output


def test_ncomp_rms_plot_cli_missing(tmp_path: Path) -> None:
    """CLI should fail when files are missing."""
    runner = CliRunner()
    result = runner.invoke(main, ["ncomp-rms-plot", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
