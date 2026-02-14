"""Tests for benchmarks.commands.performance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from benchmarks.cli import main
from benchmarks.commands.performance import _plot_timing
from click.testing import CliRunner
from matplotlib import pyplot as plt


@pytest.mark.usefixtures("docs_img_dir")
def test_plot_timing() -> None:
    """_plot_timing should produce a histogram figure."""
    ph_ms = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    gp_ms = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    fig = _plot_timing(ph_ms, gp_ms)
    assert fig is not None
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_performance_plot_cli(comparison_data_dir: Path) -> None:
    """CLI should succeed with valid data dir."""
    runner = CliRunner()
    result = runner.invoke(main, ["performance-plot", "--data-dir", str(comparison_data_dir)])
    assert result.exit_code == 0, result.output


def test_performance_plot_cli_missing(tmp_path: Path) -> None:
    """CLI should fail when files are missing."""
    runner = CliRunner()
    result = runner.invoke(main, ["performance-plot", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
