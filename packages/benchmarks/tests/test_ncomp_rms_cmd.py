"""Tests for benchmarks.commands.ncomp_rms_plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from benchmarks.cli import main
from benchmarks.commands.ncomp_rms_plot import (
    _annotate_panel,
    _build_ncomp_rms,
)
from click.testing import CliRunner
from matplotlib import pyplot as plt


def test_annotate_panel() -> None:
    """_annotate_panel should add text annotations."""
    fig, ax = plt.subplots()
    rms = np.array([0.1, 0.3])
    ncomp = np.array([1, 3])
    _annotate_panel(ax, rms, ncomp)
    assert len(ax.texts) == 2
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_build_ncomp_rms() -> None:
    """_build_ncomp_rms should produce a two-panel figure."""
    rms = np.array([0.1, 0.15, 0.25, 0.3])
    ncomp = np.array([1, 2, 3, 4])
    fig = _build_ncomp_rms(rms, ncomp, rms, ncomp)
    assert fig is not None
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_ncomp_rms_plot_cli(comparison_data_dir: Path) -> None:
    """CLI should succeed with valid data dir."""
    runner = CliRunner()
    result = runner.invoke(main, ["ncomp-rms-plot", "--data-dir", str(comparison_data_dir)])
    assert result.exit_code == 0, result.output


def test_ncomp_rms_plot_cli_missing(tmp_path: Path) -> None:
    """CLI should fail when files are missing."""
    runner = CliRunner()
    result = runner.invoke(main, ["ncomp-rms-plot", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
