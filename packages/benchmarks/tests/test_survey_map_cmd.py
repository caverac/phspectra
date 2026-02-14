"""Tests for benchmarks.commands.survey_map."""

from __future__ import annotations

from pathlib import Path

from benchmarks.cli import main
from benchmarks.commands.survey_map import _build_figure
from click.testing import CliRunner
from matplotlib import pyplot as plt


def test_build_figure(docs_img_dir: Path) -> None:
    """_build_figure should produce a 2x2 placeholder figure."""
    fig = _build_figure()
    assert fig is not None
    plt.close(fig)


def test_survey_map_cli(docs_img_dir: Path) -> None:
    """CLI should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["survey-map"])
    assert result.exit_code == 0, result.output
