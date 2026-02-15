"""Tests for benchmarks.commands.survey_map."""

from __future__ import annotations

import pytest
from benchmarks.cli import main
from benchmarks.commands.survey_map import _build_figure
from click.testing import CliRunner
from matplotlib import pyplot as plt


@pytest.mark.usefixtures("docs_img_dir")
def test_build_figure() -> None:
    """_build_figure should produce a 2x2 placeholder figure."""
    fig = _build_figure()
    assert fig is not None
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_survey_map_cli() -> None:
    """CLI should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["survey-map-plot"])
    assert result.exit_code == 0, result.output
