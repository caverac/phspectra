"""Tests for the benchmarks CLI using Click testing."""

from __future__ import annotations

from benchmarks.cli import main
from click.testing import CliRunner


def test_cli_help() -> None:
    """``benchmarks --help`` should succeed and list commands."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Benchmark suite" in result.output


def test_cli_lists_commands() -> None:
    """CLI help should list all registered commands."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in [
        "download",
        "pre-compute",
        "compare-plot",
        "train",
        "inspect",
        "performance-plot",
        "train-synthetic",
        "persistence-plot",
        "ncomp-rms-plot",
        "survey-map-plot",
        "pipeline",
        "grs-map-plot",
        "correlation-plot",
    ]:
        assert cmd in result.output, f"Missing command: {cmd}"


def test_download_help() -> None:
    """``benchmarks download --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["download", "--help"])
    assert result.exit_code == 0
    assert "--cache-dir" in result.output


def test_pre_compute_help() -> None:
    """``benchmarks pre-compute --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["pre-compute", "--help"])
    assert result.exit_code == 0
    assert "--n-spectra" in result.output


def test_compare_plot_help() -> None:
    """``benchmarks compare-plot --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["compare-plot", "--help"])
    assert result.exit_code == 0
    assert "--data-dir" in result.output


def test_train_synthetic_help() -> None:
    """``benchmarks train-synthetic --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["train-synthetic", "--help"])
    assert result.exit_code == 0
    assert "--n-per-category" in result.output


def test_train_help() -> None:
    """``benchmarks train --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "--beta-min" in result.output


def test_inspect_help() -> None:
    """``benchmarks inspect --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["inspect", "--help"])
    assert result.exit_code == 0
    assert "PX" in result.output


def test_performance_plot_help() -> None:
    """``benchmarks performance-plot --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["performance-plot", "--help"])
    assert result.exit_code == 0
    assert "--data-dir" in result.output


def test_persistence_plot_help() -> None:
    """``benchmarks persistence-plot --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["persistence-plot", "--help"])
    assert result.exit_code == 0


def test_ncomp_rms_plot_help() -> None:
    """``benchmarks ncomp-rms-plot --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["ncomp-rms-plot", "--help"])
    assert result.exit_code == 0


def test_survey_map_plot_help() -> None:
    """``benchmarks survey-map-plot --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["survey-map-plot", "--help"])
    assert result.exit_code == 0


def test_grs_map_plot_help() -> None:
    """``benchmarks grs-map-plot --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["grs-map-plot", "--help"])
    assert result.exit_code == 0


def test_correlation_plot_help() -> None:
    """``benchmarks correlation-plot --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["correlation-plot", "--help"])
    assert result.exit_code == 0
