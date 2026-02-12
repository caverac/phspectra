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
    """CLI help should list all 8 commands."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in [
        "download",
        "compare",
        "compare-plot",
        "train-beta",
        "width",
        "inspect",
        "performance",
        "synthetic",
    ]:
        assert cmd in result.output, f"Missing command: {cmd}"


def test_download_help() -> None:
    """``benchmarks download --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["download", "--help"])
    assert result.exit_code == 0
    assert "--cache-dir" in result.output


def test_compare_help() -> None:
    """``benchmarks compare --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["compare", "--help"])
    assert result.exit_code == 0
    assert "--n-spectra" in result.output


def test_compare_plot_help() -> None:
    """``benchmarks compare-plot --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["compare-plot", "--help"])
    assert result.exit_code == 0
    assert "--data-dir" in result.output


def test_synthetic_help() -> None:
    """``benchmarks synthetic --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["synthetic", "--help"])
    assert result.exit_code == 0
    assert "--n-per-category" in result.output


def test_train_beta_help() -> None:
    """``benchmarks train-beta --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["train-beta", "--help"])
    assert result.exit_code == 0
    assert "--beta-min" in result.output


def test_width_help() -> None:
    """``benchmarks width --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["width", "--help"])
    assert result.exit_code == 0
    assert "--beta" in result.output


def test_inspect_help() -> None:
    """``benchmarks inspect --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["inspect", "--help"])
    assert result.exit_code == 0
    assert "PX" in result.output


def test_performance_help() -> None:
    """``benchmarks performance --help`` should succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["performance", "--help"])
    assert result.exit_code == 0
    assert "--n-spectra" in result.output
