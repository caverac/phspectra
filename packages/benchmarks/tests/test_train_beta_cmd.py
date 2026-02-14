"""Tests for benchmarks.commands.train_beta."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from benchmarks._types import BetaSweepResult
from benchmarks.cli import main
from benchmarks.commands.train_beta import _plot_beta_sweep
from click.testing import CliRunner
from matplotlib import pyplot as plt


@pytest.mark.usefixtures("docs_img_dir")
def test_plot_beta_sweep() -> None:
    """_plot_beta_sweep should produce F1/P/R figure."""
    results = [
        BetaSweepResult(3.8, 0.8, 0.9, 0.7, 10, 12, 11, 1.0, 0.1, 0.15, 8),
        BetaSweepResult(4.0, 0.85, 0.88, 0.82, 11, 12, 13, 1.1, 0.1, 0.15, 9),
    ]
    fig = _plot_beta_sweep(results)
    assert fig is not None
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_train_beta_cli(comparison_data_dir: Path) -> None:
    """CLI should succeed with valid data and small sweep."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["train-beta", "--data-dir", str(comparison_data_dir), "--beta-steps", "2"],
    )
    assert result.exit_code == 0, result.output


def test_train_beta_cli_missing(tmp_path: Path) -> None:
    """CLI should fail when files are missing."""
    runner = CliRunner()
    result = runner.invoke(main, ["train-beta", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0


@pytest.mark.usefixtures("docs_img_dir")
def test_train_beta_skips_empty_gp(tmp_path: Path) -> None:
    """CLI should skip spectra where GP+ found no components."""
    n_spectra, n_channels = 3, 50
    signals = np.random.default_rng(0).normal(0, 0.1, (n_spectra, n_channels))
    np.savez(tmp_path / "spectra.npz", signals=signals)

    gp = {
        "amplitudes_fit": [[1.0], [], [2.0]],
        "means_fit": [[25.0], [], [25.0]],
        "stddevs_fit": [[3.0], [], [4.0]],
    }
    (tmp_path / "results.json").write_text(json.dumps(gp))

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["train-beta", "--data-dir", str(tmp_path), "--beta-steps", "2"],
    )
    assert result.exit_code == 0, result.output
    assert "2 spectra with GaussPy+ components" in result.output


@pytest.mark.usefixtures("docs_img_dir")
def test_train_beta_with_training_set(comparison_data_dir: Path) -> None:
    """CLI should accept --training-set and use curated components as reference."""
    ts = [
        {
            "survey": "GRS",
            "pixel": [10, 20],
            "components": [
                {"amplitude": 1.0, "mean": 25.0, "stddev": 3.0, "source": "gausspyplus"},
            ],
        },
        {
            "survey": "GRS",
            "pixel": [30, 40],
            "components": [
                {"amplitude": 2.0, "mean": 25.0, "stddev": 4.0, "source": "phspectra"},
                {"amplitude": 0.5, "mean": 10.0, "stddev": 2.0, "source": "manual"},
            ],
        },
    ]
    ts_path = comparison_data_dir / "training_set.json"
    ts_path.write_text(json.dumps(ts))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "train-beta",
            "--data-dir",
            str(comparison_data_dir),
            "--training-set",
            str(ts_path),
            "--beta-steps",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 curated spectra" in result.output
    assert "3 reference components" in result.output


@pytest.mark.usefixtures("docs_img_dir")
def test_train_beta_training_set_skips_unknown_pixels(comparison_data_dir: Path) -> None:
    """Pixels in training set but not in compare data should be skipped."""
    ts = [
        {
            "survey": "GRS",
            "pixel": [10, 20],
            "components": [
                {"amplitude": 1.0, "mean": 25.0, "stddev": 3.0, "source": "manual"},
            ],
        },
        {
            "survey": "GRS",
            "pixel": [99, 99],
            "components": [
                {"amplitude": 1.0, "mean": 25.0, "stddev": 3.0, "source": "manual"},
            ],
        },
    ]
    ts_path = comparison_data_dir / "training_set.json"
    ts_path.write_text(json.dumps(ts))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "train-beta",
            "--data-dir",
            str(comparison_data_dir),
            "--training-set",
            str(ts_path),
            "--beta-steps",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1 curated spectra" in result.output
    assert "1 pixel(s) not in compare data" in result.output


def test_train_beta_training_set_missing_phresults(tmp_path: Path) -> None:
    """CLI should fail when --training-set is given but phspectra_results.json is missing."""
    np.savez(tmp_path / "spectra.npz", signals=np.zeros((2, 50)))
    ts_path = tmp_path / "ts.json"
    ts_path.write_text(json.dumps([{"pixel": [0, 0], "components": [{"amplitude": 1, "mean": 25, "stddev": 3}]}]))

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["train-beta", "--data-dir", str(tmp_path), "--training-set", str(ts_path)],
    )
    assert result.exit_code != 0


def test_train_beta_gp_results_missing(tmp_path: Path) -> None:
    """CLI should fail when results.json is missing (GP+ path)."""
    np.savez(tmp_path / "spectra.npz", signals=np.zeros((2, 50)))

    runner = CliRunner()
    result = runner.invoke(main, ["train-beta", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_train_beta_training_set_empty_components(comparison_data_dir: Path) -> None:
    """Entries with no components should be skipped."""
    ts = [
        {"survey": "GRS", "pixel": [10, 20], "components": []},
        {"survey": "GRS", "pixel": [30, 40], "components": []},
    ]
    ts_path = comparison_data_dir / "ts.json"
    ts_path.write_text(json.dumps(ts))

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["train-beta", "--data-dir", str(comparison_data_dir), "--training-set", str(ts_path)],
    )
    assert result.exit_code != 0  # no training spectra → error


@pytest.mark.usefixtures("docs_img_dir")
def test_train_beta_linalg_error(tmp_path: Path) -> None:
    """CLI should handle LinAlgError from fit_gaussians."""
    n_spectra, n_channels = 2, 50
    signals = np.random.default_rng(0).normal(0, 0.1, (n_spectra, n_channels))
    np.savez(tmp_path / "spectra.npz", signals=signals)

    gp = {
        "amplitudes_fit": [[1.0], [2.0]],
        "means_fit": [[25.0], [25.0]],
        "stddevs_fit": [[3.0], [4.0]],
    }
    (tmp_path / "results.json").write_text(json.dumps(gp))

    with patch(
        "benchmarks.commands.train_beta.fit_gaussians",
        side_effect=ValueError("test"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["train-beta", "--data-dir", str(tmp_path), "--beta-steps", "2"],
        )
    assert result.exit_code == 0, result.output
