"""Tests for benchmarks.commands.train_beta (now the ``train`` command)."""

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
def test_train_cli(comparison_data_dir: Path) -> None:
    """CLI should succeed with valid data, training set, and small sweep."""
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
            ],
        },
    ]
    ts_path = comparison_data_dir / "training_set.json"
    ts_path.write_text(json.dumps(ts))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "train",
            "--data-dir",
            str(comparison_data_dir),
            "--training-set",
            str(ts_path),
            "--beta-steps",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output


def test_train_cli_missing(tmp_path: Path) -> None:
    """CLI should fail when spectra.npz is missing."""
    ts_path = tmp_path / "ts.json"
    ts_path.write_text(json.dumps([{"pixel": [0, 0], "components": [{"amplitude": 1, "mean": 25, "stddev": 3}]}]))

    runner = CliRunner()
    result = runner.invoke(main, ["train", "--data-dir", str(tmp_path), "--training-set", str(ts_path)])
    assert result.exit_code != 0


def test_train_cli_requires_training_set() -> None:
    """CLI should fail when --training-set is not provided."""
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--data-dir", "/tmp"])
    assert result.exit_code != 0
    assert "training-set" in result.output.lower() or "required" in result.output.lower()


@pytest.mark.usefixtures("docs_img_dir")
def test_train_with_training_set(comparison_data_dir: Path) -> None:
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
            "train",
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
def test_train_training_set_skips_unknown_pixels(comparison_data_dir: Path) -> None:
    """Pixels in training set but not in pre-compute data should be skipped."""
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
            "train",
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
    assert "1 pixel(s) not in pre-compute data" in result.output


def test_train_training_set_missing_data(tmp_path: Path) -> None:
    """CLI should fail when --training-set is given but no pixel data exists."""
    np.savez(tmp_path / "spectra.npz", signals=np.zeros((2, 50)))
    ts_path = tmp_path / "ts.json"
    ts_path.write_text(json.dumps([{"pixel": [0, 0], "components": [{"amplitude": 1, "mean": 25, "stddev": 3}]}]))

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["train", "--data-dir", str(tmp_path), "--training-set", str(ts_path)],
    )
    assert result.exit_code != 0


def test_train_training_set_empty_components(comparison_data_dir: Path) -> None:
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
        ["train", "--data-dir", str(comparison_data_dir), "--training-set", str(ts_path)],
    )
    assert result.exit_code != 0  # no training spectra -> error


@pytest.mark.usefixtures("docs_img_dir")
def test_train_json_fallback(comparison_data_dir: Path) -> None:
    """CLI should fall back to phspectra_results.json when pre-compute.db is missing."""
    # Remove the SQLite database so it falls back to JSON
    db_file = comparison_data_dir / "pre-compute.db"
    if db_file.exists():
        db_file.unlink()

    ts = [
        {
            "survey": "GRS",
            "pixel": [10, 20],
            "components": [
                {"amplitude": 1.0, "mean": 25.0, "stddev": 3.0, "source": "gausspyplus"},
            ],
        },
    ]
    ts_path = comparison_data_dir / "training_set.json"
    ts_path.write_text(json.dumps(ts))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "train",
            "--data-dir",
            str(comparison_data_dir),
            "--training-set",
            str(ts_path),
            "--beta-steps",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.usefixtures("docs_img_dir")
def test_train_linalg_error(comparison_data_dir: Path) -> None:
    """CLI should handle LinAlgError from fit_gaussians."""
    ts = [
        {
            "survey": "GRS",
            "pixel": [10, 20],
            "components": [{"amplitude": 1, "mean": 25, "stddev": 3}],
        },
    ]
    ts_path = comparison_data_dir / "ts.json"
    ts_path.write_text(json.dumps(ts))

    with patch(
        "benchmarks.commands.train_beta.fit_gaussians",
        side_effect=ValueError("test"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "train",
                "--data-dir",
                str(comparison_data_dir),
                "--training-set",
                str(ts_path),
                "--beta-steps",
                "2",
            ],
        )
    assert result.exit_code == 0, result.output
