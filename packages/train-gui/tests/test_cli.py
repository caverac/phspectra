"""Tests for train_gui.cli."""

# pylint: disable=protected-access

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from train_gui.cli import main


def test_cli_loads_and_launches(comparison_dir: Path) -> None:
    """CLI should load data and create SpectrumViewer."""
    ts_path = comparison_dir / "training_set.json"

    mock_viewer_instance = MagicMock()
    with patch("train_gui._viewer.SpectrumViewer", return_value=mock_viewer_instance) as mock_cls:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--data-dir",
                str(comparison_dir),
                "--training-set",
                str(ts_path),
                "--start-index",
                "0",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_cls.assert_called_once()
    mock_viewer_instance.show.assert_called_once()


def test_cli_default_survey(comparison_dir: Path) -> None:
    """Default survey is GRS."""
    ts_path = comparison_dir / "training_set.json"

    with patch("train_gui._viewer.SpectrumViewer", return_value=MagicMock()) as mock_cls:
        runner = CliRunner()
        runner.invoke(
            main,
            [
                "--data-dir",
                str(comparison_dir),
                "--training-set",
                str(ts_path),
            ],
        )

    _, kwargs = mock_cls.call_args
    ts_arg = kwargs.get("training_set") or mock_cls.call_args[0][1]
    assert ts_arg._survey == "GRS"


def test_cli_custom_survey(comparison_dir: Path) -> None:
    """--survey option is passed through."""
    ts_path = comparison_dir / "training_set.json"

    with patch("train_gui._viewer.SpectrumViewer", return_value=MagicMock()) as mock_cls:
        runner = CliRunner()
        runner.invoke(
            main,
            [
                "--data-dir",
                str(comparison_dir),
                "--training-set",
                str(ts_path),
                "--survey",
                "VGPS",
            ],
        )

    _, kwargs = mock_cls.call_args
    ts_arg = kwargs.get("training_set") or mock_cls.call_args[0][1]
    assert ts_arg._survey == "VGPS"
