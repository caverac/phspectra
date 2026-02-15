"""Tests for benchmarks.commands.download."""

from __future__ import annotations

from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from benchmarks.cli import main
from click.testing import CliRunner


def test_download_cli(tmp_path: Path) -> None:
    """CLI should download FITS, catalog, and attempt pre-compute db."""
    with patch("benchmarks.commands.download.urlretrieve") as mock_retrieve:
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    # FITS + catalog (required) + pre-compute.db (optional attempt)
    assert mock_retrieve.call_count == 3


def test_download_cached(tmp_path: Path) -> None:
    """CLI should skip downloading when files are already cached."""
    (tmp_path / "grs-test-field.fits").write_text("fake")
    (tmp_path / "gausspy-catalog.votable").write_text("fake")
    db_dir = tmp_path / "compare-docker"
    db_dir.mkdir()
    (db_dir / "pre-compute.db").write_text("fake")

    with patch("benchmarks.commands.download.urlretrieve") as mock_retrieve:
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    mock_retrieve.assert_not_called()


def test_download_force(tmp_path: Path) -> None:
    """CLI --force should remove cached files before downloading."""
    (tmp_path / "grs-test-field.fits").write_text("fake")
    (tmp_path / "gausspy-catalog.votable").write_text("fake")

    with patch("benchmarks.commands.download.urlretrieve") as mock_retrieve:
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--cache-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    assert mock_retrieve.call_count == 3


def test_download_http_error(tmp_path: Path) -> None:
    """CLI should fail with clear error on HTTP error."""
    with patch(
        "benchmarks.commands.download.urlretrieve",
        side_effect=HTTPError("url", 404, "Not Found", Message(), None),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--cache-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_download_environment(tmp_path: Path) -> None:
    """CLI should use the specified environment in the S3 URL."""
    with patch("benchmarks.commands.download.urlretrieve") as mock_retrieve:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["download", "--cache-dir", str(tmp_path), "--environment", "production"],
        )
    assert result.exit_code == 0, result.output
    # Check that the production URL was used
    url_arg = mock_retrieve.call_args_list[0][0][0]
    assert "production" in url_arg


def test_download_precompute_server_error(tmp_path: Path) -> None:
    """CLI should fail when pre-compute.db download hits a non-404 error."""
    call_count = 0

    def _side_effect(url: str, _dest: str) -> None:
        nonlocal call_count
        call_count += 1
        if "pre-compute.db" in url:
            raise HTTPError(url, 500, "Internal Server Error", Message(), None)

    with patch("benchmarks.commands.download.urlretrieve", side_effect=_side_effect):
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--cache-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_download_precompute_skipped(tmp_path: Path) -> None:
    """CLI should skip pre-compute.db when not in bucket (404)."""

    def _side_effect(url: str, _dest: str) -> None:
        if "pre-compute.db" in url:
            raise HTTPError(url, 404, "Not Found", Message(), None)

    with patch("benchmarks.commands.download.urlretrieve", side_effect=_side_effect):
        runner = CliRunner()
        result = runner.invoke(main, ["download", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Skipped" in result.output
