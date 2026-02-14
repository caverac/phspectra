"""Tests for benchmarks._docker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from benchmarks._docker import DOCKER_CONTEXT, build_image, run_gausspyplus


def test_docker_context_exists() -> None:
    """Docker context path should point to the benchmarks package root."""
    assert DOCKER_CONTEXT.endswith("benchmarks")


def test_build_image_success() -> None:
    """build_image should succeed when docker build returns 0."""
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("benchmarks._docker.subprocess.run", return_value=fake_result) as mock_run:
        build_image(context="/tmp/ctx", image="test-img")
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][:3] == ["docker", "build", "-t"]


def test_build_image_default_context() -> None:
    """build_image should use DOCKER_CONTEXT when context is None."""
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("benchmarks._docker.subprocess.run", return_value=fake_result) as mock_run:
        build_image()
    cmd = mock_run.call_args[0][0]
    assert cmd[-1].endswith("benchmarks")


def test_build_image_failure() -> None:
    """build_image should call sys.exit(1) on failure."""
    fake_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error msg")
    with (
        patch("benchmarks._docker.subprocess.run", return_value=fake_result),
        pytest.raises(SystemExit, match="1"),
    ):
        build_image(context="/tmp/ctx")


def test_run_gausspyplus_success(tmp_path: Path) -> None:
    """run_gausspyplus should return parsed results.json."""
    results = {"amplitudes_fit": [[1.0]], "means_fit": [[25.0]], "stddevs_fit": [[3.0]]}
    (tmp_path / "results.json").write_text(json.dumps(results))

    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("benchmarks._docker.subprocess.run", return_value=fake_result):
        out = run_gausspyplus(str(tmp_path))
    assert out == results


def test_run_gausspyplus_failure(tmp_path: Path) -> None:
    """run_gausspyplus should sys.exit(1) on failure."""
    fake_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="fail")
    with (
        patch("benchmarks._docker.subprocess.run", return_value=fake_result),
        pytest.raises(SystemExit, match="1"),
    ):
        run_gausspyplus(str(tmp_path))


def test_run_gausspyplus_stderr(tmp_path: Path) -> None:
    """run_gausspyplus should print stderr lines when present."""
    results = {"ok": True}
    (tmp_path / "results.json").write_text(json.dumps(results))

    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="warning line")
    with patch("benchmarks._docker.subprocess.run", return_value=fake_result):
        out = run_gausspyplus(str(tmp_path))
    assert out == results
