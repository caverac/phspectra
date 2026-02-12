"""Tests for benchmarks._docker."""

from __future__ import annotations

from benchmarks._docker import DOCKER_CONTEXT, build_image, run_gausspyplus


def test_docker_context_exists() -> None:
    """Docker context path should point to the benchmarks package root."""
    # The path should end with packages/benchmarks
    assert DOCKER_CONTEXT.endswith("benchmarks")


def test_build_image_callable() -> None:
    """build_image should be callable."""
    assert callable(build_image)


def test_run_gausspyplus_callable() -> None:
    """run_gausspyplus should be callable."""
    assert callable(run_gausspyplus)
