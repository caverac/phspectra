"""Docker build/run helpers for GaussPy+ benchmarks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks._console import console, err_console
from benchmarks._constants import DOCKER_IMAGE

# Docker context is the benchmarks package root (where Dockerfile lives)
DOCKER_CONTEXT = str(Path(__file__).resolve().parent.parent.parent)


def build_image(context: str | None = None, image: str = DOCKER_IMAGE) -> None:
    """Build the GaussPy+ benchmark Docker image.

    Parameters
    ----------
    context : str or None
        Path to Docker build context.  Defaults to the benchmarks package root.
    image : str
        Docker image tag.
    """
    if context is None:
        context = DOCKER_CONTEXT
    context = os.path.abspath(context)
    console.print(f"  Building Docker image from [blue]{context}[/blue] ...")
    result = subprocess.run(
        ["docker", "build", "-t", image, context],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        err_console.print(f"  Docker build FAILED:\n{result.stderr}")
        sys.exit(1)
    console.print("  Docker image built.", style="green")


def run_gausspyplus(data_dir: str, image: str = DOCKER_IMAGE) -> dict:
    """Run GaussPy+ benchmark in Docker with data dir mounted.

    Parameters
    ----------
    data_dir : str
        Absolute path to the directory containing spectra.npz.
    image : str
        Docker image tag.

    Returns
    -------
    dict
        Parsed results.json from the container.
    """
    data_dir = os.path.abspath(data_dir)
    console.print("  Running GaussPy+ in Docker ...")
    result = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{data_dir}:/data", image],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            console.print(f"    \\[gpp] {line}")
    if result.returncode != 0:
        err_console.print(f"  Docker FAILED (exit {result.returncode})")
        sys.exit(1)
    with open(os.path.join(data_dir, "results.json"), encoding="utf-8") as f:
        return json.load(f)
