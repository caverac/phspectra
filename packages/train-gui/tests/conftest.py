"""Shared fixtures for train-gui tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def comparison_dir(tmp_path: Path) -> Path:
    """Write minimal valid comparison data files."""
    n_spectra, n_channels = 2, 50
    rng = np.random.default_rng(42)
    signals = rng.normal(0, 0.1, (n_spectra, n_channels))
    np.savez(tmp_path / "spectra.npz", signals=signals)

    ph = {
        "beta": 3.8,
        "pixels": [[10, 20], [30, 40]],
        "amplitudes_fit": [[1.0], [2.0]],
        "means_fit": [[25.0], [25.0]],
        "stddevs_fit": [[3.0], [4.0]],
        "times": [0.01, 0.02],
        "total_time_s": 0.03,
        "mean_n_components": 1.0,
    }
    (tmp_path / "phspectra_results.json").write_text(json.dumps(ph))

    gp = {
        "amplitudes_fit": [[1.1], [1.9]],
        "means_fit": [[25.5], [24.5]],
        "stddevs_fit": [[3.1], [4.1]],
    }
    (tmp_path / "results.json").write_text(json.dumps(gp))

    return tmp_path
