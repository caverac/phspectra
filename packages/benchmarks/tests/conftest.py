"""Shared fixtures for benchmarks tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from benchmarks._database import (
    create_db,
    insert_components,
    insert_gausspyplus_run,
    insert_phspectra_run,
    insert_pixels,
)
from benchmarks._types import Component


@pytest.fixture()
def simple_components() -> list[Component]:
    """Two well-separated Gaussian components."""
    return [
        Component(amplitude=5.0, mean=100.0, stddev=5.0),
        Component(amplitude=3.0, mean=200.0, stddev=4.0),
    ]


@pytest.fixture()
def single_component() -> list[Component]:
    """A single Gaussian component."""
    return [Component(amplitude=5.0, mean=100.0, stddev=5.0)]


@pytest.fixture()
def channel_array() -> npt.NDArray[np.float64]:
    """Channel array with 424 channels."""
    return np.arange(424, dtype=np.float64)


@pytest.fixture()
def comparison_data_dir(tmp_path: Path) -> Path:
    """Write minimal valid spectra.npz, pre-compute.db, and legacy JSON files."""
    n_spectra, n_channels = 2, 50
    rng = np.random.default_rng(0)
    signals = rng.normal(0, 0.1, (n_spectra, n_channels))

    np.savez(tmp_path / "spectra.npz", signals=signals)

    # Legacy JSON files (kept for backward compat in train_beta fallback)
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
        "times": [0.5, 0.6],
        "total_time_s": 1.1,
        "mean_n_components": 1.0,
    }
    (tmp_path / "results.json").write_text(json.dumps(gp))

    # SQLite database (used by compare_plot, performance, ncomp_rms, inspect)
    db_path = str(tmp_path / "pre-compute.db")
    conn = create_db(db_path)

    ph_run_id = insert_phspectra_run(conn, beta=3.8, n_spectra=n_spectra, total_time_s=0.03)
    insert_components(conn, "phspectra_components", ph_run_id, 10, 20, [(1.0, 25.0, 3.0)])
    insert_components(conn, "phspectra_components", ph_run_id, 30, 40, [(2.0, 25.0, 4.0)])
    insert_pixels(
        conn,
        "phspectra_pixels",
        ph_run_id,
        [(10, 20, 1, 0.1, 0.01), (30, 40, 1, 0.12, 0.02)],
    )

    gp_run_id = insert_gausspyplus_run(
        conn,
        alpha1=2.89,
        alpha2=6.65,
        phase="two",
        n_spectra=n_spectra,
        total_time_s=1.1,
    )
    insert_components(conn, "gausspyplus_components", gp_run_id, 10, 20, [(1.1, 25.5, 3.1)])
    insert_components(conn, "gausspyplus_components", gp_run_id, 30, 40, [(1.9, 24.5, 4.1)])
    insert_pixels(
        conn,
        "gausspyplus_pixels",
        gp_run_id,
        [(10, 20, 1, 0.11, 0.5), (30, 40, 1, 0.13, 0.6)],
    )

    conn.commit()
    conn.close()

    return tmp_path


@pytest.fixture()
def docs_img_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Monkeypatch DOCS_IMG_DIR so @docs_figure writes to a tmp dir."""
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    monkeypatch.setattr("benchmarks._plotting.DOCS_IMG_DIR", str(img_dir))
    return img_dir
