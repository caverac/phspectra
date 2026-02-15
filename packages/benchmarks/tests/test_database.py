"""Tests for benchmarks._database read helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks._database import (
    create_db,
    insert_phspectra_run,
    load_components,
    load_pixels,
    load_run,
)


def test_load_run_by_id(tmp_path: Path) -> None:
    """load_run with explicit run_id should return that run."""
    db_path = str(tmp_path / "test.db")
    conn = create_db(db_path)
    run_id = insert_phspectra_run(conn, beta=3.8, n_spectra=10, total_time_s=1.0)
    conn.close()

    row = load_run(db_path, "phspectra", run_id=run_id)
    assert row["run_id"] == run_id
    assert row["beta"] == 3.8


def test_load_run_missing(tmp_path: Path) -> None:
    """load_run should raise ValueError when no run exists."""
    db_path = str(tmp_path / "empty.db")
    create_db(db_path).close()

    with pytest.raises(ValueError, match="No phspectra run found"):
        load_run(db_path, "phspectra")


def test_load_pixels_empty_db(tmp_path: Path) -> None:
    """load_pixels should raise ValueError on empty database."""
    db_path = str(tmp_path / "empty.db")
    create_db(db_path).close()

    with pytest.raises(ValueError, match="No phspectra run found"):
        load_pixels(db_path, "phspectra")


def test_load_components_empty_db(tmp_path: Path) -> None:
    """load_components should raise ValueError on empty database."""
    db_path = str(tmp_path / "empty.db")
    create_db(db_path).close()

    with pytest.raises(ValueError, match="No gausspyplus run found"):
        load_components(db_path, "gausspyplus")
