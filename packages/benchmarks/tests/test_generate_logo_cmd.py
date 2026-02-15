"""Tests for benchmarks.commands.generate_logo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from benchmarks.cli import main
from benchmarks.commands.generate_logo import (
    _build_signal,
    _find_peak,
    _path_d,
    _render_svg,
    _to_svg,
    _write_if_changed,
)
from click.testing import CliRunner


class TestBuildSignal:
    """Tests for _build_signal."""

    def test_shape_and_dtype(self) -> None:
        """Should return arrays of expected shape and dtype."""
        rng = np.random.default_rng(0)
        t, noisy = _build_signal(rng, 0.045)
        assert t.shape == (350,)
        assert noisy.shape == (350,)
        assert t.dtype == np.float64
        assert noisy.dtype == np.float64

    def test_non_negative(self) -> None:
        """Clipping should keep all values >= 0."""
        rng = np.random.default_rng(0)
        _, noisy = _build_signal(rng, 0.045)
        assert np.all(noisy >= 0)

    def test_has_peaks(self) -> None:
        """Signal should contain peaks well above the noise floor."""
        rng = np.random.default_rng(0)
        _, noisy = _build_signal(rng, 0.045)
        assert noisy.max() > 0.5


class TestToSvg:
    """Tests for _to_svg."""

    def test_x_range(self) -> None:
        """x should map [0, 1] to [80, 432]."""
        t = np.array([0.0, 0.5, 1.0])
        sig = np.array([0.0, 1.0, 0.0])
        x, _ = _to_svg(t, sig)
        assert x[0] == pytest.approx(80.0)
        assert x[-1] == pytest.approx(432.0)

    def test_y_baseline(self) -> None:
        """Zero signal should map to y=400 (baseline)."""
        t = np.array([0.0])
        sig = np.array([0.0])
        _, y = _to_svg(t, sig)
        assert y[0] == pytest.approx(400.0)


class TestPathD:
    """Tests for _path_d."""

    def test_open_path(self) -> None:
        """Open path should start with M and have no Z."""
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([1.0, 2.0, 3.0])
        d = _path_d(x, y)
        assert d.startswith("M10.0,1.0")
        assert "Z" not in d
        assert "L20.0,2.0" in d

    def test_closed_path(self) -> None:
        """Closed path should end with Z."""
        x = np.array([10.0, 20.0])
        y = np.array([1.0, 2.0])
        d = _path_d(x, y, closed=True)
        assert d.endswith("Z")


class TestFindPeak:
    """Tests for _find_peak."""

    def test_finds_argmax_in_range(self) -> None:
        """Should return the index of the maximum within the fractional range."""
        sig = np.zeros(350)
        sig[120] = 5.0
        idx = _find_peak(sig, 0.28, 0.43)
        assert idx == 120


class TestRenderSvg:
    """Tests for _render_svg."""

    def test_returns_valid_svg(self) -> None:
        """Rendered SVG should contain expected elements."""
        rng = np.random.default_rng(42)
        t, noisy = _build_signal(rng, 0.045)
        x_svg, y_svg = _to_svg(t, noisy)
        svg = _render_svg(noisy, x_svg, y_svg)
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "specFill" in svg
        assert "FF6F61" in svg  # peak marker colour


class TestWriteIfChanged:
    """Tests for _write_if_changed."""

    def test_writes_new_file(self, tmp_path: Path) -> None:
        """Should write and return True when file does not exist."""
        p = tmp_path / "new.svg"
        assert _write_if_changed(p, "<svg/>") is True
        assert p.read_text() == "<svg/>"

    def test_skips_unchanged(self, tmp_path: Path) -> None:
        """Should return False when content is identical."""
        p = tmp_path / "same.svg"
        p.write_text("<svg/>", encoding="utf-8")
        assert _write_if_changed(p, "<svg/>") is False

    def test_overwrites_changed(self, tmp_path: Path) -> None:
        """Should overwrite and return True when content differs."""
        p = tmp_path / "changed.svg"
        p.write_text("<svg>old</svg>", encoding="utf-8")
        assert _write_if_changed(p, "<svg>new</svg>") is True
        assert p.read_text() == "<svg>new</svg>"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Should create intermediate directories if missing."""
        p = tmp_path / "a" / "b" / "new.svg"
        assert _write_if_changed(p, "<svg/>") is True
        assert p.exists()


def test_generate_logo_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI should write both logo.svg and favicon.svg."""
    img_dir = tmp_path / "results"
    img_dir.mkdir()
    monkeypatch.setattr(
        "benchmarks.commands.generate_logo.DOCS_IMG_DIR",
        str(img_dir),
    )
    runner = CliRunner()
    result = runner.invoke(main, ["generate-logo"])
    assert result.exit_code == 0, result.output
    # DOCS_IMG_DIR is .../results; command uses .parent → tmp_path
    assert (tmp_path / "logo.svg").exists()
    assert (tmp_path / "favicon.svg").exists()
    svg = (tmp_path / "logo.svg").read_text()
    assert "<svg" in svg


def test_generate_logo_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running twice with same seed should show 'Unchanged' on second run."""
    img_dir = tmp_path / "results"
    img_dir.mkdir()
    monkeypatch.setattr(
        "benchmarks.commands.generate_logo.DOCS_IMG_DIR",
        str(img_dir),
    )
    runner = CliRunner()
    result1 = runner.invoke(main, ["generate-logo"])
    assert result1.exit_code == 0
    result2 = runner.invoke(main, ["generate-logo"])
    assert result2.exit_code == 0
    assert "Unchanged" in result2.output
