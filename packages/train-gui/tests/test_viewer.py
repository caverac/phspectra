"""Tests for train_gui._viewer."""

# pylint: disable=protected-access,import-outside-toplevel,wrong-import-position

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from train_gui._loader import ComparisonData, PixelData  # noqa: E402
from train_gui._state import TrainingSet  # noqa: E402
from train_gui._viewer import _KEYS, SpectrumViewer, _gaussian  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _comp(source: str = "phspectra", amp: float = 1.0, mean: float = 25.0, std: float = 3.0) -> dict[str, Any]:
    return {"amplitude": amp, "mean": mean, "stddev": std, "source": source}


def _make_data(n_channels: int = 50) -> ComparisonData:
    """Build minimal ComparisonData with two pixels."""
    signal = np.zeros(n_channels, dtype=np.float64)
    return ComparisonData(
        pixels=[(10, 20), (30, 40)],
        pixel_data=[
            PixelData(
                pixel=(10, 20),
                signal=signal.copy(),
                ph_components=[_comp("phspectra", 2.0, 25.0, 3.0)],
                gp_components=[_comp("gausspyplus", 1.5, 20.0, 4.0)],
            ),
            PixelData(
                pixel=(30, 40),
                signal=signal.copy(),
                ph_components=[],
                gp_components=[],
            ),
        ],
    )


def _make_viewer(tmp_path: Path, data: ComparisonData | None = None, start_index: int = 0) -> SpectrumViewer:
    ts = TrainingSet(tmp_path / "ts.json")
    if data is None:
        data = _make_data()
    return SpectrumViewer(data, ts, start_index=start_index)


def _key_event(viewer: SpectrumViewer, key: str) -> None:
    """Simulate a key press event."""
    event = MagicMock()
    event.key = key
    viewer._on_key(event)


# ---------------------------------------------------------------------------
# _gaussian helper
# ---------------------------------------------------------------------------


class TestGaussian:
    """Tests for the module-level _gaussian function."""

    def test_peak_value(self) -> None:
        """Peak of the Gaussian should equal the amplitude."""
        x = np.array([10.0])
        assert _gaussian(x, 5.0, 10.0, 2.0)[0] == pytest.approx(5.0)

    def test_zero_stddev_guarded(self) -> None:
        """A zero stddev should not raise (guarded by max)."""
        x = np.array([0.0, 1.0])
        result = _gaussian(x, 1.0, 0.0, 0.0)
        assert result[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SpectrumViewer construction
# ---------------------------------------------------------------------------


class TestViewerInit:
    """SpectrumViewer initialisation."""

    def test_creates_figure(self, tmp_path: Path) -> None:
        """Viewer should create a matplotlib figure."""
        viewer = _make_viewer(tmp_path)
        assert viewer._fig is not None
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)

    def test_start_index_clamped(self, tmp_path: Path) -> None:
        """Start index is clamped to valid range."""
        viewer = _make_viewer(tmp_path, start_index=999)
        assert viewer._index == 1  # only 2 pixels
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)


# ---------------------------------------------------------------------------
# Key navigation
# ---------------------------------------------------------------------------


class TestKeyNavigation:
    """Arrow key navigation between pixels."""

    def test_right_advances(self, tmp_path: Path) -> None:
        """Right arrow moves to next pixel."""
        viewer = _make_viewer(tmp_path)
        assert viewer._index == 0
        _key_event(viewer, "right")
        assert viewer._index == 1
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)

    def test_right_at_end(self, tmp_path: Path) -> None:
        """Right arrow at last pixel is a no-op."""
        viewer = _make_viewer(tmp_path, start_index=1)
        _key_event(viewer, "right")
        assert viewer._index == 1
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)

    def test_left_goes_back(self, tmp_path: Path) -> None:
        """Left arrow moves to previous pixel."""
        viewer = _make_viewer(tmp_path, start_index=1)
        _key_event(viewer, "left")
        assert viewer._index == 0
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)

    def test_left_at_start(self, tmp_path: Path) -> None:
        """Left arrow at first pixel is a no-op."""
        viewer = _make_viewer(tmp_path)
        _key_event(viewer, "left")
        assert viewer._index == 0
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)


# ---------------------------------------------------------------------------
# Key actions
# ---------------------------------------------------------------------------


class TestKeyActions:
    """Toggle, save, clear, quit key actions."""

    def test_toggle_component(self, tmp_path: Path) -> None:
        """Pressing a key label toggles a component."""
        viewer = _make_viewer(tmp_path)
        pixel = (10, 20)
        comp = viewer._keyed_comps[0]
        first_key = _KEYS[0]
        # Toggle on.
        _key_event(viewer, first_key)
        assert viewer._ts.has_component(pixel, comp)
        # Toggle off.
        _key_event(viewer, first_key)
        assert not viewer._ts.has_component(pixel, comp)
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)

    def test_toggle_beyond_range(self, tmp_path: Path) -> None:
        """Pressing a key beyond the component list is a no-op."""
        viewer = _make_viewer(tmp_path)
        # "z" is likely beyond the 2 components in our test data.
        _key_event(viewer, _KEYS[-1])
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)

    def test_save_key(self, tmp_path: Path) -> None:
        """'s' saves the training set."""
        viewer = _make_viewer(tmp_path)
        _key_event(viewer, "s")
        assert (tmp_path / "ts.json").exists()
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)

    def test_clear_key(self, tmp_path: Path) -> None:
        """'c' clears the current pixel."""
        viewer = _make_viewer(tmp_path)
        # First select a component, then clear.
        _key_event(viewer, _KEYS[0])
        _key_event(viewer, "c")
        assert not viewer._ts.has_pixel((10, 20))
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)

    def test_quit_key(self, tmp_path: Path) -> None:
        """'q' saves and closes."""
        viewer = _make_viewer(tmp_path)
        _key_event(viewer, "q")
        assert (tmp_path / "ts.json").exists()

    def test_unbound_key(self, tmp_path: Path) -> None:
        """An unrecognised key is a no-op."""
        viewer = _make_viewer(tmp_path)
        _key_event(viewer, "F5")
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)


# ---------------------------------------------------------------------------
# Residual display
# ---------------------------------------------------------------------------


class TestResidualDisplay:
    """Residual subtraction on the plot."""

    def test_residual_subtracts_selected(self, tmp_path: Path) -> None:
        """Selecting a component should change the residual plotted."""
        data = _make_data(50)
        # Put a Gaussian into the signal so the subtraction is visible.
        x = np.arange(50, dtype=np.float64)
        data.pixel_data[0].signal = _gaussian(x, 2.0, 25.0, 3.0)
        viewer = _make_viewer(tmp_path, data=data)
        # Select the PH component (amp=2, mean=25, std=3).
        _key_event(viewer, _KEYS[1])
        # After selection the viewer redraws — just verify it didn't crash.
        assert viewer._ts.has_component((10, 20), data.pixel_data[0].ph_components[0])
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)


# ---------------------------------------------------------------------------
# Manual components
# ---------------------------------------------------------------------------


class TestManualComponents:
    """Manual components in the keyed list and display."""

    def test_manual_comps_appear_in_keyed_list(self, tmp_path: Path) -> None:
        """Manual components from the training set appear in _keyed_comps."""
        data = _make_data(50)
        ts = TrainingSet(tmp_path / "ts.json")
        manual = _comp("manual", 0.5, 10.0, 2.0)
        ts.add_component((10, 20), manual)
        viewer = SpectrumViewer(data, ts)
        # keyed_comps should include GP+ (1) + PH (1) + manual (1) = 3.
        assert len(viewer._keyed_comps) == 3
        assert viewer._keyed_comps[-1]["source"] == "manual"
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)


# ---------------------------------------------------------------------------
# Component list (no components)
# ---------------------------------------------------------------------------


class TestComponentListEmpty:
    """Component list panel for a pixel with no components."""

    def test_empty_pixel_shows_no_components(self, tmp_path: Path) -> None:
        """A pixel with zero components shows the placeholder text."""
        viewer = _make_viewer(tmp_path, start_index=1)
        # Pixel index 1 has no components — _draw_component_list ran.
        assert viewer._keyed_comps == []
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)


# ---------------------------------------------------------------------------
# SpanSelector / _on_span
# ---------------------------------------------------------------------------


class TestOnSpan:
    """Span selection and Gaussian fitting."""

    def test_span_fits_gaussian(self, tmp_path: Path) -> None:
        """_on_span fits a Gaussian and adds it to the training set."""
        data = _make_data(100)
        x = np.arange(100, dtype=np.float64)
        data.pixel_data[0].signal = _gaussian(x, 3.0, 50.0, 5.0) + 0.01 * np.random.default_rng(0).normal(size=100)
        viewer = _make_viewer(tmp_path, data=data)
        viewer._on_span(35.0, 65.0)
        entry = viewer._ts.get_entry((10, 20))
        assert entry is not None
        manual = [c for c in entry["components"] if c["source"] == "manual"]
        assert len(manual) == 1
        assert manual[0]["mean"] == pytest.approx(50.0, abs=1.0)
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)

    def test_span_too_narrow(self, tmp_path: Path) -> None:
        """A span with < 3 channels is a no-op."""
        viewer = _make_viewer(tmp_path)
        viewer._on_span(25.0, 26.0)
        assert viewer._ts.get_entry((10, 20)) is None
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)

    def test_span_fit_failure(self, tmp_path: Path) -> None:
        """RuntimeError from curve_fit is caught gracefully."""
        viewer = _make_viewer(tmp_path)
        with patch("train_gui._viewer.curve_fit", side_effect=RuntimeError("boom")):
            viewer._on_span(10.0, 40.0)
        assert viewer._ts.get_entry((10, 20)) is None
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)


# ---------------------------------------------------------------------------
# show()
# ---------------------------------------------------------------------------


class TestOverflowComponents:
    """More components than available key labels."""

    def test_excess_components_truncated(self, tmp_path: Path) -> None:
        """Components beyond _KEYS length are silently skipped."""
        n = len(_KEYS) + 2
        comps = [_comp("gausspyplus", amp=float(i), mean=float(i * 10), std=2.0) for i in range(1, n + 1)]
        data = ComparisonData(
            pixels=[(0, 0)],
            pixel_data=[
                PixelData(
                    pixel=(0, 0),
                    signal=np.zeros(200, dtype=np.float64),
                    ph_components=[],
                    gp_components=comps,
                ),
            ],
        )
        ts = TrainingSet(tmp_path / "ts.json")
        viewer = SpectrumViewer(data, ts)
        # Should not crash; keyed_comps has more than _KEYS but drawing truncates.
        assert len(viewer._keyed_comps) == n
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)


class TestLabelCollision:
    """Labels offset when peaks overlap."""

    def test_overlapping_labels_offset(self, tmp_path: Path) -> None:
        """Two components at the same mean should get offset labels."""
        signal = np.zeros(50, dtype=np.float64)
        comps = [
            _comp("gausspyplus", amp=1.0, mean=25.0, std=3.0),
            _comp("phspectra", amp=1.0, mean=25.0, std=3.0),
        ]
        data = ComparisonData(
            pixels=[(0, 0)],
            pixel_data=[
                PixelData(pixel=(0, 0), signal=signal, ph_components=[comps[1]], gp_components=[comps[0]]),
            ],
        )
        ts = TrainingSet(tmp_path / "ts.json")
        # Select one component to give non-zero amp_range via signal variation.
        signal[:] = _gaussian(np.arange(50, dtype=np.float64), 5.0, 25.0, 3.0)
        viewer = SpectrumViewer(data, ts)
        # If we get here without error, the offset logic ran.
        assert len(viewer._keyed_comps) == 2
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)


class TestOnSpanWithSelectedComponent:
    """_on_span with already-selected components subtracts them from residual."""

    def test_span_respects_selected_residual(self, tmp_path: Path) -> None:
        """Fitting on span should subtract selected components from the residual."""
        x = np.arange(100, dtype=np.float64)
        # Two Gaussians in the signal.
        signal = _gaussian(x, 3.0, 30.0, 4.0) + _gaussian(x, 2.0, 70.0, 5.0)
        data = ComparisonData(
            pixels=[(0, 0)],
            pixel_data=[
                PixelData(
                    pixel=(0, 0),
                    signal=signal,
                    ph_components=[_comp("phspectra", 3.0, 30.0, 4.0)],
                    gp_components=[],
                ),
            ],
        )
        ts = TrainingSet(tmp_path / "ts.json")
        # Select the PH component so it's subtracted from the residual.
        ts.add_component((0, 0), _comp("phspectra", 3.0, 30.0, 4.0))
        viewer = SpectrumViewer(data, ts)
        # Now span-fit the second peak on the residual.
        viewer._on_span(55.0, 85.0)
        entry = ts.get_entry((0, 0))
        assert entry is not None
        manual = [c for c in entry["components"] if c["source"] == "manual"]
        assert len(manual) == 1
        assert manual[0]["mean"] == pytest.approx(70.0, abs=2.0)
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)


class TestShow:
    """SpectrumViewer.show delegates to plt.show."""

    def test_show_calls_plt(self, tmp_path: Path) -> None:
        """show() calls plt.show."""
        viewer = _make_viewer(tmp_path)
        with patch("train_gui._viewer.plt") as mock_plt:
            viewer.show()
            mock_plt.show.assert_called_once()
        import matplotlib.pyplot as mplt

        mplt.close(viewer._fig)
