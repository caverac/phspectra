"""Tests for train_gui._state."""

# pylint: disable=protected-access

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from train_gui._state import TrainingSet


def _comp(source: str = "phspectra", amp: float = 1.0, mean: float = 25.0, std: float = 3.0) -> dict[str, Any]:
    return {"amplitude": amp, "mean": mean, "stddev": std, "source": source}


class TestTrainingSet:
    """TrainingSet CRUD and persistence."""

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Start empty when the file does not exist."""
        ts = TrainingSet(tmp_path / "ts.json")
        assert ts._entries == []

    def test_load_existing_file(self, tmp_path: Path) -> None:
        """Load entries from an existing JSON file."""
        data = [{"survey": "GRS", "pixel": [1, 2], "components": [_comp()]}]
        (tmp_path / "ts.json").write_text(json.dumps(data))
        ts = TrainingSet(tmp_path / "ts.json")
        assert len(ts._entries) == 1

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Save creates parent directories if needed."""
        path = tmp_path / "sub" / "dir" / "ts.json"
        ts = TrainingSet(path)
        ts.save()
        assert path.exists()
        assert json.loads(path.read_text()) == []

    def test_save_round_trip(self, tmp_path: Path) -> None:
        """Save then reload preserves entries."""
        ts = TrainingSet(tmp_path / "ts.json")
        ts.add_component((1, 2), _comp())
        ts.save()
        ts2 = TrainingSet(tmp_path / "ts.json")
        assert ts2.has_component((1, 2), _comp())

    def test_has_pixel_empty(self, tmp_path: Path) -> None:
        """has_pixel returns False for unknown pixel."""
        ts = TrainingSet(tmp_path / "ts.json")
        assert not ts.has_pixel((0, 0))

    def test_has_pixel_with_components(self, tmp_path: Path) -> None:
        """has_pixel returns True when components exist."""
        ts = TrainingSet(tmp_path / "ts.json")
        ts.add_component((1, 2), _comp())
        assert ts.has_pixel((1, 2))

    def test_has_pixel_after_clear(self, tmp_path: Path) -> None:
        """has_pixel returns False after clearing."""
        ts = TrainingSet(tmp_path / "ts.json")
        ts.add_component((1, 2), _comp())
        ts.clear_pixel((1, 2))
        assert not ts.has_pixel((1, 2))

    def test_get_entry_returns_none(self, tmp_path: Path) -> None:
        """get_entry returns None for missing pixel."""
        ts = TrainingSet(tmp_path / "ts.json")
        assert ts.get_entry((0, 0)) is None

    def test_get_entry_returns_entry(self, tmp_path: Path) -> None:
        """get_entry returns the entry for a known pixel."""
        ts = TrainingSet(tmp_path / "ts.json")
        ts.add_component((5, 6), _comp())
        entry = ts.get_entry((5, 6))
        assert entry is not None
        assert entry["pixel"] == [5, 6]

    def test_curated_pixels(self, tmp_path: Path) -> None:
        """curated_pixels returns set of pixels with components."""
        ts = TrainingSet(tmp_path / "ts.json")
        ts.add_component((1, 2), _comp())
        ts.add_component((3, 4), _comp(amp=2.0))
        assert ts.curated_pixels == {(1, 2), (3, 4)}

    def test_curated_pixels_excludes_empty(self, tmp_path: Path) -> None:
        """curated_pixels excludes cleared pixels."""
        ts = TrainingSet(tmp_path / "ts.json")
        ts.add_component((1, 2), _comp())
        ts.clear_pixel((1, 2))
        assert ts.curated_pixels == set()

    def test_add_component_creates_entry(self, tmp_path: Path) -> None:
        """Adding to a new pixel creates the entry."""
        ts = TrainingSet(tmp_path / "ts.json")
        ts.add_component((1, 2), _comp())
        assert len(ts._entries) == 1
        assert ts._entries[0]["survey"] == "GRS"

    def test_add_component_custom_survey(self, tmp_path: Path) -> None:
        """Survey name is stored from constructor."""
        ts = TrainingSet(tmp_path / "ts.json", survey="VGPS")
        ts.add_component((1, 2), _comp())
        assert ts._entries[0]["survey"] == "VGPS"

    def test_add_component_deduplicates(self, tmp_path: Path) -> None:
        """Adding the same component twice does not duplicate."""
        ts = TrainingSet(tmp_path / "ts.json")
        ts.add_component((1, 2), _comp())
        ts.add_component((1, 2), _comp())
        assert len(ts.get_entry((1, 2))["components"]) == 1  # type: ignore[index]

    def test_remove_component(self, tmp_path: Path) -> None:
        """Remove a previously added component."""
        ts = TrainingSet(tmp_path / "ts.json")
        c = _comp()
        ts.add_component((1, 2), c)
        ts.remove_component((1, 2), c)
        assert not ts.has_component((1, 2), c)

    def test_remove_component_missing_pixel(self, tmp_path: Path) -> None:
        """Remove on a missing pixel is a no-op."""
        ts = TrainingSet(tmp_path / "ts.json")
        ts.remove_component((99, 99), _comp())  # should not raise

    def test_clear_pixel_missing(self, tmp_path: Path) -> None:
        """Clear on a missing pixel is a no-op."""
        ts = TrainingSet(tmp_path / "ts.json")
        ts.clear_pixel((99, 99))  # should not raise

    def test_has_component_missing_pixel(self, tmp_path: Path) -> None:
        """has_component returns False for missing pixel."""
        ts = TrainingSet(tmp_path / "ts.json")
        assert not ts.has_component((99, 99), _comp())

    def test_has_component_different_source(self, tmp_path: Path) -> None:
        """Components with different source don't match."""
        ts = TrainingSet(tmp_path / "ts.json")
        ts.add_component((1, 2), _comp(source="phspectra"))
        assert not ts.has_component((1, 2), _comp(source="gausspyplus"))

    def test_find_component_no_match(self, tmp_path: Path) -> None:
        """_find_component returns False when no component matches."""
        ts = TrainingSet(tmp_path / "ts.json")
        ts.add_component((1, 2), _comp(amp=1.0))
        assert not ts.has_component((1, 2), _comp(amp=999.0))
