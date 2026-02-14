"""Training set persistence: load/save JSON, add/remove component entries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Tolerance for matching floating-point component parameters.
_ABS_TOL = 1e-4


def _match(a: float, b: float) -> bool:
    return abs(a - b) < _ABS_TOL


class TrainingSet:
    """Mutable container backed by a JSON file.

    Parameters
    ----------
    path : str | Path
        Path to the JSON file (created if missing).
    survey : str
        Survey name stored in each entry.
    """

    def __init__(self, path: str | Path, survey: str = "GRS") -> None:  # noqa: D107
        self._path = Path(path)
        self._survey = survey
        self._entries: list[dict[str, Any]] = []
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Read the JSON file into memory."""
        if self._path.exists():
            with open(self._path, encoding="utf-8") as fh:
                self._entries = json.load(fh)
        else:
            self._entries = []

    def save(self) -> None:
        """Write current state back to disk (pretty-printed, sorted keys)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._entries, fh, indent=2, sort_keys=True)
            fh.write("\n")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_pixel(self, pixel: tuple[int, int]) -> bool:
        """Return *True* if *pixel* has at least one selected component."""
        entry = self.get_entry(pixel)
        return entry is not None and len(entry["components"]) > 0

    def get_entry(self, pixel: tuple[int, int]) -> dict[str, Any] | None:
        """Return the entry dict for *pixel*, or *None*."""
        for entry in self._entries:
            if entry["pixel"] == list(pixel):
                return entry
        return None

    @property
    def curated_pixels(self) -> set[tuple[int, int]]:
        """Set of ``(x, y)`` tuples that have at least one component."""
        return {(e["pixel"][0], e["pixel"][1]) for e in self._entries if e.get("components")}

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_component(
        self,
        pixel: tuple[int, int],
        component: dict[str, Any],
    ) -> None:
        """Add *component* to *pixel*'s entry (upsert)."""
        entry = self.get_entry(pixel)
        if entry is None:
            entry = {
                "survey": self._survey,
                "pixel": list(pixel),
                "components": [],
            }
            self._entries.append(entry)
        # Avoid duplicates.
        if not self._find_component(entry, component):
            entry["components"].append(component)

    def remove_component(
        self,
        pixel: tuple[int, int],
        component: dict[str, Any],
    ) -> None:
        """Remove *component* from *pixel* (matched by amplitude/mean/stddev)."""
        entry = self.get_entry(pixel)
        if entry is None:
            return
        entry["components"] = [
            c
            for c in entry["components"]
            if not (
                c.get("source") == component.get("source")
                and _match(c["amplitude"], component["amplitude"])
                and _match(c["mean"], component["mean"])
                and _match(c["stddev"], component["stddev"])
            )
        ]

    def clear_pixel(self, pixel: tuple[int, int]) -> None:
        """Remove all components for *pixel*."""
        entry = self.get_entry(pixel)
        if entry is not None:
            entry["components"] = []

    def has_component(
        self,
        pixel: tuple[int, int],
        component: dict[str, Any],
    ) -> bool:
        """Return *True* if *component* is already selected for *pixel*."""
        entry = self.get_entry(pixel)
        if entry is None:
            return False
        return self._find_component(entry, component)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _find_component(entry: dict[str, Any], comp: dict[str, Any]) -> bool:
        for c in entry["components"]:
            if (
                c.get("source") == comp.get("source")
                and _match(c["amplitude"], comp["amplitude"])
                and _match(c["mean"], comp["mean"])
                and _match(c["stddev"], comp["stddev"])
            ):
                return True
        return False
