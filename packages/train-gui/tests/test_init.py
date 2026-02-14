"""Tests for train_gui.__init__ (matplotlib rcParams)."""

# pylint: disable=import-outside-toplevel,unused-import

from __future__ import annotations


def test_font_config() -> None:
    """Importing train_gui sets serif font family."""
    import matplotlib
    import train_gui  # noqa: F401

    assert matplotlib.rcParams["font.family"] == ["serif"]
    assert "stix" == matplotlib.rcParams["mathtext.fontset"]
