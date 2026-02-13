"""Shared fixtures for benchmarks tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
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
