"""Shared dataclasses for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Component:
    """Gaussian component in channel space."""

    amplitude: float
    mean: float
    stddev: float


@dataclass
class ComparisonResult:
    """One spectrum with both decompositions + timing."""

    pixel: tuple[int, int]
    signal: np.ndarray
    gp_comps: list[Component]
    ph_comps: list[Component]
    ph_rms: float
    gp_rms: float
    ph_time: float
    gp_time: float


@dataclass
class SyntheticSpectrum:
    """A synthetic spectrum with known ground-truth components."""

    category: str
    index: int
    signal: np.ndarray
    components: list[Component]
