"""Shared data types for phspectra."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GaussianComponent:
    """A single Gaussian component from the decomposition.

    Attributes
    ----------
    amplitude : float
        Peak height of the Gaussian.
    mean : float
        Centre position (in array-index or channel units).
    stddev : float
        Standard deviation (width).
    """

    amplitude: float
    mean: float
    stddev: float
