"""Gaussian model building and residual helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
from benchmarks._types import Component


def gaussian(x: npt.NDArray[np.float64], amp: float, mean: float, stddev: float) -> npt.NDArray[np.float64]:
    """Evaluate a single Gaussian at *x*.

    Parameters
    ----------
    x : npt.NDArray[np.float64]
        Channel array.
    amp : float
        Peak amplitude.
    mean : float
        Centre position.
    stddev : float
        Standard deviation.

    Returns
    -------
    npt.NDArray[np.float64]
        Evaluated Gaussian.
    """
    return amp * np.exp(-0.5 * ((x - mean) / stddev) ** 2)


def gaussian_model(x: npt.NDArray[np.float64], comps: Sequence[Component]) -> npt.NDArray[np.float64]:
    """Build a sum-of-Gaussians model.

    Parameters
    ----------
    x : npt.NDArray[np.float64]
        Channel array.
    comps : Sequence[Component]
        Gaussian components.

    Returns
    -------
    npt.NDArray[np.float64]
        Summed model.
    """
    model = np.zeros_like(x, dtype=np.float64)
    for c in comps:
        model += c.amplitude * np.exp(-0.5 * ((x - c.mean) / max(c.stddev, 1e-10)) ** 2)
    return model


def residual_rms(signal: npt.NDArray[np.float64], comps: Sequence[Component]) -> float:
    """Root-mean-square residual of the model against *signal*.

    Parameters
    ----------
    signal : npt.NDArray[np.float64]
        Observed signal.
    comps : Sequence[Component]
        Fitted components.

    Returns
    -------
    float
        RMS residual.
    """
    x = np.arange(len(signal), dtype=np.float64)
    model = gaussian_model(x, comps) if comps else np.zeros_like(signal)
    return float(np.sqrt(np.mean((signal - model) ** 2)))
