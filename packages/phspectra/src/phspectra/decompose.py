"""Gaussian decomposition driven by persistent-homology peak detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from phspectra.persistence import PersistentPeak, find_peaks_by_persistence

#: Default persistence threshold in units of noise sigma.
DEFAULT_BETA: float = 5.0


def estimate_rms(signal: NDArray[np.floating]) -> float:
    """Estimate the noise RMS of a 1-D signal using the median absolute deviation.

    The MAD is robust to outliers (real signal peaks) and is converted to a
    Gaussian-equivalent sigma via ``MAD / 0.6745``.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if len(signal) == 0:
        return 0.0
    med = float(np.median(signal))
    mad = float(np.median(np.abs(signal - med)))
    return mad / 0.6745


@dataclass(frozen=True, slots=True)
class GaussianComponent:
    """A single Gaussian component from the decomposition.

    Attributes:
        amplitude: Peak height of the Gaussian.
        mean: Centre position (in array-index or channel units).
        stddev: Standard deviation (width).
    """

    amplitude: float
    mean: float
    stddev: float


def _multi_gaussian(x: NDArray[np.floating], *params: float) -> NDArray[np.floating]:
    """Sum of Gaussians.  ``params`` is a flat array [a0, mu0, sig0, a1, ...]."""
    y = np.zeros_like(x, dtype=np.float64)
    for i in range(0, len(params), 3):
        a, mu, sig = params[i], params[i + 1], params[i + 2]
        y += a * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    return y


def fit_gaussians(
    signal: NDArray[np.floating],
    *,
    peaks: list[PersistentPeak] | None = None,
    beta: float = DEFAULT_BETA,
    min_persistence: float | None = None,
    max_components: int | None = None,
) -> list[GaussianComponent]:
    """Fit a sum of Gaussians to *signal* using persistence-detected peaks.

    The persistence threshold is set automatically as ``beta * rms``, where
    *rms* is estimated from the signal via the median absolute deviation.
    This makes **beta** the single free parameter of the model.

    Parameters
    ----------
    signal:
        1-D spectrum (flux values).
    peaks:
        Pre-computed peaks.  If *None*, :func:`find_peaks_by_persistence` is
        called with the computed persistence threshold.
    beta:
        Persistence threshold in units of noise sigma.  ``min_persistence`` is
        computed as ``beta * estimate_rms(signal)``.  Ignored when
        *min_persistence* is given explicitly.
    min_persistence:
        Absolute persistence threshold.  When set, overrides *beta*.
    max_components:
        Cap on the number of Gaussians to fit.

    Returns
    -------
    list[GaussianComponent]
        Fitted Gaussians sorted by mean position.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    x = np.arange(len(signal), dtype=np.float64)

    if peaks is None:
        if min_persistence is None:
            min_persistence = beta * estimate_rms(signal)
        peaks = find_peaks_by_persistence(signal, min_persistence=min_persistence)

    if max_components is not None:
        peaks = peaks[:max_components]

    if not peaks:
        return []

    # Initial guesses: amplitude = signal value at peak, sigma = 1 channel
    p0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for pk in peaks:
        amp_guess = max(signal[pk.index], 1e-10)
        p0.extend([amp_guess, float(pk.index), 1.0])
        lower.extend([0.0, 0.0, 0.3])
        upper.extend([np.inf, float(len(signal)), float(len(signal)) / 2])

    try:
        popt, _ = curve_fit(
            _multi_gaussian,
            x,
            signal,
            p0=p0,
            bounds=(lower, upper),
            maxfev=10_000,
        )
    except RuntimeError:
        # curve_fit failed to converge -- return initial guesses
        popt = np.array(p0)

    components = [
        GaussianComponent(
            amplitude=float(popt[i]),
            mean=float(popt[i + 1]),
            stddev=float(abs(popt[i + 2])),
        )
        for i in range(0, len(popt), 3)
    ]
    components.sort(key=lambda c: c.mean)
    return components
