"""Model quality metrics and component validation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from phspectra._types import GaussianComponent

_FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))  # 2.3548
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def aicc(residuals: NDArray[np.floating], n_params: int) -> float:
    r"""Corrected Akaike Information Criterion (Riener et al. 2019, Sect 3.2.1.5).

    .. math::

        AIC  = N \ln(RSS / N) + 2k
        AICc = AIC + (2k^2 + 2k) / (N - k - 1)

    Parameters
    ----------
    residuals:
        Residual array (data - model).
    n_params:
        Number of free parameters *k* in the model.

    Returns
    -------
    float
        AICc value, or ``inf`` if the model is over-parameterized.
    """
    residuals = np.asarray(residuals, dtype=np.float64).ravel()
    n = len(residuals)
    k = n_params

    if n <= k + 1:
        return float("inf")

    rss = float(np.sum(residuals**2))
    if rss <= 0:
        rss = 1e-300  # avoid log(0)

    aic = n * math.log(rss / n) + 2.0 * k
    correction = (2.0 * k**2 + 2.0 * k) / (n - k - 1)
    return aic + correction


def validate_components(
    components: list[GaussianComponent],
    rms: float,
    n_channels: int,
    *,
    snr_min: float = 1.5,
    sig_min: float = 5.0,
    fwhm_min_channels: float = 1.0,
) -> list[GaussianComponent]:
    """Discard unphysical or insignificant components (Riener Sect 3.2.1).

    A component is rejected if **any** of the following hold:

    * FWHM < *fwhm_min_channels* (sub-channel width)
    * Mean outside ``[0, n_channels)``
    * Amplitude < ``snr_min * rms`` (below noise)
    * Significance < *sig_min*:
      ``W_i / (sqrt(2 * FWHM_i) * rms)``
      where ``W_i = amplitude * stddev * sqrt(2 pi)``
    """
    valid: list[GaussianComponent] = []
    for c in components:
        fwhm = _FWHM_FACTOR * c.stddev

        # Sub-channel width
        if fwhm < fwhm_min_channels:
            continue

        # Off-spectrum
        if c.mean < 0 or c.mean >= n_channels:
            continue

        # Below noise floor
        if c.amplitude < snr_min * rms:
            continue

        # Significance test
        if rms > 0:
            w_i = c.amplitude * c.stddev * _SQRT_2PI
            significance = w_i / (math.sqrt(2.0 * fwhm) * rms)
            if significance < sig_min:
                continue

        valid.append(c)
    return valid


def find_blended_pairs(
    components: list[GaussianComponent],
    f_sep: float = 1.2,
) -> list[tuple[int, int]]:
    """Identify blended component pairs (Riener Sect 3.2.2.3).

    Two components are considered blended when
    ``|mu_i - mu_j| < f_sep * min(FWHM_i, FWHM_j)``.

    Returns a list of ``(i, j)`` index pairs with ``i < j``.
    """
    pairs: list[tuple[int, int]] = []
    n = len(components)
    for i in range(n):
        fwhm_i = _FWHM_FACTOR * components[i].stddev
        for j in range(i + 1, n):
            fwhm_j = _FWHM_FACTOR * components[j].stddev
            sep = abs(components[i].mean - components[j].mean)
            if sep < f_sep * min(fwhm_i, fwhm_j):
                pairs.append((i, j))
    return pairs
