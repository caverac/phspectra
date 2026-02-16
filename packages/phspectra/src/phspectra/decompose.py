"""Gaussian decomposition driven by persistent-homology peak detection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from phspectra._types import GaussianComponent
from phspectra.noise import estimate_rms, estimate_rms_simple
from phspectra.persistence import PersistentPeak, find_peaks_by_persistence
from phspectra.quality import aicc, find_blended_pairs, validate_components

try:
    from phspectra._gaussfit import bounded_lm_fit as _c_bounded_lm_fit

    _HAS_C_EXT: bool = True
except ImportError:  # pragma: no cover
    _HAS_C_EXT = False

__all__ = [
    "GaussianComponent",
    "estimate_rms",
    "estimate_rms_simple",
    "fit_gaussians",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _multi_gaussian(x: NDArray[np.floating], *params: float) -> NDArray[np.floating]:
    """Sum of Gaussians.  ``params`` is a flat array [a0, mu0, sig0, a1, ...]."""
    y = np.zeros_like(x, dtype=np.float64)
    for i in range(0, len(params), 3):
        a, mu, sig = params[i], params[i + 1], params[i + 2]
        y += a * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    return y


def _components_to_params(
    components: list[GaussianComponent],
    n_channels: int,
) -> tuple[list[float], list[float], list[float]]:
    """Flatten components to (p0, lower, upper) for curve_fit."""
    p0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for c in components:
        p0.extend([c.amplitude, c.mean, c.stddev])
        lower.extend([0.0, 0.0, 0.3])
        upper.extend([np.inf, float(n_channels), float(n_channels) / 2])
    return p0, lower, upper


def _fit_components(
    x: NDArray[np.floating],
    signal: NDArray[np.floating],
    components: list[GaussianComponent],
    n_channels: int,
    *,
    maxfev: int = 10_000,
) -> tuple[list[GaussianComponent], NDArray[np.floating]]:
    """Run curve_fit and return (fitted components, residual)."""
    if not components:
        return [], signal.copy()

    p0, lower, upper = _components_to_params(components, n_channels)

    if _HAS_C_EXT:
        try:
            popt = _c_bounded_lm_fit(
                np.ascontiguousarray(x, dtype=np.float64),
                np.ascontiguousarray(signal, dtype=np.float64),
                np.array(p0, dtype=np.float64),
                np.array(lower, dtype=np.float64),
                np.array(upper, dtype=np.float64),
                maxfev,
            )
        except RuntimeError:
            popt = np.array(p0)
        except ValueError:
            # Too many parameters for C solver; fall back to scipy
            try:
                popt, _ = curve_fit(
                    _multi_gaussian,
                    x,
                    signal,
                    p0=p0,
                    bounds=(lower, upper),
                    maxfev=maxfev,
                )
            except (RuntimeError, np.linalg.LinAlgError):
                popt = np.array(p0)
    else:  # pragma: no cover
        try:
            popt, _ = curve_fit(
                _multi_gaussian,
                x,
                signal,
                p0=p0,
                bounds=(lower, upper),
                maxfev=maxfev,
            )
        except (RuntimeError, np.linalg.LinAlgError):
            popt = np.array(p0)

    fitted = [
        GaussianComponent(
            amplitude=float(popt[i]),
            mean=float(popt[i + 1]),
            stddev=float(abs(popt[i + 2])),
        )
        for i in range(0, len(popt), 3)
    ]
    model = _multi_gaussian(x, *popt)
    residual = signal - model
    return fitted, residual


def _estimate_stddev(pk: PersistentPeak) -> float:
    """Estimate initial Gaussian sigma from the peak-to-saddle distance.

    For a Gaussian that equals ``birth`` at the peak and ``death`` at
    the saddle, the exact relationship is::

        sigma = d / sqrt(2 * ln(birth / death))

    where ``d = |peak_index - saddle_index|``.  Falls back to 1.0 when
    the saddle is unknown (global maximum) or the estimate is degenerate.
    """
    if pk.saddle_index < 0 or pk.death <= 0.0:
        return 1.0
    d = abs(pk.index - pk.saddle_index)
    if d == 0:
        return 1.0
    ratio = pk.birth / pk.death
    if ratio <= 1.0:
        return 1.0
    return float(d / np.sqrt(2.0 * np.log(ratio)))


def _peaks_to_components(
    peaks: list[PersistentPeak],
    signal: NDArray[np.floating],
) -> list[GaussianComponent]:
    """Convert PersistentPeak list to initial-guess GaussianComponents."""
    return [
        GaussianComponent(
            amplitude=max(float(signal[pk.index]), 1e-10),
            mean=float(pk.index),
            stddev=_estimate_stddev(pk),
        )
        for pk in peaks
    ]


def _find_residual_peaks(
    residual: NDArray[np.floating],
    rms: float,
    snr_min: float,
    mf_snr_min: float,
) -> list[GaussianComponent]:
    """Find peaks in the residual above S/N and matched-filter SNR thresholds."""
    threshold = snr_min * rms
    peaks = find_peaks_by_persistence(residual, min_persistence=threshold)
    if not peaks:
        return []

    n_channels = len(residual)
    candidates = _peaks_to_components(peaks, residual)
    return validate_components(
        candidates,
        rms,
        n_channels,
        snr_min=snr_min,
        mf_snr_min=mf_snr_min,
    )


def _has_negative_dip(
    residual: NDArray[np.floating],
    rms: float,
    neg_thresh: float,
) -> int | None:
    """Find the channel of the deepest negative dip exceeding threshold.

    Returns the channel index, or None if no significant dip exists.
    """
    threshold = -neg_thresh * rms
    below = residual < threshold
    if not np.any(below):
        return None
    return int(np.argmin(residual))


def _split_component_at(
    components: list[GaussianComponent],
    dip_channel: int,
) -> list[GaussianComponent] | None:
    """Split the broadest component overlapping *dip_channel* into two.

    Returns a new component list with the split applied, or None if no
    component overlaps the dip.
    """
    # Find the broadest component whose mean is closest to the dip
    best_idx = None
    best_stddev = 0.0
    for i, c in enumerate(components):
        if abs(c.mean - dip_channel) < 3.0 * c.stddev and c.stddev > best_stddev:
            best_idx = i
            best_stddev = c.stddev

    if best_idx is None:
        return None

    c = components[best_idx]
    offset = c.stddev * 0.5
    left = GaussianComponent(
        amplitude=c.amplitude * 0.7,
        mean=c.mean - offset,
        stddev=c.stddev * 0.5,
    )
    right = GaussianComponent(
        amplitude=c.amplitude * 0.7,
        mean=c.mean + offset,
        stddev=c.stddev * 0.5,
    )
    result = list(components)
    result[best_idx] = left
    result.append(right)
    return result


def _merge_blended(
    components: list[GaussianComponent],
    i: int,
    j: int,
) -> list[GaussianComponent]:
    """Merge two blended components into one."""
    ci, cj = components[i], components[j]
    # Flux-weighted merge
    w_i = ci.amplitude * ci.stddev
    w_j = cj.amplitude * cj.stddev
    w_total = w_i + w_j
    if w_total == 0:
        w_total = 1.0

    merged = GaussianComponent(
        amplitude=ci.amplitude + cj.amplitude,
        mean=(ci.mean * w_i + cj.mean * w_j) / w_total,
        stddev=(ci.stddev * w_i + cj.stddev * w_j) / w_total,
    )
    result = [c for k, c in enumerate(components) if k not in (i, j)]
    result.append(merged)
    return result


def _try_refit(
    x: NDArray[np.floating],
    signal: NDArray[np.floating],
    candidate: list[GaussianComponent],
    n_channels: int,
    current_aicc: float,
) -> tuple[list[GaussianComponent], NDArray[np.floating], float, bool]:
    """Refit *candidate* and accept if AICc improves."""
    new_comps, new_resid = _fit_components(x, signal, candidate, n_channels, maxfev=5_000)
    new_aicc = aicc(new_resid, 3 * len(new_comps))
    if new_aicc < current_aicc:
        return new_comps, new_resid, new_aicc, True
    return [], new_resid, current_aicc, False


def _refine_iteration(  # pylint: disable=too-many-arguments
    x: NDArray[np.floating],
    signal: NDArray[np.floating],
    components: list[GaussianComponent],
    residual: NDArray[np.floating],
    n_channels: int,
    current_aicc: float,
    *,
    rms: float,
    snr_min: float,
    mf_snr_min: float,
    neg_thresh: float,
    f_sep: float,
    max_components: int | None,
) -> tuple[list[GaussianComponent], NDArray[np.floating], float, bool]:
    """Run one refinement iteration: residual peaks, dip split, blended merge."""
    changed = False

    # Search residual for new peaks
    new_peaks = _find_residual_peaks(residual, rms, snr_min, mf_snr_min)
    if new_peaks:
        candidate = list(components) + new_peaks
        if max_components is not None:
            candidate = candidate[:max_components]
        new_comps, new_resid, current_aicc, accepted = _try_refit(x, signal, candidate, n_channels, current_aicc)
        if accepted:
            components, residual = new_comps, new_resid
            changed = True

    # Check for negative residual dips
    dip = _has_negative_dip(residual, rms, neg_thresh)
    if dip is not None:
        split_comps = _split_component_at(components, dip)
        if split_comps is not None:
            new_comps, new_resid, current_aicc, accepted = _try_refit(x, signal, split_comps, n_channels, current_aicc)
            if accepted:
                components, residual = new_comps, new_resid
                changed = True

    # Check for blended pairs
    blended = find_blended_pairs(components, f_sep)
    if blended:
        i, j = blended[0]
        merged_comps = _merge_blended(components, i, j)
        new_comps, new_resid, current_aicc, accepted = _try_refit(x, signal, merged_comps, n_channels, current_aicc)
        if accepted:
            components, residual = new_comps, new_resid
            changed = True

    return components, residual, current_aicc, changed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_gaussians(
    signal: NDArray[np.floating],
    *,
    peaks: list[PersistentPeak] | None = None,
    beta: float = 3.8,
    min_persistence: float | None = None,
    max_components: int | None = None,
    refine: bool = True,
    max_refine_iter: int = 3,
    snr_min: float = 1.5,
    mf_snr_min: float = 5.0,
    f_sep: float = 1.2,
    neg_thresh: float = 5.0,
) -> list[GaussianComponent]:
    r"""Fit a sum of Gaussians to *signal* using persistence-detected peaks.

    The persistence threshold is set automatically as ``beta * rms``, where
    *rms* is estimated from the signal via signal-masked MAD estimation.
    This makes **beta** the single free parameter of the model.

    When *refine* is ``True`` (default), an iterative refinement loop applies:

    * Component quality validation (SNR, matched-filter SNR, FWHM)
    * Residual peak search (adds missed components)
    * Negative-dip splitting (decomposes blended peaks)
    * Blended-pair merging (removes redundant components)

    Each refinement step is accepted only if it improves AICc.

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
    refine:
        Enable iterative refinement.  Set to ``False`` for legacy behaviour.
    max_refine_iter:
        Maximum number of refinement iterations.
    snr_min:
        Minimum amplitude signal-to-noise ratio for a component to be valid.
    mf_snr_min:
        Minimum matched-filter SNR.
        :math:`\mathrm{SNR}_\mathrm{mf} = (A / \sigma_\mathrm{rms})\,\sqrt{\sigma}\;\pi^{1/4}`.
        Narrow peaks must have proportionally higher amplitude to pass.
    f_sep:
        Blended-pair separation factor (in units of FWHM).
    neg_thresh:
        Negative residual threshold in units of RMS for dip detection.

    Returns
    -------
    list[GaussianComponent]
        Fitted Gaussians sorted by mean position.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n_channels = len(signal)
    x = np.arange(n_channels, dtype=np.float64)

    # --- Step 1: RMS estimation --------------------------------------------
    if refine:
        rms = estimate_rms(signal)
    else:
        rms = estimate_rms_simple(signal)

    # --- Step 2: Peak detection --------------------------------------------
    if peaks is None:
        if min_persistence is None:
            min_persistence = beta * rms
        peaks = find_peaks_by_persistence(signal, min_persistence=min_persistence)

    if max_components is not None:
        peaks = peaks[:max_components]

    if not peaks:
        return []

    # --- Step 3: Initial fit -----------------------------------------------
    initial_guesses = _peaks_to_components(peaks, signal)
    components, residual = _fit_components(x, signal, initial_guesses, n_channels)

    if not refine:
        components.sort(key=lambda c: c.mean)
        return components

    # --- Step 4: Validation ------------------------------------------------
    validated = validate_components(components, rms, n_channels, snr_min=snr_min, mf_snr_min=mf_snr_min)
    if len(validated) != len(components):
        if not validated:
            return []
        components, residual = _fit_components(x, signal, validated, n_channels, maxfev=5_000)

    # --- Step 5: AICc baseline ---------------------------------------------
    current_aicc = aicc(residual, 3 * len(components))

    # --- Early exit: nothing to refine? ------------------------------------
    residual_peaks = _find_residual_peaks(residual, rms, snr_min, mf_snr_min)
    blended = find_blended_pairs(components, f_sep)
    if not residual_peaks and not blended:
        components.sort(key=lambda c: c.mean)
        return components

    # --- Step 6: Iterative refinement loop ---------------------------------
    for _ in range(max_refine_iter):
        components, residual, current_aicc, changed = _refine_iteration(
            x,
            signal,
            components,
            residual,
            n_channels,
            current_aicc,
            rms=rms,
            snr_min=snr_min,
            mf_snr_min=mf_snr_min,
            neg_thresh=neg_thresh,
            f_sep=f_sep,
            max_components=max_components,
        )
        if not changed:
            break

    # --- Step 7: Return sorted by mean -------------------------------------
    components.sort(key=lambda c: c.mean)
    return components
