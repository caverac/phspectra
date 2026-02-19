"""Noise estimation for 1-D spectral signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_MAD_TO_SIGMA = 1.4826  # 1 / 0.6745


def estimate_rms_simple(signal: NDArray[np.floating]) -> float:
    """Estimate noise RMS via the median absolute deviation of the full signal.

    This is the original (v0) estimator, used internally as a fallback by
    :func:`estimate_rms` when fewer than 5 channels survive masking.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if len(signal) == 0:
        return 0.0
    if np.any(np.isnan(signal)):
        raise ValueError("signal contains NaN values; caller must handle missing channels before fitting")
    med = float(np.median(signal))
    mad = float(np.median(np.abs(signal - med)))
    return mad / 0.6745


def estimate_rms(
    signal: NDArray[np.floating],
    *,
    mask_pad: int = 2,
    mad_clip: float = 5.0,
) -> float:
    """Signal-masked RMS estimation (Riener et al. 2019, Sect 3.1.1).

    Steps:

    1. Find runs of consecutive positive channels, mask runs longer than 2
       channels (plus *mask_pad* padding on each side).
    2. Compute MAD from **negative** unmasked channels only.
    3. Mask channels exceeding ``+/- mad_clip * MAD_sigma``.
    4. Return final RMS from surviving unmasked channels.
    5. Fall back to :func:`estimate_rms_simple` if fewer than 5 channels survive.

    Parameters
    ----------
    signal:
        1-D spectrum (flux values).
    mask_pad:
        Number of channels to pad on each side of masked positive runs.
    mad_clip:
        Number of sigma for the clipping step.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = len(signal)
    if n == 0:
        return 0.0
    if np.any(np.isnan(signal)):
        raise ValueError("signal contains NaN values; caller must handle missing channels before fitting")

    # --- Step 1: mask runs of consecutive positive channels > 2 wide --------
    positive = signal > 0
    mask = np.zeros(n, dtype=bool)

    # Find run starts and lengths using diff
    if np.any(positive):
        padded = np.concatenate(([False], positive, [False]))
        diffs = np.diff(padded.astype(np.int8))
        run_starts = np.where(diffs == 1)[0]
        run_ends = np.where(diffs == -1)[0]
        run_lengths = run_ends - run_starts

        for start, length in zip(run_starts, run_lengths):
            if length > 2:
                lo = max(0, start - mask_pad)
                hi = min(n, start + length + mask_pad)
                mask[lo:hi] = True

    # --- Step 2: MAD from negative unmasked channels only -------------------
    neg_unmasked = signal[~mask & ~positive]
    if len(neg_unmasked) < 5:
        return estimate_rms_simple(signal)

    med_neg = float(np.median(neg_unmasked))
    mad_sigma = float(np.median(np.abs(neg_unmasked - med_neg))) * _MAD_TO_SIGMA

    if mad_sigma <= 0:
        return estimate_rms_simple(signal)

    # --- Step 3: clip channels exceeding +/- mad_clip * sigma ---------------
    clip_thresh = mad_clip * mad_sigma
    mask |= np.abs(signal) > clip_thresh

    # --- Step 4: final RMS from surviving channels --------------------------
    surviving = signal[~mask]
    if len(surviving) < 5:
        return estimate_rms_simple(signal)

    return float(np.sqrt(np.mean(surviving**2)))
