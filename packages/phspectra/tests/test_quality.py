"""Tests for phspectra.quality."""

from __future__ import annotations

import math

import numpy as np

from phspectra._types import GaussianComponent
from phspectra.quality import aicc, find_blended_pairs, validate_components

FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))


# ---- AICc tests -----------------------------------------------------------


def test_aicc_penalizes_extra_params() -> None:
    """A 2-component fit should be preferred over 3-component when RSS gain is tiny."""
    rng = np.random.default_rng(99)
    n = 200
    residual_2comp = rng.normal(0, 1.0, size=n)
    # 3-comp residual barely better (0.1% improvement -- not enough to justify 3 more params)
    residual_3comp = residual_2comp * 0.999

    aicc_2 = aicc(residual_2comp, n_params=6)
    aicc_3 = aicc(residual_3comp, n_params=9)

    assert aicc_2 < aicc_3  # fewer params wins


def test_aicc_overparameterized() -> None:
    """n <= k+1 should return inf."""
    residuals = np.array([1.0, 2.0, 3.0])
    result = aicc(residuals, n_params=3)
    assert result == float("inf")


# ---- validate_components tests --------------------------------------------


def _good_component() -> GaussianComponent:
    """A clearly valid component."""
    return GaussianComponent(amplitude=5.0, mean=50.0, stddev=3.0)


def test_validate_rejects_narrow_fwhm() -> None:
    """Component with FWHM < 1 channel should be rejected."""
    # stddev such that FWHM < 1.0
    narrow = GaussianComponent(amplitude=5.0, mean=50.0, stddev=0.3)
    assert narrow.stddev * FWHM_FACTOR < 1.0  # confirm it's narrow
    result = validate_components([narrow], rms=0.5, n_channels=100)
    assert result == []


def test_validate_rejects_low_snr() -> None:
    """Component with amplitude < snr_min * rms should be rejected."""
    weak = GaussianComponent(amplitude=0.1, mean=50.0, stddev=3.0)
    result = validate_components([weak], rms=1.0, n_channels=100, snr_min=1.5)
    assert result == []


def test_validate_rejects_low_significance() -> None:
    """Narrow weak component below significance threshold should be rejected."""
    # Component with modest amplitude and narrow width -> low significance
    comp = GaussianComponent(amplitude=2.0, mean=50.0, stddev=0.5)
    result = validate_components([comp], rms=0.5, n_channels=100, snr_min=1.0, sig_min=5.0)
    assert result == []


def test_validate_rejects_oob_mean() -> None:
    """Component with mean outside [0, n_channels) should be rejected."""
    oob = GaussianComponent(amplitude=5.0, mean=-1.0, stddev=3.0)
    result = validate_components([oob], rms=0.5, n_channels=100)
    assert result == []

    oob2 = GaussianComponent(amplitude=5.0, mean=100.0, stddev=3.0)
    result2 = validate_components([oob2], rms=0.5, n_channels=100)
    assert result2 == []


def test_validate_passes_good_component() -> None:
    """A clearly valid component should survive all checks."""
    good = _good_component()
    result = validate_components([good], rms=0.5, n_channels=100)
    assert len(result) == 1
    assert result[0] == good


# ---- find_blended_pairs tests ---------------------------------------------


def test_blended_detected() -> None:
    """Two overlapping Gaussians should be identified as blended."""
    c1 = GaussianComponent(amplitude=5.0, mean=50.0, stddev=5.0)
    c2 = GaussianComponent(amplitude=4.0, mean=53.0, stddev=5.0)
    pairs = find_blended_pairs([c1, c2], f_sep=1.2)
    assert len(pairs) == 1
    assert pairs[0] == (0, 1)


def test_separated_not_blended() -> None:
    """Two well-separated Gaussians should not be flagged."""
    c1 = GaussianComponent(amplitude=5.0, mean=20.0, stddev=3.0)
    c2 = GaussianComponent(amplitude=4.0, mean=80.0, stddev=3.0)
    pairs = find_blended_pairs([c1, c2], f_sep=1.2)
    assert pairs == []
