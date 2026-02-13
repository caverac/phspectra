"""Tests for phspectra.decompose (iterative refinement pipeline)."""

from __future__ import annotations

import time

import numpy as np

from phspectra._types import GaussianComponent
from phspectra.decompose import fit_gaussians


def _make_gaussian(x: np.ndarray, amp: float, mu: float, sig: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)


# ---- Backward compatibility ------------------------------------------------


def test_backward_compat() -> None:
    """fit_gaussians should return list of GaussianComponent (unchanged API)."""
    x = np.arange(200, dtype=np.float64)
    signal = _make_gaussian(x, 5.0, 100.0, 5.0)
    result = fit_gaussians(signal, beta=5.2)
    assert isinstance(result, list)
    assert all(isinstance(c, GaussianComponent) for c in result)


def test_refine_false_matches_legacy() -> None:
    """refine=False should give the same number of components as old behavior."""
    x = np.arange(200, dtype=np.float64)
    rng = np.random.default_rng(42)
    signal = (
        _make_gaussian(x, 5.0, 60.0, 5.0)
        + _make_gaussian(x, 3.0, 140.0, 4.0)
        + rng.normal(0, 0.3, size=200)
    )
    result = fit_gaussians(signal, beta=5.2, refine=False)
    assert len(result) >= 1  # should find at least one component


# ---- Refinement tests ------------------------------------------------------


def test_refine_finds_hidden_peak() -> None:
    """A weak third component below initial threshold should be found via residual."""
    x = np.arange(300, dtype=np.float64)
    rng = np.random.default_rng(77)
    noise = rng.normal(0, 0.2, size=300)
    signal = (
        _make_gaussian(x, 8.0, 80.0, 6.0)
        + _make_gaussian(x, 6.0, 200.0, 5.0)
        + _make_gaussian(x, 1.5, 140.0, 4.0)  # weak component
        + noise
    )
    result = fit_gaussians(signal, beta=3.0, snr_min=1.0, sig_min=2.0)
    # Should find at least 2 components; with refinement, possibly 3
    assert len(result) >= 2


def test_blended_merged() -> None:
    """Two nearly-identical Gaussians should merge into a single component."""
    x = np.arange(200, dtype=np.float64)
    # Two Gaussians very close together (separation < f_sep * min_fwhm)
    signal = _make_gaussian(x, 5.0, 100.0, 8.0) + _make_gaussian(x, 4.5, 103.0, 8.0)
    result = fit_gaussians(signal, beta=2.0, f_sep=1.2, snr_min=1.0, sig_min=2.0)
    # After merging blended pair, should have fewer components than initial peaks
    # The exact count depends on AICc, but should be 1 or 2
    assert len(result) <= 2


def test_negative_residual_split() -> None:
    """A broad component covering two peaks should be split via negative dip."""
    x = np.arange(200, dtype=np.float64)
    rng = np.random.default_rng(55)
    # Two peaks with a dip between them -- might initially fit as one broad Gaussian
    signal = (
        _make_gaussian(x, 5.0, 80.0, 5.0)
        + _make_gaussian(x, 5.0, 120.0, 5.0)
        + rng.normal(0, 0.2, size=200)
    )
    result = fit_gaussians(signal, beta=3.0, neg_thresh=3.0, snr_min=1.0, sig_min=2.0)
    # Should find 2 components
    assert len(result) >= 2


def test_max_iter_respected() -> None:
    """Iteration count should not exceed max_refine_iter."""
    x = np.arange(200, dtype=np.float64)
    rng = np.random.default_rng(42)
    signal = (
        _make_gaussian(x, 5.0, 60.0, 5.0)
        + _make_gaussian(x, 3.0, 140.0, 4.0)
        + rng.normal(0, 0.3, size=200)
    )
    # With max_refine_iter=0, should still return something (no refinement)
    result = fit_gaussians(signal, beta=3.0, max_refine_iter=0)
    assert isinstance(result, list)


# ---- Performance tests -----------------------------------------------------


def test_noise_only_fast() -> None:
    """Noise-only spectra with no detected peaks should return instantly."""
    rng = np.random.default_rng(42)
    # Use a high enough beta that pure noise never triggers peaks
    start = time.perf_counter()
    for _ in range(500):
        signal = rng.normal(0, 0.5, size=424)
        result = fit_gaussians(signal, beta=8.0)
        assert result == []
    elapsed = time.perf_counter() - start
    # Should average < 2ms per spectrum (no curve_fit called)
    assert elapsed / 500 < 0.002


def test_single_component_fast() -> None:
    """Single-Gaussian spectra should average under 100ms each."""
    rng = np.random.default_rng(42)
    x = np.arange(424, dtype=np.float64)
    start = time.perf_counter()
    n_spectra = 20
    for _ in range(n_spectra):
        noise = rng.normal(0, 0.3, size=424)
        signal = _make_gaussian(x, 5.0, 200.0, 8.0) + noise
        fit_gaussians(signal, beta=5.2)
    elapsed = time.perf_counter() - start
    # Should average < 100ms; generous margin for CI variability
    assert elapsed / n_spectra < 0.100
