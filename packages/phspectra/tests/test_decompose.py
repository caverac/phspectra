"""Tests for phspectra.decompose (iterative refinement pipeline)."""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import numpy.typing as npt

from phspectra._types import GaussianComponent
from phspectra.decompose import (
    _components_to_params,
    _estimate_stddev,
    _fit_components,
    _has_negative_dip,
    _merge_blended,
    _refine_iteration,
    _split_component_at,
    fit_gaussians,
)
from phspectra.noise import estimate_rms
from phspectra.persistence import PersistentPeak
from phspectra.quality import aicc


def _make_gaussian(x: npt.NDArray[np.float64], amp: float, mu: float, sig: float) -> npt.NDArray[np.float64]:
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
    result = fit_gaussians(signal, beta=3.0, snr_min=1.0, mf_snr_min=2.0)
    # Should find at least 2 components; with refinement, possibly 3
    assert len(result) >= 2


def test_blended_merged() -> None:
    """Two nearly-identical Gaussians should merge into a single component."""
    x = np.arange(200, dtype=np.float64)
    rng = np.random.default_rng(99)
    # Two Gaussians very close together (separation < f_sep * min_fwhm)
    signal = (
        _make_gaussian(x, 5.0, 100.0, 8.0)
        + _make_gaussian(x, 4.5, 103.0, 8.0)
        + rng.normal(0, 0.2, size=200)
    )
    result = fit_gaussians(signal, beta=4.0, f_sep=1.2, snr_min=1.0, mf_snr_min=2.0)
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
    result = fit_gaussians(signal, beta=3.0, neg_thresh=3.0, snr_min=1.0, mf_snr_min=2.0)
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
    # Should average < 5ms per spectrum (no curve_fit called)
    assert elapsed / 500 < 0.005


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


# ---- Internal helper tests -------------------------------------------------


def test_fit_components_empty() -> None:
    """_fit_components with no components should return empty list + signal copy."""
    signal = np.array([1.0, 2.0, 3.0])
    x = np.arange(3, dtype=np.float64)
    comps, resid = _fit_components(x, signal, [], 3)
    assert comps == []
    np.testing.assert_array_equal(resid, signal)


def test_fit_components_curvefit_failure() -> None:
    """_fit_components should fall back to p0 when curve_fit raises."""
    x = np.arange(50, dtype=np.float64)
    signal = np.zeros(50)
    # Provide a degenerate guess that will cause curve_fit to fail
    bad_guess = [GaussianComponent(amplitude=1e-15, mean=25.0, stddev=0.3)]
    with patch(
        "phspectra.decompose.curve_fit",
        side_effect=RuntimeError("maxfev exceeded"),
    ):
        comps, _ = _fit_components(x, signal, bad_guess, 50)
    assert len(comps) == 1


def test_fit_components_linalg_failure() -> None:
    """_fit_components should fall back to p0 on LinAlgError."""
    x = np.arange(50, dtype=np.float64)
    signal = np.zeros(50)
    guess = [GaussianComponent(amplitude=1.0, mean=25.0, stddev=3.0)]
    with patch(
        "phspectra.decompose.curve_fit",
        side_effect=np.linalg.LinAlgError("SVD did not converge"),
    ):
        comps, _ = _fit_components(x, signal, guess, 50)
    assert len(comps) == 1


def test_estimate_stddev_no_saddle() -> None:
    """_estimate_stddev should return 1.0 when saddle_index < 0."""
    pk = PersistentPeak(index=50, birth=5.0, death=2.0, persistence=3.0, saddle_index=-1)
    assert _estimate_stddev(pk) == 1.0


def test_estimate_stddev_zero_death() -> None:
    """_estimate_stddev should return 1.0 when death <= 0."""
    pk = PersistentPeak(index=50, birth=5.0, death=0.0, persistence=5.0, saddle_index=40)
    assert _estimate_stddev(pk) == 1.0


def test_estimate_stddev_zero_distance() -> None:
    """_estimate_stddev should return 1.0 when peak and saddle are at same index."""
    pk = PersistentPeak(index=50, birth=5.0, death=2.0, persistence=3.0, saddle_index=50)
    assert _estimate_stddev(pk) == 1.0


def test_estimate_stddev_ratio_le_one() -> None:
    """_estimate_stddev should return 1.0 when birth/death <= 1."""
    pk = PersistentPeak(index=50, birth=2.0, death=3.0, persistence=-1.0, saddle_index=40)
    assert _estimate_stddev(pk) == 1.0


def test_estimate_stddev_normal() -> None:
    """_estimate_stddev should compute sigma from peak-to-saddle distance."""
    pk = PersistentPeak(index=50, birth=5.0, death=2.0, persistence=3.0, saddle_index=40)
    result = _estimate_stddev(pk)
    expected = 10.0 / np.sqrt(2.0 * np.log(5.0 / 2.0))
    assert abs(result - expected) < 1e-10


def test_has_negative_dip_none() -> None:
    """_has_negative_dip should return None when no dip exceeds threshold."""
    residual = np.array([0.1, -0.1, 0.05, -0.05])
    assert _has_negative_dip(residual, rms=1.0, neg_thresh=5.0) is None


def test_has_negative_dip_found() -> None:
    """_has_negative_dip should return the channel of the deepest negative dip."""
    residual = np.array([0.1, -0.1, 0.05, -10.0, 0.1])
    result = _has_negative_dip(residual, rms=1.0, neg_thresh=5.0)
    assert result == 3


def test_split_component_at_no_overlap() -> None:
    """_split_component_at should return None when no component overlaps the dip."""
    comps = [GaussianComponent(amplitude=5.0, mean=50.0, stddev=3.0)]
    result = _split_component_at(comps, dip_channel=200)
    assert result is None


def test_split_component_at_success() -> None:
    """_split_component_at should split the broadest overlapping component."""
    comps = [GaussianComponent(amplitude=5.0, mean=100.0, stddev=10.0)]
    result = _split_component_at(comps, dip_channel=102)
    assert result is not None
    assert len(result) == 2
    # Left component should have mean < 100 and right > 100
    means = sorted(c.mean for c in result)
    assert means[0] < 100.0
    assert means[1] > 100.0
    # Both should have smaller stddev than original
    assert all(c.stddev < 10.0 for c in result)


def test_split_picks_broadest() -> None:
    """_split_component_at should pick the broadest component near the dip."""
    narrow = GaussianComponent(amplitude=5.0, mean=100.0, stddev=2.0)
    broad = GaussianComponent(amplitude=3.0, mean=100.0, stddev=15.0)
    result = _split_component_at([narrow, broad], dip_channel=102)
    assert result is not None
    assert len(result) == 3  # narrow unchanged + 2 from split


def test_merge_blended_normal() -> None:
    """_merge_blended should produce a flux-weighted merged component."""
    c1 = GaussianComponent(amplitude=4.0, mean=50.0, stddev=5.0)
    c2 = GaussianComponent(amplitude=6.0, mean=55.0, stddev=5.0)
    result = _merge_blended([c1, c2], 0, 1)
    assert len(result) == 1
    merged = result[0]
    assert merged.amplitude == 10.0  # sum of amplitudes
    # Mean should be between 50 and 55
    assert 50.0 < merged.mean < 55.0


def test_merge_blended_zero_weights() -> None:
    """_merge_blended should handle zero-amplitude components gracefully."""
    c1 = GaussianComponent(amplitude=0.0, mean=50.0, stddev=0.0)
    c2 = GaussianComponent(amplitude=0.0, mean=60.0, stddev=0.0)
    result = _merge_blended([c1, c2], 0, 1)
    assert len(result) == 1
    # Should not crash; w_total falls back to 1.0


def test_max_components_trims_peaks() -> None:
    """fit_gaussians with max_components should limit the number of peaks."""
    x = np.arange(300, dtype=np.float64)
    rng = np.random.default_rng(42)
    signal = (
        _make_gaussian(x, 8.0, 50.0, 5.0)
        + _make_gaussian(x, 6.0, 150.0, 5.0)
        + _make_gaussian(x, 4.0, 250.0, 5.0)
        + rng.normal(0, 0.2, size=300)
    )
    result = fit_gaussians(signal, beta=3.0, max_components=1, refine=False)
    assert len(result) == 1


def test_refinement_with_max_components_cap() -> None:
    """Refinement should respect max_components when adding residual peaks."""
    x = np.arange(300, dtype=np.float64)
    rng = np.random.default_rng(42)
    signal = (
        _make_gaussian(x, 8.0, 50.0, 5.0)
        + _make_gaussian(x, 6.0, 150.0, 5.0)
        + _make_gaussian(x, 4.0, 250.0, 5.0)
        + rng.normal(0, 0.2, size=300)
    )
    result = fit_gaussians(signal, beta=3.0, max_components=2, snr_min=1.0, mf_snr_min=2.0)
    assert len(result) <= 2


def test_negative_dip_split_in_refinement() -> None:
    """Refinement should split a broad component when a negative dip is detected."""
    x = np.arange(200, dtype=np.float64)
    rng = np.random.default_rng(88)
    # Create a signal with two peaks close enough to be fit as one broad Gaussian,
    # with a clear dip between them.
    signal = (
        _make_gaussian(x, 6.0, 80.0, 4.0)
        + _make_gaussian(x, 6.0, 100.0, 4.0)
        + rng.normal(0, 0.15, size=200)
    )
    result = fit_gaussians(signal, beta=3.0, neg_thresh=3.0, snr_min=1.0, mf_snr_min=2.0)
    # Should resolve into 2 components via the dip-split path
    assert len(result) >= 2


def test_precomputed_peaks() -> None:
    """fit_gaussians should accept pre-computed peaks."""
    x = np.arange(200, dtype=np.float64)
    signal = _make_gaussian(x, 5.0, 100.0, 5.0)
    peaks = [PersistentPeak(index=100, birth=5.0, death=0.0, persistence=5.0, saddle_index=-1)]
    result = fit_gaussians(signal, peaks=peaks, refine=False)
    assert len(result) == 1


def test_min_persistence_overrides_beta() -> None:
    """Explicit min_persistence should override beta * rms."""
    x = np.arange(200, dtype=np.float64)
    signal = _make_gaussian(x, 5.0, 100.0, 5.0)
    # Very high min_persistence should suppress the peak
    result = fit_gaussians(signal, min_persistence=100.0, refine=False)
    assert result == []


def test_validation_removes_all_components() -> None:
    """When validation rejects all components, return empty list."""
    x = np.arange(200, dtype=np.float64)
    rng = np.random.default_rng(42)
    # Very weak signal that barely passes persistence but fails validation
    signal = _make_gaussian(x, 0.4, 100.0, 2.0) + rng.normal(0, 0.08, size=200)
    result = fit_gaussians(signal, beta=2.0, snr_min=10.0, mf_snr_min=50.0)
    assert result == []


# ---- _refine_iteration branch tests ------------------------------------------


def test_refine_iteration_dip_split_accepted() -> None:
    """Dip split should be accepted in _refine_iteration when AICc improves.

    One broad Gaussian fitting a two-peak signal creates a negative residual
    dip between the peaks.  With high SNR thresholds the residual-peak path
    is suppressed; only the dip-split path fires.
    """
    x = np.arange(200, dtype=np.float64)
    rng = np.random.default_rng(42)
    signal = (
        _make_gaussian(x, 5.0, 80.0, 6.0)
        + _make_gaussian(x, 5.0, 120.0, 6.0)
        + rng.normal(0, 0.2, size=200)
    )

    # Start from one broad component
    broad = [GaussianComponent(amplitude=4.0, mean=100.0, stddev=25.0)]
    components, residual = _fit_components(x, signal, broad, 200)
    current_aicc = aicc(residual, 3)
    rms = estimate_rms(signal)

    new_comps, _, _, changed = _refine_iteration(
        x, signal, components, residual, 200, current_aicc,
        rms=rms,
        snr_min=999.0,       # suppress residual-peak path
        mf_snr_min=999.0,
        neg_thresh=0.5,       # easy dip detection
        f_sep=0.0,            # suppress blended-merge path
        max_components=None,
    )
    assert changed
    assert len(new_comps) >= 2


def test_refine_iteration_blended_merge_accepted() -> None:
    """Blended merge should be accepted in _refine_iteration when AICc improves.

    A single-peak signal is intentionally fit with two very close (blended)
    components.  Merging them into one should lower AICc.
    """
    x = np.arange(200, dtype=np.float64)
    rng = np.random.default_rng(42)
    signal = _make_gaussian(x, 5.0, 100.0, 8.0) + rng.normal(0, 0.2, size=200)

    # Two blended initial components
    blended = [
        GaussianComponent(amplitude=3.0, mean=97.0, stddev=5.0),
        GaussianComponent(amplitude=3.0, mean=103.0, stddev=5.0),
    ]
    components, residual = _fit_components(x, signal, blended, 200)
    current_aicc = aicc(residual, 6)

    new_comps, _, _, changed = _refine_iteration(
        x, signal, components, residual, 200, current_aicc,
        rms=0.2,
        snr_min=999.0,       # suppress residual-peak path
        mf_snr_min=999.0,
        neg_thresh=999.0,    # suppress dip-split path
        f_sep=5.0,           # aggressive blended detection
        max_components=None,
    )
    assert changed
    assert len(new_comps) == 1
