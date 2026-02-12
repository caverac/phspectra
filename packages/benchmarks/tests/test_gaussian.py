"""Tests for benchmarks._gaussian."""

from __future__ import annotations

import numpy as np
from benchmarks._gaussian import gaussian, gaussian_model, residual_rms
from benchmarks._types import Component


def test_gaussian_peak() -> None:
    """Gaussian should peak at the mean with the given amplitude."""
    x = np.arange(200, dtype=np.float64)
    y = gaussian(x, amp=5.0, mean=100.0, stddev=5.0)
    assert y[100] == np.float64(5.0)
    assert y[0] < 1e-10


def test_gaussian_symmetry() -> None:
    """Gaussian should be symmetric about the mean."""
    x = np.arange(200, dtype=np.float64)
    y = gaussian(x, amp=3.0, mean=100.0, stddev=8.0)
    np.testing.assert_allclose(y[90], y[110], atol=1e-12)


def test_gaussian_model_sum() -> None:
    """gaussian_model should sum multiple components."""
    x = np.arange(300, dtype=np.float64)
    comps = [
        Component(amplitude=5.0, mean=100.0, stddev=5.0),
        Component(amplitude=3.0, mean=200.0, stddev=4.0),
    ]
    model = gaussian_model(x, comps)
    assert model[100] > 4.9
    assert model[200] > 2.9
    assert model.shape == (300,)


def test_gaussian_model_empty() -> None:
    """gaussian_model with no components should return zeros."""
    x = np.arange(100, dtype=np.float64)
    model = gaussian_model(x, [])
    np.testing.assert_array_equal(model, np.zeros(100))


def test_residual_rms_perfect_fit() -> None:
    """RMS should be near zero when model matches signal exactly."""
    x = np.arange(200, dtype=np.float64)
    comps = [Component(amplitude=5.0, mean=100.0, stddev=5.0)]
    signal = gaussian_model(x, comps)
    rms = residual_rms(signal, comps)
    assert rms < 1e-10


def test_residual_rms_empty() -> None:
    """RMS with no components should be the signal RMS."""
    signal = np.ones(100)
    rms = residual_rms(signal, [])
    assert abs(rms - 1.0) < 1e-10
