"""Tests for benchmarks.commands.train_synthetic."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from benchmarks._types import Component
from benchmarks.cli import main
from benchmarks.commands.train_synthetic import (
    GENERATORS,
    _agg_f1,
    _EvalResult,
    _evaluate_one,
    _gen_crowded,
    _gen_multi_blended,
    _gen_multi_separated,
    _gen_single_bright,
    _gen_single_broad,
    _gen_single_faint,
    _gen_single_narrow,
    _make_spectrum,
    _plot_error_boxplots,
    _plot_f1_vs_beta,
    _WorkItem,
)
from click.testing import CliRunner
from matplotlib import pyplot as plt


def test_make_spectrum() -> None:
    """_make_spectrum should return a noisy signal of N_CHANNELS length."""
    rng = np.random.default_rng(0)
    comps = [Component(2.0, 100.0, 5.0)]
    sig = _make_spectrum(rng, comps)
    assert sig.shape == (424,)
    assert sig[100] > sig[0]


def test_gen_single_bright() -> None:
    """_gen_single_bright should generate spectra with high SNR."""
    rng = np.random.default_rng(0)
    spectra = _gen_single_bright(rng, 2)
    assert len(spectra) == 2
    assert all(s.category == "single_bright" for s in spectra)
    assert all(len(s.components) == 1 for s in spectra)


def test_gen_single_faint() -> None:
    """_gen_single_faint should generate spectra with low SNR."""
    rng = np.random.default_rng(0)
    spectra = _gen_single_faint(rng, 2)
    assert len(spectra) == 2


def test_gen_single_narrow() -> None:
    """_gen_single_narrow should generate spectra with narrow lines."""
    rng = np.random.default_rng(0)
    spectra = _gen_single_narrow(rng, 2)
    assert len(spectra) == 2


def test_gen_single_broad() -> None:
    """_gen_single_broad should generate spectra with broad lines."""
    rng = np.random.default_rng(0)
    spectra = _gen_single_broad(rng, 2)
    assert len(spectra) == 2


def test_gen_multi_separated() -> None:
    """_gen_multi_separated should generate multi-component spectra."""
    rng = np.random.default_rng(0)
    spectra = _gen_multi_separated(rng, 2)
    assert len(spectra) == 2
    assert all(len(s.components) >= 2 for s in spectra)


def test_gen_multi_blended() -> None:
    """_gen_multi_blended should generate blended multi-component spectra."""
    rng = np.random.default_rng(0)
    spectra = _gen_multi_blended(rng, 2)
    assert len(spectra) == 2


def test_gen_crowded() -> None:
    """_gen_crowded should generate crowded spectra with 4+ components."""
    rng = np.random.default_rng(0)
    spectra = _gen_crowded(rng, 2)
    assert len(spectra) == 2
    assert all(len(s.components) >= 4 for s in spectra)


def test_generators_dict() -> None:
    """GENERATORS should contain all 7 categories."""
    assert len(GENERATORS) == 7


def test_evaluate_one() -> None:
    """_evaluate_one should return an _EvalResult."""
    rng = np.random.default_rng(0)
    comp = Component(2.0, 100.0, 5.0)
    sig = _make_spectrum(rng, [comp])
    item = _WorkItem(
        signal=sig,
        category="single_bright",
        spec_idx=0,
        true_comps_raw=[{"amplitude": 2.0, "mean": 100.0, "stddev": 5.0}],
        beta=3.8,
    )
    result = _evaluate_one(item)
    assert isinstance(result, _EvalResult)
    assert 0.0 <= result.f1 <= 1.0


def test_evaluate_one_exception() -> None:
    """_evaluate_one should handle LinAlgError gracefully."""
    item = _WorkItem(
        signal=np.zeros(424),
        category="single_bright",
        spec_idx=0,
        true_comps_raw=[{"amplitude": 2.0, "mean": 100.0, "stddev": 5.0}],
        beta=3.8,
    )
    with patch("benchmarks.commands.train_synthetic.fit_gaussians", side_effect=ValueError("test")):
        result = _evaluate_one(item)
    assert result.n_detected == 0
    assert result.f1 == 0.0


def test_agg_f1() -> None:
    """_agg_f1 should compute micro-averaged F1."""
    rows = [
        _EvalResult("cat_a", 0, 3.8, 1.0, 1.0, 1.0, 1, 1, 1, 1, 0, 0.1, None, None, None, None),
        _EvalResult("cat_b", 0, 3.8, 0.0, 0.0, 0.0, 0, 1, 0, 0, -1, 0.2, None, None, None, None),
    ]
    overall = _agg_f1(rows, 3.8)
    assert 0.0 <= overall <= 1.0

    cat_a = _agg_f1(rows, 3.8, "cat_a")
    assert cat_a == 1.0

    empty = _agg_f1(rows, 99.9)
    assert empty == 0.0


@pytest.mark.usefixtures("docs_img_dir")
def test_plot_f1_vs_beta() -> None:
    """_plot_f1_vs_beta should produce a figure."""
    rows = [
        _EvalResult("single_bright", 0, b, 1.0, 1.0, 1.0, 1, 1, 1, 1, 0, 0.1, None, None, None, None)
        for b in [3.8, 4.0]
    ]
    categories = ["single_bright"]
    fig = _plot_f1_vs_beta(rows, categories)
    assert fig is not None
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_plot_error_boxplots() -> None:
    """_plot_error_boxplots should produce a figure."""
    rows = [
        _EvalResult(
            "single_bright",
            0,
            3.8,
            1.0,
            1.0,
            1.0,
            1,
            1,
            1,
            1,
            0,
            0.1,
            10.0,
            0.01,
            0.02,
            0.03,
        )
    ]
    categories = ["single_bright"]
    fig = _plot_error_boxplots(rows, categories, 3.8)
    assert fig is not None
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_synthetic_cli() -> None:
    """CLI should run end-to-end with small parameters."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["train-synthetic", "--n-per-category", "2", "--beta-steps", "2", "--seed", "42"],
    )
    assert result.exit_code == 0, result.output
