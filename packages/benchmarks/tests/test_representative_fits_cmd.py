"""Tests for benchmarks.commands.representative_fits_plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from benchmarks._types import Component, SyntheticSpectrum
from benchmarks.cli import main
from benchmarks.commands.representative_fits_plot import (
    _build_panels,
    _build_representative_fits_figure,
    _draw_panel,
    _fit_signal,
    _FitPanel,
    _load_grs_noisy_panel,
    _make_synthetic_panel,
    _select_best_panel,
    _select_failure_panel,
)
from click.testing import CliRunner
from matplotlib import pyplot as plt

# pylint: disable=redefined-outer-name


def _make_synthetic_spectrum(seed: int) -> SyntheticSpectrum:
    """A 60-channel single-Gaussian synthetic spectrum with controlled noise."""
    rng = np.random.default_rng(seed)
    x = np.arange(60, dtype=np.float64)
    comp = Component(amplitude=3.0, mean=30.0, stddev=4.0)
    signal = comp.amplitude * np.exp(-0.5 * ((x - comp.mean) / comp.stddev) ** 2)
    signal = signal + rng.normal(0.0, 0.05, 60)
    return SyntheticSpectrum("single_bright", 0, signal, [comp])


@pytest.fixture()
def noisy_grs_fits(tmp_path: Path) -> Path:
    """Tiny GRS-like cube with at least one pixel above the 0.2 K noise floor."""
    rng = np.random.default_rng(7)
    cube = rng.normal(0.0, 0.05, (40, 4, 4)).astype(np.float32)
    # Inject an explicit noisy pixel (sigma well above 0.2 K).
    cube[:, 1, 2] = rng.normal(0.0, 0.4, 40)
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 4
    header["NAXIS2"] = 4
    header["NAXIS3"] = 40
    path = tmp_path / "grs-test-field.fits"
    fits.PrimaryHDU(data=cube, header=header).writeto(str(path), overwrite=True)
    return path


@pytest.fixture()
def quiet_grs_fits(tmp_path: Path) -> Path:
    """GRS-like cube whose every pixel sits below the 0.2 K noise floor."""
    rng = np.random.default_rng(11)
    cube = rng.normal(0.0, 0.05, (40, 4, 4)).astype(np.float32)
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 4
    header["NAXIS2"] = 4
    header["NAXIS3"] = 40
    path = tmp_path / "grs-test-field.fits"
    fits.PrimaryHDU(data=cube, header=header).writeto(str(path), overwrite=True)
    return path


def test_fit_signal_returns_components_or_empty() -> None:
    """_fit_signal returns a list of Component objects."""
    spec = _make_synthetic_spectrum(seed=1)
    fitted = _fit_signal(spec.signal, beta=3.5)
    assert isinstance(fitted, list)
    assert all(isinstance(c, Component) for c in fitted)


def test_make_synthetic_panel_populates_metrics() -> None:
    """_make_synthetic_panel fills truth, label, RMS, and F1."""
    spec = _make_synthetic_spectrum(seed=2)
    panel = _make_synthetic_panel(spec, beta=3.5, label="Test")
    assert panel.label == "Test"
    assert panel.truth == spec.components
    assert panel.rms >= 0.0
    assert panel.f1 is not None


def test_select_best_panel_picks_higher_f1() -> None:
    """_select_best_panel returns the candidate with the highest F1."""
    good = _make_synthetic_spectrum(seed=3)
    bad = SyntheticSpectrum(
        category="single_bright",
        index=1,
        signal=np.zeros(60, dtype=np.float64),  # zero signal -> 0 fitted comps
        components=[Component(3.0, 30.0, 4.0)],
    )
    best = _select_best_panel([bad, good], beta=3.5, label="Best")
    assert best.label == "Best"
    assert best.f1 is not None and best.f1 > 0.0


def test_select_failure_panel_picks_lower_f1() -> None:
    """_select_failure_panel returns the candidate with the lowest F1."""
    good = _make_synthetic_spectrum(seed=4)
    bad = SyntheticSpectrum(
        category="multi_blended",
        index=0,
        signal=np.zeros(60, dtype=np.float64),
        components=[Component(3.0, 30.0, 4.0)],
    )
    worst = _select_failure_panel([good, bad], beta=3.5)
    assert worst.label == "Failure"
    assert worst.f1 is not None and worst.f1 == 0.0


def test_load_grs_noisy_panel_noisy_branch(noisy_grs_fits: Path) -> None:
    """A pixel above the 0.2 K threshold is selected when one exists."""
    panel = _load_grs_noisy_panel(str(noisy_grs_fits), beta=3.5)
    assert panel.truth is None
    assert panel.f1 is None
    assert "Noisy GRS pixel" in panel.label
    assert panel.signal.shape == (40,)


def test_load_grs_noisy_panel_fallback_branch(quiet_grs_fits: Path) -> None:
    """Fallback to the highest-sigma pixel when nothing exceeds 0.2 K."""
    panel = _load_grs_noisy_panel(str(quiet_grs_fits), beta=3.5)
    assert panel.truth is None
    assert "Noisy GRS pixel" in panel.label


@pytest.mark.usefixtures("docs_img_dir")
def test_draw_panel_with_truth() -> None:
    """_draw_panel renders the synthetic case (with truth overlay)."""
    spec = _make_synthetic_spectrum(seed=5)
    panel = _make_synthetic_panel(spec, beta=3.5, label="Synthetic")
    fig, ax = plt.subplots()
    _draw_panel(ax, panel)
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_draw_panel_without_truth_and_no_fit() -> None:
    """_draw_panel handles the no-truth, no-fitted-components branch."""
    panel = _FitPanel(
        signal=np.zeros(60, dtype=np.float64),
        fitted=[],
        truth=None,
        label="Empty",
        rms=0.0,
        f1=None,
    )
    fig, ax = plt.subplots()
    _draw_panel(ax, panel)
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_build_representative_fits_figure_shape() -> None:
    """The figure has six axes (2x3 grid)."""
    spec = _make_synthetic_spectrum(seed=6)
    panels = [_make_synthetic_panel(spec, beta=3.5, label=f"P{i}") for i in range(6)]
    fig = _build_representative_fits_figure(panels)
    assert len(fig.axes) == 6
    plt.close(fig)


@pytest.mark.usefixtures("docs_img_dir")
def test_build_panels_returns_six(noisy_grs_fits: Path) -> None:
    """_build_panels returns exactly six panels with the GRS one having no truth."""
    panels = _build_panels(str(noisy_grs_fits), beta=3.5, seed=42)
    assert len(panels) == 6
    assert panels[-1].truth is None  # GRS is always the last panel


@pytest.mark.usefixtures("docs_img_dir")
def test_cli_end_to_end(noisy_grs_fits: Path) -> None:
    """The CLI runs to completion with a custom --fits-path."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["representative-fits-plot", "--fits-path", str(noisy_grs_fits)],
    )
    assert result.exit_code == 0, result.output


def test_cli_missing_fits_aborts(tmp_path: Path) -> None:
    """The CLI exits non-zero when the FITS file does not exist."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["representative-fits-plot", "--fits-path", str(tmp_path / "does-not-exist.fits")],
    )
    assert result.exit_code != 0
