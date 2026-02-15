"""Tests for the Docker benchmark module (batch decomposer with timing).

Since ``gausspy`` and ``gausspyplus`` are not available locally, these tests
cover the pure-Python helpers (_load_batch_results, main) and mock the
external batch decomposer interface.
"""

# pylint: disable=protected-access

from __future__ import annotations

import json
import pickle  # noqa: S403
from typing import Any
from unittest.mock import patch

import benchmark
import numpy as np
import pytest


class TestLoadBatchResults:
    """Tests for _load_batch_results."""

    def test_finds_pickle(self, tmp_path: Any) -> None:
        """Should load and return the batch decomposition pickle."""
        decomposed = tmp_path / "gpy_decomposed"
        decomposed.mkdir()
        data = {
            "N_components": [1, 0],
            "amplitudes_fit": [[2.0], []],
            "means_fit": [[100.0], []],
            "fwhms_fit": [[10.0], []],
        }
        pkl_path = decomposed / "result.pickle"
        with open(pkl_path, "wb") as fobj:
            pickle.dump(data, fobj)

        with patch.object(benchmark, "WORK_DIR", str(tmp_path)):
            result = benchmark._load_batch_results(2)
        assert result["N_components"] == [1, 0]
        assert result["amplitudes_fit"][0] == [2.0]

    def test_returns_empty_when_no_pickle(self, tmp_path: Any) -> None:
        """Should return empty lists when no pickle file exists."""
        with patch.object(benchmark, "WORK_DIR", str(tmp_path)):
            result = benchmark._load_batch_results(3)
        assert result["N_components"] == [0, 0, 0]
        assert all(lst == [] for lst in result["amplitudes_fit"])

    def test_returns_empty_when_no_dir(self, tmp_path: Any) -> None:
        """Should return empty lists when gpy_decomposed dir is missing."""
        with patch.object(benchmark, "WORK_DIR", str(tmp_path / "nonexistent")):
            result = benchmark._load_batch_results(2)
        assert result["N_components"] == [0, 0]


class TestMain:
    """Tests for the main() function with mocked batch decomposer."""

    def test_writes_results_json(self, tmp_path: Any) -> None:
        """main() should write results JSON with expected keys."""
        n_spectra, n_channels = 3, 50
        signals = np.random.default_rng(42).standard_normal((n_spectra, n_channels))
        spectra_path = str(tmp_path / "spectra.npz")
        results_path = str(tmp_path / "results.json")
        np.savez(spectra_path, signals=signals)

        batch_output = {
            "N_components": [1, 2, 0],
            "amplitudes_fit": [[2.0], [1.0, 3.0], []],
            "means_fit": [[25.0], [10.0, 40.0], []],
            "fwhms_fit": [[5.0], [8.0, 12.0], []],
        }

        def fake_run_batch(
            _sigs: Any,
            _x: Any,
            _errs: Any,
        ) -> dict[str, Any]:
            # Simulate per-spectrum timing
            benchmark.per_spectrum_times.extend([0.1, 0.2, 0.05])
            return batch_output

        with (
            patch.object(benchmark, "SPECTRA_PATH", spectra_path),
            patch.object(benchmark, "RESULTS_PATH", results_path),
            patch.object(benchmark, "_run_batch", side_effect=fake_run_batch),
        ):
            benchmark.main()

        with open(results_path, encoding="utf-8") as fobj:
            output = json.load(fobj)

        assert output["tool"] == "gausspyplus"
        assert output["n_spectra"] == n_spectra
        assert output["total_time_s"] >= 0
        assert len(output["times"]) == n_spectra
        assert len(output["n_components"]) == n_spectra
        assert output["n_components"] == [1, 2, 0]
        assert len(output["amplitudes_fit"]) == n_spectra
        assert len(output["means_fit"]) == n_spectra
        assert len(output["stddevs_fit"]) == n_spectra
        assert output["amplitudes_fit"][0] == [2.0]
        assert output["amplitudes_fit"][2] == []

    def test_pads_short_times(self, tmp_path: Any) -> None:
        """main() should pad per_spectrum_times if shorter than n_spectra."""
        n_spectra, n_channels = 3, 50
        signals = np.random.default_rng(42).standard_normal((n_spectra, n_channels))
        spectra_path = str(tmp_path / "spectra.npz")
        results_path = str(tmp_path / "results.json")
        np.savez(spectra_path, signals=signals)

        batch_output = {
            "N_components": [0, 0, 0],
            "amplitudes_fit": [[], [], []],
            "means_fit": [[], [], []],
            "fwhms_fit": [[], [], []],
        }

        def fake_run_batch(
            _sigs: Any,
            _x: Any,
            _errs: Any,
        ) -> dict[str, Any]:
            # Only 1 time recorded (simulating skipped None spectra)
            benchmark.per_spectrum_times.extend([0.1])
            return batch_output

        with (
            patch.object(benchmark, "SPECTRA_PATH", spectra_path),
            patch.object(benchmark, "RESULTS_PATH", results_path),
            patch.object(benchmark, "_run_batch", side_effect=fake_run_batch),
        ):
            benchmark.main()

        with open(results_path, encoding="utf-8") as fobj:
            output = json.load(fobj)

        assert len(output["times"]) == n_spectra
        assert output["times"][0] == 0.1
        assert output["times"][1] == 0.0
        assert output["times"][2] == 0.0

    def test_fwhm_to_sigma_conversion(self, tmp_path: Any) -> None:
        """Batch FWHM values should be converted to sigma in output."""
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        n_spectra, n_channels = 1, 50
        signals = np.random.default_rng(42).standard_normal((n_spectra, n_channels))
        spectra_path = str(tmp_path / "spectra.npz")
        results_path = str(tmp_path / "results.json")
        np.savez(spectra_path, signals=signals)

        batch_output = {
            "N_components": [1],
            "amplitudes_fit": [[2.0]],
            "means_fit": [[25.0]],
            "fwhms_fit": [[10.0]],
        }

        def fake_run_batch(
            _sigs: Any,
            _x: Any,
            _errs: Any,
        ) -> dict[str, Any]:
            benchmark.per_spectrum_times.extend([0.05])
            return batch_output

        with (
            patch.object(benchmark, "SPECTRA_PATH", spectra_path),
            patch.object(benchmark, "RESULTS_PATH", results_path),
            patch.object(benchmark, "_run_batch", side_effect=fake_run_batch),
        ):
            benchmark.main()

        with open(results_path, encoding="utf-8") as fobj:
            output = json.load(fobj)

        expected_sigma = round(10.0 * fwhm_to_sigma, 4)
        assert output["stddevs_fit"][0][0] == pytest.approx(expected_sigma, abs=1e-4)


class TestConstants:
    """Sanity checks on module-level constants."""

    def test_alpha_values_positive(self) -> None:
        """Both alpha smoothing parameters should be positive."""
        assert benchmark.ALPHA1 > 0
        assert benchmark.ALPHA2 > 0

    def test_alpha2_greater_than_alpha1(self) -> None:
        """Alpha2 (broad) should be larger than alpha1 (narrow)."""
        assert benchmark.ALPHA2 > benchmark.ALPHA1

    def test_noise_sigma_positive(self) -> None:
        """Noise sigma should be a small positive number."""
        assert 0 < benchmark.NOISE_SIGMA < 1.0

    def test_paths_are_absolute(self) -> None:
        """Docker paths should be absolute."""
        assert benchmark.SPECTRA_PATH.startswith("/")
        assert benchmark.RESULTS_PATH.startswith("/")

    def test_fwhm_to_sigma(self) -> None:
        """FWHM_TO_SIGMA should match the analytical conversion factor."""
        expected = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        assert benchmark.FWHM_TO_SIGMA == pytest.approx(expected)
