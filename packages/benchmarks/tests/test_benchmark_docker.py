"""Tests for the Docker benchmark module (per-spectrum decomposer with timing).

Since ``gausspy`` and ``gausspyplus`` are not available locally, these tests
cover the pure-Python helpers (_load_result, main) and mock the
external GaussPyDecompose interface.
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


class TestLoadResult:
    """Tests for _load_result."""

    def test_finds_pickle(self, tmp_path: Any) -> None:
        """Should load and return the decomposition result pickle."""
        decomposed = tmp_path / "gpy_decomposed"
        decomposed.mkdir()
        data = {
            "N_components": [1],
            "amplitudes_fit": [[2.0]],
            "means_fit": [[100.0]],
            "fwhms_fit": [[10.0]],
        }
        pkl_path = decomposed / "result.pickle"
        with open(pkl_path, "wb") as fobj:
            pickle.dump(data, fobj)

        result = benchmark._load_result(str(tmp_path))
        assert result["N_components"] == [1]
        assert result["amplitudes_fit"][0] == [2.0]

    def test_returns_empty_when_no_pickle(self, tmp_path: Any) -> None:
        """Should return empty lists when no pickle file exists."""
        result = benchmark._load_result(str(tmp_path))
        assert result["N_components"] == [0]
        assert result["amplitudes_fit"] == [[]]

    def test_returns_empty_when_no_dir(self, tmp_path: Any) -> None:
        """Should return empty lists when gpy_decomposed dir is missing."""
        result = benchmark._load_result(str(tmp_path / "nonexistent"))
        assert result["N_components"] == [0]


class TestMain:
    """Tests for the main() function with mocked per-spectrum decomposer."""

    def test_writes_results_json(self, tmp_path: Any) -> None:
        """main() should write results JSON with expected keys."""
        n_spectra, n_channels = 3, 50
        signals = np.random.default_rng(42).standard_normal((n_spectra, n_channels))
        spectra_path = str(tmp_path / "spectra.npz")
        results_path = str(tmp_path / "results.json")
        np.savez(spectra_path, signals=signals)

        call_count = 0
        single_results = [
            {
                "N_components": [1],
                "amplitudes_fit": [[2.0]],
                "means_fit": [[25.0]],
                "fwhms_fit": [[5.0]],
            },
            {
                "N_components": [2],
                "amplitudes_fit": [[1.0, 3.0]],
                "means_fit": [[10.0, 40.0]],
                "fwhms_fit": [[8.0, 12.0]],
            },
            {
                "N_components": [0],
                "amplitudes_fit": [[]],
                "means_fit": [[]],
                "fwhms_fit": [[]],
            },
        ]

        def fake_decompose_one(
            _sig: Any,
            _x: Any,
            _err: Any,
            idx: int,
        ) -> tuple[dict[str, Any], float]:
            nonlocal call_count
            result = single_results[call_count]
            call_count += 1
            return result, 0.1 * (idx + 1)

        with (
            patch.object(benchmark, "SPECTRA_PATH", spectra_path),
            patch.object(benchmark, "RESULTS_PATH", results_path),
            patch.object(benchmark, "_decompose_one", side_effect=fake_decompose_one),
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

    def test_fwhm_to_sigma_conversion(self, tmp_path: Any) -> None:
        """FWHM values should be converted to sigma in output."""
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        n_spectra, n_channels = 1, 50
        signals = np.random.default_rng(42).standard_normal((n_spectra, n_channels))
        spectra_path = str(tmp_path / "spectra.npz")
        results_path = str(tmp_path / "results.json")
        np.savez(spectra_path, signals=signals)

        single_result = {
            "N_components": [1],
            "amplitudes_fit": [[2.0]],
            "means_fit": [[25.0]],
            "fwhms_fit": [[10.0]],
        }

        def fake_decompose_one(
            _sig: Any,
            _x: Any,
            _err: Any,
            _idx: int,
        ) -> tuple[dict[str, Any], float]:
            return single_result, 0.05

        with (
            patch.object(benchmark, "SPECTRA_PATH", spectra_path),
            patch.object(benchmark, "RESULTS_PATH", results_path),
            patch.object(benchmark, "_decompose_one", side_effect=fake_decompose_one),
        ):
            benchmark.main()

        with open(results_path, encoding="utf-8") as fobj:
            output = json.load(fobj)

        expected_sigma = round(10.0 * fwhm_to_sigma, 4)
        assert output["stddevs_fit"][0][0] == pytest.approx(expected_sigma, abs=1e-4)

    def test_progress_logging(self, tmp_path: Any) -> None:
        """main() should log progress every 100 spectra."""
        n_spectra, n_channels = 200, 50
        signals = np.random.default_rng(42).standard_normal((n_spectra, n_channels))
        spectra_path = str(tmp_path / "spectra.npz")
        results_path = str(tmp_path / "results.json")
        np.savez(spectra_path, signals=signals)

        empty = {
            "N_components": [0],
            "amplitudes_fit": [[]],
            "means_fit": [[]],
            "fwhms_fit": [[]],
        }

        def fake_decompose_one(
            _sig: Any,
            _x: Any,
            _err: Any,
            _idx: int,
        ) -> tuple[dict[str, Any], float]:
            return empty, 0.001

        with (
            patch.object(benchmark, "SPECTRA_PATH", spectra_path),
            patch.object(benchmark, "RESULTS_PATH", results_path),
            patch.object(benchmark, "_decompose_one", side_effect=fake_decompose_one),
        ):
            benchmark.main()

        with open(results_path, encoding="utf-8") as fobj:
            output = json.load(fobj)

        assert output["n_spectra"] == n_spectra
        assert len(output["times"]) == n_spectra


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
