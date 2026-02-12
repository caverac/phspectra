"""Tests for the Docker benchmark module (from gausspyplus-test).

Since ``gausspy`` and ``gausspyplus`` are not available locally, these tests
cover the pure-Python helpers (BatchResults, pickle parsing, result-file
discovery) and mock the external decomposer interface.
"""

from __future__ import annotations

import json
import os
import pickle  # noqa: S403
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

# Import the Docker benchmark module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docker"))

import benchmark  # noqa: E402


class TestBatchResults:
    """Tests for the BatchResults helper class."""

    def test_init_zeros(self) -> None:
        """Empty BatchResults should have zero-filled component counts."""
        batch = benchmark.BatchResults(5)
        assert batch.n_components == [0, 0, 0, 0, 0]
        assert all(lst == [] for lst in batch.amplitudes_fit)

    def test_init_lengths(self) -> None:
        """All lists should have length equal to n_spectra."""
        batch = benchmark.BatchResults(10)
        assert len(batch.n_components) == 10
        assert len(batch.amplitudes_fit) == 10

    def test_to_dict_keys(self) -> None:
        """to_dict should return the four expected keys."""
        batch = benchmark.BatchResults(3)
        result = batch.to_dict()
        assert set(result.keys()) == {
            "n_components",
            "amplitudes_fit",
            "means_fit",
            "stddevs_fit",
        }

    def test_to_dict_roundtrip(self) -> None:
        """to_dict output should be JSON-serialisable."""
        batch = benchmark.BatchResults(2)
        batch.n_components = [3, 1]
        batch.amplitudes_fit[0] = [1.0, 2.0, 3.0]
        serialised = json.dumps(batch.to_dict())
        loaded = json.loads(serialised)
        assert loaded["n_components"] == [3, 1]

    def test_lists_are_independent(self) -> None:
        """Each spectrum's fit list should be an independent object."""
        batch = benchmark.BatchResults(3)
        batch.amplitudes_fit[0].append(1.0)
        assert batch.amplitudes_fit[1] == []


class TestParseBatchResults:
    """Tests for _parse_batch_results with real pickle files."""

    def test_none_path_returns_empty(self) -> None:
        """A None result path should return an empty BatchResults."""
        batch = benchmark._parse_batch_results(None, 5)
        assert batch.n_components == [0, 0, 0, 0, 0]

    def test_parses_n_components(self, tmp_path: Any) -> None:
        """Should parse N_components from a pickle file."""
        data = {"N_components": [2, 0, 3]}
        pkl_path = str(tmp_path / "result.pickle")
        with open(pkl_path, "wb") as fobj:
            pickle.dump(data, fobj)
        batch = benchmark._parse_batch_results(pkl_path, 3)
        assert batch.n_components == [2, 0, 3]

    def test_parses_fit_parameters(self, tmp_path: Any) -> None:
        """Should parse amplitudes, means, and FWHMs from the pickle."""
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        data = {
            "N_components": [1, 0],
            "amplitudes_fit": [[5.0], None],
            "means_fit": [[100.0], None],
            "fwhms_fit": [[10.0], None],
        }
        pkl_path = str(tmp_path / "result.pickle")
        with open(pkl_path, "wb") as fobj:
            pickle.dump(data, fobj)
        batch = benchmark._parse_batch_results(pkl_path, 2)
        assert batch.n_components == [1, 0]
        assert batch.amplitudes_fit[0][0] == pytest.approx(5.0, abs=1e-5)
        expected_sigma = round(10.0 * fwhm_to_sigma, 4)
        assert batch.stddevs_fit[0][0] == pytest.approx(expected_sigma, abs=1e-4)


class TestFindResultPickle:
    """Tests for _find_result_pickle."""

    def test_finds_in_gpy_decomposed(self, tmp_path: Any) -> None:
        """Should find pickle in the gpy_decomposed subdirectory."""
        with patch.object(benchmark, "WORK_DIR", str(tmp_path)):
            decomposed = tmp_path / "gpy_decomposed"
            decomposed.mkdir()
            (decomposed / "output.pickle").write_bytes(b"data")
            result = benchmark._find_result_pickle(set())
            assert result is not None
            assert result.endswith("output.pickle")

    def test_returns_none_when_no_pickle(self, tmp_path: Any) -> None:
        """Should return None when no pickle file exists."""
        with patch.object(benchmark, "WORK_DIR", str(tmp_path)):
            result = benchmark._find_result_pickle(set())
            assert result is None


class TestDecomposeOne:
    """Tests for _decompose_one with a mock decomposer."""

    def test_returns_count_and_time(self) -> None:
        """Should return the detected component count and a positive elapsed time."""

        class FakeDecomposer:
            def set(self, key: str, value: object) -> None:
                pass

            def decompose(
                self,
                x: npt.NDArray[np.float64],
                signal: npt.NDArray[np.float64],
                errors: npt.NDArray[np.float64],
            ) -> dict[str, object]:
                return {"N_components": 3}

        decomposer = FakeDecomposer()
        x = np.arange(100, dtype=np.float64)
        signal = np.zeros(100, dtype=np.float64)
        errors = np.ones(100, dtype=np.float64)
        n_det, elapsed = benchmark._decompose_one(decomposer, signal, x, errors)
        assert n_det == 3
        assert elapsed > 0

    def test_handles_exception(self) -> None:
        """Should return 0 components when decompose raises an exception."""

        class FailingDecomposer:
            def set(self, key: str, value: object) -> None:
                pass

            def decompose(
                self,
                x: npt.NDArray[np.float64],
                signal: npt.NDArray[np.float64],
                errors: npt.NDArray[np.float64],
            ) -> dict[str, object]:
                raise RuntimeError("decomposition failed")

        decomposer = FailingDecomposer()
        x = np.arange(100, dtype=np.float64)
        signal = np.zeros(100, dtype=np.float64)
        errors = np.ones(100, dtype=np.float64)
        n_det, elapsed = benchmark._decompose_one(decomposer, signal, x, errors)
        assert n_det == 0
        assert elapsed > 0


class TestConstants:
    """Sanity checks on module-level constants."""

    def test_alpha_values_positive(self) -> None:
        assert benchmark.ALPHA1 > 0
        assert benchmark.ALPHA2 > 0

    def test_alpha2_greater_than_alpha1(self) -> None:
        assert benchmark.ALPHA2 > benchmark.ALPHA1

    def test_noise_sigma_positive(self) -> None:
        assert 0 < benchmark.NOISE_SIGMA < 1.0

    def test_paths_are_absolute(self) -> None:
        assert benchmark.SPECTRA_PATH.startswith("/")
        assert benchmark.RESULTS_PATH.startswith("/")
