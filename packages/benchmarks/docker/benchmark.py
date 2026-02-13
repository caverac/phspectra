"""GaussPy+ performance benchmark -- runs inside Docker.

Loads real GRS spectra from a mounted .npz file, decomposes them with
GaussPy+ in batch mode, then times each spectrum individually for the
timing distribution.  Writes results as JSON to /data/results.json.

Note: ``gausspy`` and ``gausspyplus`` are not installed in the local
environment because they conflict with numpy>=2 and other dependencies.
They are only available inside the Docker container (Python 3.10,
numpy==1.23.5).  We define strict Protocol interfaces below so that
the code is type-safe without the actual imports.
"""

# pylint: disable=import-outside-toplevel, import-error

from __future__ import annotations

import json
import logging
import os
import pickle  # noqa: S403
import time
from typing import Protocol

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class GaussianDecomposer(Protocol):
    """Protocol for ``gausspy.gp.GaussianDecomposer``."""

    def set(self, key: str, value: object) -> None:
        """Set a decomposer parameter."""

    def decompose(
        self,
        x: npt.NDArray[np.float64],
        signal: npt.NDArray[np.float64],
        errors: npt.NDArray[np.float64],
    ) -> dict[str, float | list[float] | None]:
        """Decompose a single spectrum."""
        ...


class BatchDecomposer(Protocol):
    """Protocol for ``gausspyplus.decompose.GaussPyDecompose``."""

    dirpath_gpy: str
    alpha1: float
    alpha2: float
    two_phase_decomposition: bool
    snr_thresh: float
    snr2_thresh: float
    use_ncpus: int
    verbose: bool
    suffix: str

    def decompose(self) -> None:
        """Run batch decomposition on the loaded spectra."""


# Trained smoothing parameters for GRS (Riener et al. 2019, Sect. 4.1)
# Two-phase decomposition: alpha1 for narrow peaks, alpha2 for broad peaks
SPECTRA_PATH = "/data/spectra.npz"
RESULTS_PATH = "/data/results.json"
NOISE_SIGMA = 0.13  # GRS noise level (K)
ALPHA1 = 2.89
ALPHA2 = 6.65
WORK_DIR = "/tmp/gpp"


def _make_decomposer() -> GaussianDecomposer:
    """Create a configured GaussPy decomposer (GaussPy+ core engine).

    Returns
    -------
    GaussianDecomposer
        Configured decomposer with two-phase parameters.
    """
    from gausspy import gp  # type: ignore[import-not-found]

    decomposer: GaussianDecomposer = gp.GaussianDecomposer()
    decomposer.set("phase", "two")
    decomposer.set("alpha1", ALPHA1)
    decomposer.set("alpha2", ALPHA2)
    decomposer.set("SNR_thresh", [3.0, 3.0])
    return decomposer


def _decompose_one(
    decomposer: GaussianDecomposer,
    signal: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    errors: npt.NDArray[np.float64],
) -> tuple[int, float]:
    """Decompose one spectrum and return (n_components, elapsed).

    Parameters
    ----------
    decomposer : GaussianDecomposer
        Pre-configured GaussPy decomposer.
    signal : npt.NDArray[np.float64]
        1-D spectrum array.
    x : npt.NDArray[np.float64]
        Channel index array.
    errors : npt.NDArray[np.float64]
        Per-channel error array.

    Returns
    -------
    tuple[int, float]
        Number of detected components and elapsed wall-clock time.
    """
    t_start = time.perf_counter()
    try:
        result = decomposer.decompose(x, signal, errors)
        n_det = int(result.get("N_components", 0))  # type: ignore[arg-type]
    except Exception:  # pylint: disable=broad-exception-caught
        n_det = 0
    elapsed = time.perf_counter() - t_start
    return n_det, elapsed


# Batch decomposition


class BatchResults:
    """Parsed results from a GaussPy+ batch decomposition."""

    def __init__(self, n_spectra: int) -> None:
        """Initialise empty result arrays for *n_spectra* spectra."""
        self.n_components: list[int] = [0] * n_spectra
        self.amplitudes_fit: list[list[float]] = [[] for _ in range(n_spectra)]
        self.means_fit: list[list[float]] = [[] for _ in range(n_spectra)]
        self.stddevs_fit: list[list[float]] = [[] for _ in range(n_spectra)]

    def to_dict(self) -> dict[str, list[int] | list[list[float]]]:
        """Convert to a JSON-serialisable dictionary.

        Returns
        -------
        dict[str, list[int] | list[list[float]]]
            Keys: ``n_components``, ``amplitudes_fit``, ``means_fit``,
            ``stddevs_fit``.
        """
        return {
            "n_components": self.n_components,
            "amplitudes_fit": self.amplitudes_fit,
            "means_fit": self.means_fit,
            "stddevs_fit": self.stddevs_fit,
        }


def _run_batch(
    signals: npt.NDArray[np.float64],
    x_values: npt.NDArray[np.float64],
    errors: npt.NDArray[np.float64],
) -> tuple[BatchResults, float]:
    """Run GaussPy+ batch decomposition and return (results, elapsed).

    Parameters
    ----------
    signals : npt.NDArray[np.float64]
        2-D array of spectra ``(n_spectra, n_channels)``.
    x_values : npt.NDArray[np.float64]
        Channel index array.
    errors : npt.NDArray[np.float64]
        Per-channel error array.

    Returns
    -------
    tuple[BatchResults, float]
        Parsed batch results and elapsed wall-clock time.
    """
    n_spectra = signals.shape[0]

    input_data = {
        "data_list": [signals[i].astype(np.float64) for i in range(n_spectra)],
        "x_values": x_values,
        "error": [errors.copy() for _ in range(n_spectra)],
        "index": list(range(n_spectra)),
        "location": [(i, 0) for i in range(n_spectra)],
    }

    pickle_path = os.path.join(WORK_DIR, "spectra.pickle")
    with open(pickle_path, "wb") as fobj:
        pickle.dump(input_data, fobj, protocol=2)

    files_before = set(os.listdir(WORK_DIR))

    from gausspyplus.decompose import GaussPyDecompose  # type: ignore[import-not-found]

    decomposer: BatchDecomposer = GaussPyDecompose(path_to_pickle_file=pickle_path)
    decomposer.dirpath_gpy = WORK_DIR
    decomposer.alpha1 = ALPHA1
    decomposer.alpha2 = ALPHA2
    decomposer.two_phase_decomposition = True
    decomposer.snr_thresh = 3.0
    decomposer.snr2_thresh = 3.0
    decomposer.use_ncpus = 1
    decomposer.verbose = False
    decomposer.suffix = ""

    logger.info("Running GaussPy+ batch decomposition...")
    t_start = time.perf_counter()
    decomposer.decompose()
    elapsed = time.perf_counter() - t_start
    logger.info("Batch complete in %.1fs", elapsed)

    result_path = _find_result_pickle(files_before)
    return _parse_batch_results(result_path, n_spectra), elapsed


def _find_result_pickle(files_before: set[str]) -> str | None:
    """Locate the GaussPy+ output pickle file.

    Parameters
    ----------
    files_before : set[str]
        Files in ``WORK_DIR`` before decomposition ran.

    Returns
    -------
    str or None
        Path to the output pickle, or ``None`` if not found.
    """
    decomposed_dir = os.path.join(WORK_DIR, "gpy_decomposed")
    if os.path.isdir(decomposed_dir):
        for name in sorted(os.listdir(decomposed_dir)):
            if name.endswith(".pickle"):
                return os.path.join(decomposed_dir, name)

    files_after = set(os.listdir(WORK_DIR))
    for name in sorted(files_after - files_before):
        candidate = os.path.join(WORK_DIR, name)
        if os.path.isdir(candidate):
            for fname in sorted(os.listdir(candidate)):
                if fname.endswith(".pickle"):
                    return os.path.join(candidate, fname)

    return None


def _parse_batch_results(
    result_path: str | None,
    n_spectra: int,
) -> BatchResults:
    """Parse the GaussPy+ batch output pickle into component lists.

    Parameters
    ----------
    result_path : str or None
        Path to the GaussPy+ output pickle.
    n_spectra : int
        Expected number of spectra.

    Returns
    -------
    BatchResults
        Parsed component lists.
    """
    batch = BatchResults(n_spectra)

    if result_path is None:
        logger.warning("Could not find output pickle")
        return batch

    logger.info("Loading results from %s", os.path.basename(result_path))
    with open(result_path, "rb") as fobj:
        results = pickle.load(fobj)  # noqa: S301
    logger.debug("Result keys: %s", list(results.keys()))

    if "N_components" in results:
        batch.n_components = [int(n) for n in results["N_components"]]

    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if "amplitudes_fit" in results:
        for i in range(min(n_spectra, len(results["amplitudes_fit"]))):
            amps = results["amplitudes_fit"][i]
            means = results["means_fit"][i]
            fwhms = results["fwhms_fit"][i]
            if amps is not None and len(amps) > 0:
                batch.amplitudes_fit[i] = [round(float(a), 6) for a in amps]
                batch.means_fit[i] = [round(float(m), 4) for m in means]
                batch.stddevs_fit[i] = [round(float(f) * fwhm_to_sigma, 4) for f in fwhms]
                if not batch.n_components[i]:
                    batch.n_components[i] = len(amps)

    return batch


def main() -> None:
    """Run the two-phase GaussPy+ benchmark and write results to JSON."""
    os.makedirs(WORK_DIR, exist_ok=True)

    logger.info("Loading spectra...")
    data = np.load(SPECTRA_PATH)
    signals = data["signals"]
    n_spectra, n_channels = signals.shape
    logger.info("%d spectra, %d channels each", n_spectra, n_channels)

    x_values = np.arange(n_channels, dtype=np.float64)
    errors = np.full(n_channels, NOISE_SIGMA, dtype=np.float64)

    # Phase 1: GaussPy+ batch decomposition (accuracy)
    logger.info("Preparing GaussPy+ input...")
    batch, t_batch = _run_batch(signals, x_values, errors)

    # Phase 2: Per-spectrum timing
    logger.info("Timing individual spectra...")
    decomposer = _make_decomposer()
    times: list[float] = []

    t_total_start = time.perf_counter()
    for i in range(n_spectra):
        _, elapsed = _decompose_one(decomposer, signals[i], x_values, errors)
        times.append(elapsed)
        if (i + 1) % 50 == 0:
            logger.info("  %d/%d", i + 1, n_spectra)
    t_total = time.perf_counter() - t_total_start

    mean_n = float(np.mean(batch.n_components))
    times_arr = np.array(times)

    output: dict[str, object] = {
        "tool": "gausspyplus",
        "alpha1": ALPHA1,
        "alpha2": ALPHA2,
        "phase": "two",
        "n_spectra": n_spectra,
        "batch_time_s": round(t_batch, 3),
        "total_time_s": round(t_total, 3),
        "mean_time_per_spectrum_s": round(t_total / n_spectra, 6),
        "median_time_per_spectrum_s": round(float(np.median(times_arr)), 6),
        "p95_time_per_spectrum_s": round(float(np.percentile(times_arr, 95)), 6),
        "max_time_per_spectrum_s": round(float(np.max(times_arr)), 6),
        "mean_n_components": round(mean_n, 2),
        **batch.to_dict(),
        "times": [round(t, 6) for t in times],
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as fobj:
        json.dump(output, fobj, indent=2)

    logger.info(
        "Done: %.1fs total, %.1fms/spectrum, mean %.1f components",
        t_total,
        t_total / n_spectra * 1000,
        mean_n,
    )
    logger.info("Results written to %s", RESULTS_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
