"""GaussPy+ performance benchmark -- runs inside Docker.

Loads real GRS spectra from a mounted .npz file and decomposes each one
individually with the full GaussPy+ pipeline (two-phase decomposition
with improve_fitting).  Running one spectrum per batch ensures that
per-spectrum timing includes all GaussPy+ overhead -- initialization,
decomposition, refinement -- giving a fair apples-to-apples comparison
with PHSpectra which also processes spectra individually.

Writes results as JSON to /data/results.json.

Note: ``gausspy`` and ``gausspyplus`` are not installed in the local
environment because they conflict with numpy>=2 and other dependencies.
They are only available inside the Docker container (Python 3.10,
numpy==1.23.5).
"""

# pylint: disable=import-outside-toplevel, import-error

from __future__ import annotations

import json
import logging
import os
import pickle  # noqa: S403
import shutil
import time
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Trained smoothing parameters for GRS (Riener et al. 2019, Sect. 4.1)
# Two-phase decomposition: alpha1 for narrow peaks, alpha2 for broad peaks
SPECTRA_PATH = "/data/spectra.npz"
RESULTS_PATH = "/data/results.json"
NOISE_SIGMA = 0.13  # GRS noise level (K)
ALPHA1 = 2.89
ALPHA2 = 6.65
WORK_DIR = "/tmp/gpp"
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _decompose_one(
    signal: npt.NDArray[np.float64],
    x_values: npt.NDArray[np.float64],
    errors: npt.NDArray[np.float64],
    idx: int,
) -> tuple[dict[str, Any], float]:
    """Run the full GaussPy+ pipeline on a single spectrum.

    Creates a single-spectrum pickle and runs ``GaussPyDecompose`` on it,
    capturing wall-clock time for the entire pipeline.

    Parameters
    ----------
    signal : npt.NDArray[np.float64]
        1-D spectrum.
    x_values : npt.NDArray[np.float64]
        Channel index array.
    errors : npt.NDArray[np.float64]
        Per-channel error array.
    idx : int
        Spectrum index (used for work directory isolation).

    Returns
    -------
    tuple[dict, float]
        ``(batch_result_dict, elapsed_seconds)``
    """
    work = os.path.join(WORK_DIR, f"s{idx}")
    os.makedirs(work, exist_ok=True)

    input_data = {
        "data_list": [signal.astype(np.float64)],
        "x_values": x_values,
        "error": [errors.copy()],
        "index": [0],
        "location": [(0, 0)],
    }

    pickle_path = os.path.join(work, "spectra.pickle")
    with open(pickle_path, "wb") as fobj:
        pickle.dump(input_data, fobj, protocol=2)

    from gausspyplus.decompose import GaussPyDecompose  # type: ignore[import-not-found]

    decomposer = GaussPyDecompose(path_to_pickle_file=pickle_path)
    decomposer.dirpath_gpy = work
    decomposer.alpha1 = ALPHA1
    decomposer.alpha2 = ALPHA2
    decomposer.two_phase_decomposition = True
    decomposer.snr_thresh = 3.0
    decomposer.snr2_thresh = 3.0
    decomposer.use_ncpus = 1
    decomposer.verbose = False
    decomposer.suffix = f"_s{idx}"

    t0 = time.perf_counter()
    decomposer.decompose()
    elapsed = time.perf_counter() - t0

    # Load result pickle
    result = _load_result(work)

    # Clean up to avoid filling /tmp
    shutil.rmtree(work, ignore_errors=True)

    return result, elapsed


def _load_result(work_dir: str) -> dict[str, Any]:
    """Load the decomposition result pickle from a work directory."""
    decomposed_dir = os.path.join(work_dir, "gpy_decomposed")
    if os.path.isdir(decomposed_dir):
        for fname in os.listdir(decomposed_dir):
            if fname.endswith(".pickle"):
                with open(os.path.join(decomposed_dir, fname), "rb") as fobj:
                    result: dict[str, Any] = pickle.load(fobj)  # noqa: S301
                return result

    return {
        "N_components": [0],
        "amplitudes_fit": [[]],
        "means_fit": [[]],
        "fwhms_fit": [[]],
    }


def main() -> None:
    """Run GaussPy+ on each spectrum individually and collect timing."""
    logger.info("Loading spectra...")
    data = np.load(SPECTRA_PATH)
    signals = data["signals"]
    n_spectra, n_channels = signals.shape
    logger.info("%d spectra, %d channels each", n_spectra, n_channels)

    x_values = np.arange(n_channels, dtype=np.float64)
    errors = np.full(n_channels, NOISE_SIGMA, dtype=np.float64)

    os.makedirs(WORK_DIR, exist_ok=True)

    n_components: list[int] = []
    amplitudes_fit: list[list[float]] = []
    means_fit: list[list[float]] = []
    stddevs_fit: list[list[float]] = []
    times: list[float] = []

    t_total_start = time.perf_counter()

    for i in range(n_spectra):
        result, elapsed = _decompose_one(signals[i], x_values, errors, i)
        times.append(round(elapsed, 6))

        n_comp = int(result["N_components"][0])
        n_components.append(n_comp)

        amps = result["amplitudes_fit"][0]
        means = result["means_fit"][0]
        fwhms = result["fwhms_fit"][0]

        if amps is not None and len(amps) > 0:
            amplitudes_fit.append([round(float(a), 6) for a in amps])
            means_fit.append([round(float(m), 4) for m in means])
            stddevs_fit.append([round(float(f) * FWHM_TO_SIGMA, 4) for f in fwhms])
        else:
            amplitudes_fit.append([])
            means_fit.append([])
            stddevs_fit.append([])

        if (i + 1) % 100 == 0:
            elapsed_total = time.perf_counter() - t_total_start
            mean_ms = elapsed_total / (i + 1) * 1000
            eta_s = mean_ms * (n_spectra - i - 1) / 1000
            logger.info(
                "  %d/%d (%.1f ms/spectrum, ETA %.0fs)",
                i + 1,
                n_spectra,
                mean_ms,
                eta_s,
            )

    t_total = time.perf_counter() - t_total_start

    mean_n = float(np.mean(n_components))
    times_arr = np.array(times)

    output: dict[str, object] = {
        "tool": "gausspyplus",
        "alpha1": ALPHA1,
        "alpha2": ALPHA2,
        "phase": "two",
        "n_spectra": n_spectra,
        "total_time_s": round(t_total, 3),
        "mean_time_per_spectrum_s": round(t_total / n_spectra, 6),
        "median_time_per_spectrum_s": round(float(np.median(times_arr)), 6),
        "p95_time_per_spectrum_s": round(float(np.percentile(times_arr, 95)), 6),
        "max_time_per_spectrum_s": round(float(np.max(times_arr)), 6),
        "mean_n_components": round(mean_n, 2),
        "n_components": n_components,
        "amplitudes_fit": amplitudes_fit,
        "means_fit": means_fit,
        "stddevs_fit": stddevs_fit,
        "times": times,
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
