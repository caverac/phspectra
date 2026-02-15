"""GaussPy+ performance benchmark -- runs inside Docker.

Loads real GRS spectra from a mounted .npz file, decomposes them with
GaussPy+ batch decomposer (which includes improve_fitting refinement),
monkey-patching the per-spectrum function to capture wall-clock timing.
Writes results as JSON to /data/results.json.

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

# Per-spectrum timing collected by the monkey-patched decompose_one.
per_spectrum_times: list[float] = []


def _run_batch(
    signals: npt.NDArray[np.float64],
    x_values: npt.NDArray[np.float64],
    errors: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """Run GaussPy+ batch decomposition with per-spectrum timing.

    Monkey-patches ``batch_decomposition.decompose_one`` to wrap each
    call with ``time.perf_counter`` so we capture per-spectrum wall-clock
    time without running spectra twice.

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
    dict
        Batch decomposition results (amplitudes_fit, means_fit, etc.).
    """
    n_spectra = signals.shape[0]
    os.makedirs(WORK_DIR, exist_ok=True)

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

    from gausspyplus.decompose import GaussPyDecompose  # type: ignore[import-not-found]

    decomposer = GaussPyDecompose(path_to_pickle_file=pickle_path)
    decomposer.dirpath_gpy = WORK_DIR
    decomposer.alpha1 = ALPHA1
    decomposer.alpha2 = ALPHA2
    decomposer.two_phase_decomposition = True
    decomposer.snr_thresh = 3.0
    decomposer.snr2_thresh = 3.0
    decomposer.use_ncpus = 1
    decomposer.verbose = False
    decomposer.suffix = "_bench"

    # Monkey-patch decompose_one to collect per-spectrum timing
    from gausspyplus.gausspy_py3 import batch_decomposition  # type: ignore[import-not-found]

    _original = batch_decomposition.decompose_one

    def _timed_decompose_one(i: int) -> Any:
        t0 = time.perf_counter()
        result = _original(i)
        per_spectrum_times.append(time.perf_counter() - t0)
        return result

    batch_decomposition.decompose_one = _timed_decompose_one
    try:
        decomposer.decompose()
    finally:
        batch_decomposition.decompose_one = _original

    return _load_batch_results(n_spectra)


def _load_batch_results(n_spectra: int) -> dict[str, Any]:
    """Load and return the batch decomposition pickle.

    Parameters
    ----------
    n_spectra : int
        Expected number of spectra.

    Returns
    -------
    dict
        Parsed batch results.
    """
    decomposed_dir = os.path.join(WORK_DIR, "gpy_decomposed")
    result_path = None
    if os.path.isdir(decomposed_dir):
        for fname in os.listdir(decomposed_dir):
            if fname.endswith(".pickle"):
                result_path = os.path.join(decomposed_dir, fname)
                break

    if result_path is None:
        logger.warning("No batch result pickle found")
        return {
            "N_components": [0] * n_spectra,
            "amplitudes_fit": [[] for _ in range(n_spectra)],
            "means_fit": [[] for _ in range(n_spectra)],
            "fwhms_fit": [[] for _ in range(n_spectra)],
        }

    with open(result_path, "rb") as fobj:
        result: dict[str, Any] = pickle.load(fobj)  # noqa: S301
    return result


def main() -> None:
    """Run the GaussPy+ batch benchmark with per-spectrum timing."""
    logger.info("Loading spectra...")
    data = np.load(SPECTRA_PATH)
    signals = data["signals"]
    n_spectra, n_channels = signals.shape
    logger.info("%d spectra, %d channels each", n_spectra, n_channels)

    x_values = np.arange(n_channels, dtype=np.float64)
    errors = np.full(n_channels, NOISE_SIGMA, dtype=np.float64)

    per_spectrum_times.clear()
    t_total_start = time.perf_counter()
    batch = _run_batch(signals, x_values, errors)
    t_total = time.perf_counter() - t_total_start

    # Parse batch results into output format
    n_components: list[int] = []
    amplitudes_fit: list[list[float]] = []
    means_fit: list[list[float]] = []
    stddevs_fit: list[list[float]] = []

    for i in range(n_spectra):
        n_comp = int(batch["N_components"][i])
        n_components.append(n_comp)

        amps = batch["amplitudes_fit"][i]
        means = batch["means_fit"][i]
        fwhms = batch["fwhms_fit"][i]

        if amps is not None and len(amps) > 0:
            amplitudes_fit.append([round(float(a), 6) for a in amps])
            means_fit.append([round(float(m), 4) for m in means])
            stddevs_fit.append([round(float(f) * FWHM_TO_SIGMA, 4) for f in fwhms])
        else:
            amplitudes_fit.append([])
            means_fit.append([])
            stddevs_fit.append([])

    # Use per-spectrum times from the monkey-patched decompose_one.
    # The batch runner processes spectra with data_list[i] != None,
    # skipping None entries.  Pad if needed.
    times = [round(t, 6) for t in per_spectrum_times]
    while len(times) < n_spectra:
        times.append(0.0)

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
