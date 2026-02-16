"""Type stubs for the _gaussfit C extension."""

import numpy as np
from numpy.typing import NDArray

def bounded_lm_fit(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    p0: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    maxfev: int,
) -> NDArray[np.float64]:
    """Fit a sum of Gaussians using bounded Levenberg-Marquardt.

    Parameters
    ----------
    x : NDArray[np.float64]
        Independent variable (channel indices).
    y : NDArray[np.float64]
        Observed signal values.
    p0 : NDArray[np.float64]
        Initial parameter guess (amplitude, mean, stddev per component).
    lower : NDArray[np.float64]
        Lower bounds for each parameter.
    upper : NDArray[np.float64]
        Upper bounds for each parameter.
    maxfev : int
        Maximum number of function evaluations.

    Returns
    -------
    NDArray[np.float64]
        Optimised parameters.

    Raises
    ------
    RuntimeError
        If *maxfev* is reached without convergence.
    ValueError
        If the parameter vector exceeds the compiled limit.
    """

def find_peaks(
    signal: NDArray[np.float64],
    min_persistence: float,
) -> list[tuple[int, float, float, float, int]]:
    """Detect peaks via 0-dimensional persistent homology.

    Parameters
    ----------
    signal : NDArray[np.float64]
        1-D array of function values.
    min_persistence : float
        Discard peaks whose persistence is below this threshold.

    Returns
    -------
    list[tuple[int, float, float, float, int]]
        Each tuple is ``(index, birth, death, persistence, saddle_index)``,
        sorted by persistence descending.
    """
