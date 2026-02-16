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
) -> NDArray[np.float64]: ...

def find_peaks(
    signal: NDArray[np.float64],
    min_persistence: float,
) -> list[tuple[int, float, float, float, int]]: ...
