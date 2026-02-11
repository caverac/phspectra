"""phspectra -- persistent homology spectral line decomposition."""

__version__ = "0.1.0"

from phspectra._types import GaussianComponent
from phspectra.decompose import DEFAULT_BETA, fit_gaussians
from phspectra.noise import estimate_rms
from phspectra.persistence import find_peaks_by_persistence

__all__ = [
    "DEFAULT_BETA",
    "GaussianComponent",
    "estimate_rms",
    "find_peaks_by_persistence",
    "fit_gaussians",
]
