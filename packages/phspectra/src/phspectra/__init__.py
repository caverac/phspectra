"""phspectra -- persistent homology spectral line decomposition."""

__version__ = "1.0.1"

from phspectra._types import GaussianComponent
from phspectra.decompose import fit_gaussians
from phspectra.noise import estimate_rms
from phspectra.persistence import find_peaks_by_persistence

__all__ = [
    "GaussianComponent",
    "estimate_rms",
    "find_peaks_by_persistence",
    "fit_gaussians",
]
