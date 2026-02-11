"""phspectra -- persistent homology spectral line decomposition."""

__version__ = "0.1.0"

from phspectra.decompose import DEFAULT_BETA, estimate_rms, fit_gaussians
from phspectra.persistence import find_peaks_by_persistence

__all__ = ["DEFAULT_BETA", "estimate_rms", "find_peaks_by_persistence", "fit_gaussians"]
