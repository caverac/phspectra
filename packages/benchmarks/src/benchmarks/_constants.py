"""Shared constants for benchmark scripts."""

from __future__ import annotations

import os

# ── Remote data sources ──────────────────────────────────────────────────────

GAUSSPY_FITS_URL = (
    "https://github.com/mriener/gausspyplus/raw/master/" + "gausspyplus/data/grs-test_field.fits"
)
TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"

# ── Local cache paths ────────────────────────────────────────────────────────

CACHE_DIR = "/tmp/phspectra"
FITS_CACHE = os.path.join(CACHE_DIR, "grs-test-field.fits")
CATALOG_CACHE = os.path.join(CACHE_DIR, "gausspy-catalog.votable")

# ── Docker settings ──────────────────────────────────────────────────────────

DOCKER_IMAGE = "phspectra-gausspyplus-bench"

# ── Default decomposition parameters ────────────────────────────────────────

DEFAULT_BETA = 4.0
DEFAULT_SEED = 2025_02_12
DEFAULT_N_SPECTRA = 400
DEFAULT_MAX_COMPONENTS = 10

# ── Synthetic benchmark parameters ───────────────────────────────────────────

N_CHANNELS = 424
NOISE_SIGMA = 0.13  # K — GRS-realistic
MEAN_MARGIN = 10  # keep means within [MEAN_MARGIN, N_CHANNELS - MEAN_MARGIN]

# ── F1 matching criteria (Lindner et al. 2015, Eq. 7) ───────────────────────

AMP_RATIO_BOUNDS = (0.0, 10.0)
POS_TOLERANCE_SIGMA = 1.0
WIDTH_RATIO_BOUNDS = (0.3, 2.5)
