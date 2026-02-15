"""Shared constants for benchmark scripts."""

from __future__ import annotations

import os
from pathlib import Path

# Remote data sources

GAUSSPY_FITS_URL = "https://github.com/mriener/gausspyplus/raw/master/" + "gausspyplus/data/grs-test_field.fits"
TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"

# S3 public resources

RESOURCES_BUCKET_TEMPLATE = "phspectra-{environment}-resources"
RESOURCES_BASE_URL_TEMPLATE = "https://phspectra-{environment}-resources.s3.amazonaws.com"
RESOURCE_FITS = "grs-test-field.fits"
RESOURCE_CATALOG = "gausspy-catalog.votable"
RESOURCE_PRECOMPUTE_DB = "pre-compute.db"

# Local cache paths

CACHE_DIR = "/tmp/phspectra"
FITS_CACHE = os.path.join(CACHE_DIR, "grs-test-field.fits")
CATALOG_CACHE = os.path.join(CACHE_DIR, "gausspy-catalog.votable")

# Documentation site paths

_DOCS_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "docs"
DOCS_DIR = str(_DOCS_ROOT / "docs")
DOCS_IMG_DIR = str(_DOCS_ROOT / "static" / "img" / "results")

# Docker settings

DOCKER_IMAGE = "phspectra-gausspyplus-bench"

# Default decomposition parameters

DEFAULT_BETA = 3.8
DEFAULT_SEED = 2026_02_12
DEFAULT_N_SPECTRA = 400
DEFAULT_MAX_COMPONENTS = 10

# Synthetic benchmark parameters

N_CHANNELS = 424
NOISE_SIGMA = 0.25  # K -- GRS-realistic
MEAN_MARGIN = 10  # keep means within [MEAN_MARGIN, N_CHANNELS - MEAN_MARGIN]

# F1 matching criteria (Lindner et al. 2015, Eq. 7)

AMP_RATIO_BOUNDS = (0.0, 10.0)
POS_TOLERANCE_SIGMA = 1.0
WIDTH_RATIO_BOUNDS = (0.3, 2.5)
