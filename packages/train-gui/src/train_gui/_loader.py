"""Load comparison data produced by ``benchmarks compare``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from astropy.io import fits


@dataclass
class PixelData:
    """All data for a single pixel."""

    pixel: tuple[int, int]
    signal: npt.NDArray[np.float64]
    ph_components: list[dict[str, Any]]
    gp_components: list[dict[str, Any]]


@dataclass
class ComparisonData:
    """All loaded comparison data, indexed by pixel order."""

    pixels: list[tuple[int, int]]
    pixel_data: list[PixelData] = field(default_factory=list)

    def __len__(self) -> int:  # noqa: D105
        return len(self.pixel_data)


def load_comparison_data(data_dir: str | Path) -> ComparisonData:
    """Load FITS cube, phspectra results, and GaussPy+ results.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing ``phspectra_results.json``,
        ``results.json``, and ``spectra.npz`` from a
        ``benchmarks compare`` run.

    Returns
    -------
    ComparisonData
        Structured object with all pixel data.
    """
    data_dir = Path(data_dir)

    # -- phspectra results ---------------------------------------------------
    with open(data_dir / "phspectra_results.json", encoding="utf-8") as fh:
        ph = json.load(fh)

    pixels: list[tuple[int, int]] = [tuple(p) for p in ph["pixels"]]
    ph_amps: list[list[float]] = ph["amplitudes_fit"]
    ph_means: list[list[float]] = ph["means_fit"]
    ph_stds: list[list[float]] = ph["stddevs_fit"]

    # -- GaussPy+ results ----------------------------------------------------
    with open(data_dir / "results.json", encoding="utf-8") as fh:
        gp = json.load(fh)

    gp_amps: list[list[float]] = gp["amplitudes_fit"]
    gp_means: list[list[float]] = gp["means_fit"]
    gp_stds: list[list[float]] = gp["stddevs_fit"]

    # -- Raw spectra ----------------------------------------------------------
    signals: npt.NDArray[np.float64]
    npz_path = data_dir / "spectra.npz"
    if npz_path.exists():
        signals = np.load(npz_path)["signals"]
    else:
        # Fall back to FITS cube extraction if npz is missing.
        fits_path = _find_fits(data_dir)
        with fits.open(fits_path) as hdul:
            cube = np.array(hdul[0].data, dtype=np.float64)  # pylint: disable=no-member
        cube = np.nan_to_num(cube, nan=0.0)
        signals = np.array(
            [cube[:, py, px] for px, py in pixels],
            dtype=np.float64,
        )

    # -- Assemble per-pixel data ---------------------------------------------
    n = len(pixels)
    pixel_data: list[PixelData] = []
    for i in range(n):
        ph_comps = [
            {"amplitude": a, "mean": m, "stddev": s, "source": "phspectra"}
            for a, m, s in zip(ph_amps[i], ph_means[i], ph_stds[i])
        ]
        gp_comps = [
            {"amplitude": a, "mean": m, "stddev": s, "source": "gausspyplus"}
            for a, m, s in zip(gp_amps[i], gp_means[i], gp_stds[i])
        ]
        pixel_data.append(
            PixelData(
                pixel=pixels[i],
                signal=signals[i],
                ph_components=ph_comps,
                gp_components=gp_comps,
            )
        )

    return ComparisonData(pixels=pixels, pixel_data=pixel_data)


def _find_fits(data_dir: Path) -> Path:
    """Locate the FITS cube, checking common locations."""
    candidates = [
        data_dir / "grs-test-field.fits",
        data_dir.parent / "grs-test-field.fits",
        Path("/tmp/phspectra/grs-test-field.fits"),
    ]
    for p in candidates:
        if p.exists():
            return p
    msg = f"Cannot find FITS cube near {data_dir}"
    raise FileNotFoundError(msg)
