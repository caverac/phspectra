"""FITS download/cache, catalog fetch, and pixel selection utilities."""

from __future__ import annotations

import io
import os
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from benchmarks._console import console
from benchmarks._constants import (
    CATALOG_CACHE,
    FITS_CACHE,
    GAUSSPY_FITS_URL,
    TAP_URL,
)


def ensure_fits(
    url: str = GAUSSPY_FITS_URL,
    path: str = FITS_CACHE,
) -> tuple[fits.Header, np.ndarray]:
    """Download the GRS test field FITS if not cached.

    Parameters
    ----------
    url : str
        Remote URL of the FITS file.
    path : str
        Local cache path.

    Returns
    -------
    tuple[fits.Header, np.ndarray]
        FITS header and data cube.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        console.print(f"  Using cached FITS: [blue]{path}[/blue]")
    else:
        console.print(f"  Downloading {url} ...")
        urlretrieve(url, path)  # noqa: S310
        console.print(f"  Saved to [blue]{path}[/blue]")
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
        data = hdul[0].data.copy()  # type: ignore[union-attr]
    return header, data


def fits_bounds(header: fits.Header) -> tuple[float, float, float, float]:
    """Return (glon_min, glon_max, glat_min, glat_max) from the FITS WCS.

    Parameters
    ----------
    header : fits.Header
        FITS header with WCS information.

    Returns
    -------
    tuple[float, float, float, float]
        Galactic coordinate bounds.
    """
    wcs_cel = WCS(header, naxis=2)
    ny, nx = header["NAXIS2"], header["NAXIS1"]
    x_corners = np.array([0, nx - 1, 0, nx - 1])
    y_corners = np.array([0, 0, ny - 1, ny - 1])
    lon, lat = wcs_cel.pixel_to_world_values(x_corners, y_corners)
    ctype1 = header.get("CTYPE1", "").upper()
    if "GLON" in ctype1:
        glon, glat = lon, lat
    else:
        sky = SkyCoord(ra=lon, dec=lat, unit="deg", frame="icrs")
        gal = sky.galactic
        glon, glat = gal.l.deg, gal.b.deg
    pad = 0.01
    return (
        float(np.min(glon)) - pad,
        float(np.max(glon)) + pad,
        float(np.min(glat)) - pad,
        float(np.max(glat)) + pad,
    )


def ensure_catalog(
    glon_min: float,
    glon_max: float,
    glat_min: float,
    glat_max: float,
    path: str = CATALOG_CACHE,
) -> Table:
    """Query VizieR TAP if not cached, return the astropy Table.

    Parameters
    ----------
    glon_min, glon_max, glat_min, glat_max : float
        Galactic coordinate bounds.
    path : str
        Local cache path.

    Returns
    -------
    Table
        Astropy table with catalog entries.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        console.print(f"  Using cached catalog: [blue]{path}[/blue]")
        return Table.read(path, format="votable")
    adql = (
        "SELECT TB, VLSR, sigma, Ncomp, rms, GLON, GLAT, Xpos, Ypos "
        'FROM "J/A+A/633/A14/table1" '
        f"WHERE GLON BETWEEN {glon_min:.4f} AND {glon_max:.4f} "
        f"AND GLAT BETWEEN {glat_min:.4f} AND {glat_max:.4f}"
    )
    url = f"{TAP_URL}?REQUEST=doQuery&LANG=ADQL&FORMAT=votable&QUERY={quote(adql)}"
    console.print("  Querying VizieR TAP ...")
    resp = urlopen(url, timeout=120)  # noqa: S310
    data = resp.read()
    with open(path, "wb") as f:
        f.write(data)
    console.print(f"  Saved to [blue]{path}[/blue]")
    table = Table.read(io.BytesIO(data), format="votable")
    console.print(f"  {len(table)} component rows")
    return table


def match_catalog_pixels(
    catalog: Table,
    header: fits.Header,
) -> dict[tuple[int, int], int]:
    """Return pixel -> n_components for eligible pixels.

    Parameters
    ----------
    catalog : Table
        VizieR catalog table.
    header : fits.Header
        FITS header with WCS information.

    Returns
    -------
    dict[tuple[int, int], int]
        Mapping from (x, y) pixel to component count.
    """
    wcs_cel = WCS(header, naxis=2)
    ny, nx = header["NAXIS2"], header["NAXIS1"]
    cat_glon = np.array(catalog["GLON"], dtype=np.float64)
    cat_glat = np.array(catalog["GLAT"], dtype=np.float64)
    ctype1 = header.get("CTYPE1", "").upper()
    if "GLON" in ctype1:
        cat_lon, cat_lat = cat_glon, cat_glat
    else:
        sky = SkyCoord(l=cat_glon, b=cat_glat, unit="deg", frame="galactic")
        icrs = sky.icrs
        cat_lon, cat_lat = icrs.ra.deg, icrs.dec.deg
    pix_x, pix_y = wcs_cel.world_to_pixel_values(cat_lon, cat_lat)
    pix_x = np.round(pix_x).astype(int)
    pix_y = np.round(pix_y).astype(int)
    counts: dict[tuple[int, int], int] = {}
    for i in range(len(catalog)):
        x, y = int(pix_x[i]), int(pix_y[i])
        if 0 <= x < nx and 0 <= y < ny:
            counts[(x, y)] = counts.get((x, y), 0) + 1
    return counts


def select_spectra(
    cube: np.ndarray,
    header: fits.Header,
    catalog: Table,
    n_spectra: int,
    seed: int,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Select random spectra that have catalog components.

    Parameters
    ----------
    cube : np.ndarray
        Data cube (n_channels, ny, nx).
    header : fits.Header
        FITS header.
    catalog : Table
        VizieR catalog.
    n_spectra : int
        Number of spectra to select.
    seed : int
        Random seed.

    Returns
    -------
    tuple[list[tuple[int, int]], np.ndarray]
        Selected pixel coordinates and signal array (n_select, n_channels).
    """
    rng = np.random.default_rng(seed)
    pixel_counts = match_catalog_pixels(catalog, header)
    eligible = {k: v for k, v in pixel_counts.items() if 1 <= v <= 8}
    console.print(f"  {len(eligible)} eligible pixels")

    eligible_keys = list(eligible.keys())
    n_select = min(n_spectra, len(eligible_keys))
    idx = rng.choice(len(eligible_keys), size=n_select, replace=False)
    selected = [eligible_keys[i] for i in idx]

    n_channels = cube.shape[0]
    signals = np.zeros((n_select, n_channels), dtype=np.float64)
    for i, (px, py) in enumerate(selected):
        signals[i] = np.nan_to_num(cube[:, py, px].astype(np.float64), nan=0.0)
    console.print(f"  Selected {n_select} spectra")
    return selected, signals
