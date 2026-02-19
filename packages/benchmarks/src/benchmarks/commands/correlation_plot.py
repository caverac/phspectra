"""``benchmarks correlation-plot`` -- two-point autocorrelation of decomposition fields.

Computes the angular autocorrelation function of four scalar fields
derived from the Gaussian decomposition of a single GRS tile:

1. **N_comp** -- number of components per pixel (topological complexity).
2. **Integrated intensity** -- sum(amp_i * sigma_i) per pixel.
3. **Intensity-weighted mean velocity** -- first moment of fitted components.
4. **Velocity dispersion** -- intensity-weighted second central moment
   of component centroids.

Uses FFT-based estimation on the regular pixel grid with zero-padding
and proper mask normalisation.
"""

from __future__ import annotations

import glob
import os

import click
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from benchmarks._console import console
from benchmarks._constants import CACHE_DIR
from benchmarks._plotting import configure_axes, docs_figure
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import LogFormatterMathtext

SURVEY_DEFAULT = "grs-26"
MAX_LAG_ARCMIN_DEFAULT = 15.0

# ---------------------------------------------------------------------------
# Field labels for the four panels
# ---------------------------------------------------------------------------

FIELD_SPECS: list[tuple[str, str, str]] = [
    ("ncomp", r"$\xi_{N_{\mathrm{comp}}}(\theta)$", r"$N_{\mathrm{components}}$"),
    ("intensity", r"$\xi_{I_{\mathrm{tot}}}(\theta)$", r"$I_{\mathrm{tot}}$"),
    ("velocity", r"$\xi_{\bar{v}}(\theta)$", r"$\bar{v}$"),
    ("dispersion", r"$\xi_{\sigma_v}(\theta)$", r"$\sigma_v$"),
]


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------


def _load_decomposition_table(survey: str, cache_dir: str) -> pa.Table:
    """Load cached Parquet decomposition data for *survey*."""
    parquet_dir = os.path.join(cache_dir, "decompositions", survey)
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        msg = f"No Parquet files in {parquet_dir}"
        raise FileNotFoundError(msg)
    return pa.concat_tables([pq.read_table(f) for f in files])


def _build_scalar_grids(
    table: pa.Table,
) -> dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]]:
    """Build the four scalar fields and their masks from a decomposition table.

    Returns
    -------
    dict mapping field key -> (grid, mask), where *grid* has shape
    ``(ny, nx)`` and *mask* is True for valid pixels.
    """
    x = table.column("x").to_numpy().astype(np.int32)
    y = table.column("y").to_numpy().astype(np.int32)
    nc = table.column("n_components").to_numpy().astype(np.int32)
    amps = table.column("component_amplitudes").to_pylist()
    means = table.column("component_means").to_pylist()
    stddevs = table.column("component_stddevs").to_pylist()

    nx, ny = int(x.max()) + 1, int(y.max()) + 1
    sentinel = np.nan

    ncomp_grid = np.full((ny, nx), sentinel)
    intensity_grid = np.full((ny, nx), sentinel)
    velocity_grid = np.full((ny, nx), sentinel)
    dispersion_grid = np.full((ny, nx), sentinel)

    for i, _ in enumerate(x):
        xi, yi = int(x[i]), int(y[i])
        ncomp_grid[yi, xi] = nc[i]

        a = np.asarray(amps[i], dtype=np.float64)
        m = np.asarray(means[i], dtype=np.float64)
        s = np.asarray(stddevs[i], dtype=np.float64)

        if len(a) == 0 or nc[i] <= 0:
            continue

        total_intensity = np.sum(a * s)
        intensity_grid[yi, xi] = total_intensity

        w_total = np.sum(a)
        if w_total > 0:
            v_bar = np.sum(a * m) / w_total
            velocity_grid[yi, xi] = v_bar
            dispersion_grid[yi, xi] = np.sqrt(np.sum(a * (m - v_bar) ** 2) / w_total)

    grids: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]] = {}
    for key, grid in [
        ("ncomp", ncomp_grid),
        ("intensity", intensity_grid),
        ("velocity", velocity_grid),
        ("dispersion", dispersion_grid),
    ]:
        mask = np.isfinite(grid)
        grid_clean = np.where(mask, grid, 0.0)
        grids[key] = (grid_clean, mask)

    return grids


# ---------------------------------------------------------------------------
# FFT autocorrelation
# ---------------------------------------------------------------------------


def _autocorrelation_fft(
    field: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """FFT-based masked autocorrelation with proper normalisation.

    Parameters
    ----------
    field : (ny, nx)
        Scalar field (0 where masked).
    mask : (ny, nx), bool
        True for valid pixels.

    Returns
    -------
    radii : 1-D array
        Lag in pixels (integer bins).
    xi : 1-D array
        Normalised autocorrelation xi(r), where xi(0) = 1.
    """
    w = mask.astype(np.float64)
    w_sum = np.sum(w)
    if w_sum == 0:
        return np.array([0.0]), np.array([0.0])
    mean = np.sum(field * w) / w_sum
    delta = (field - mean) * w

    pad_ny, pad_nx = 2 * field.shape[0], 2 * field.shape[1]

    ft_delta = np.fft.rfft2(delta, s=(pad_ny, pad_nx))
    ft_w = np.fft.rfft2(w, s=(pad_ny, pad_nx))

    corr = np.fft.irfft2(ft_delta * np.conj(ft_delta), s=(pad_ny, pad_nx))
    counts = np.fft.irfft2(ft_w * np.conj(ft_w), s=(pad_ny, pad_nx))

    ny, nx = field.shape
    yy, xx = np.mgrid[:pad_ny, :pad_nx]
    yy = np.where(yy > pad_ny // 2, yy - pad_ny, yy)
    xx = np.where(xx > pad_nx // 2, xx - pad_nx, xx)
    r = np.sqrt(xx**2.0 + yy**2.0)

    max_lag = min(ny, nx) // 2
    r_int = np.round(r).astype(int)
    valid = (counts > 1.0) & (r_int <= max_lag)

    radii_out = np.arange(max_lag + 1, dtype=np.float64)
    xi_out = np.zeros(max_lag + 1)
    weight_out = np.zeros(max_lag + 1)

    np.add.at(xi_out, r_int[valid], corr[valid])
    np.add.at(weight_out, r_int[valid], counts[valid])

    nonzero = weight_out > 0
    xi_out[nonzero] /= weight_out[nonzero]

    if xi_out[0] > 0:
        xi_out /= xi_out[0]

    return radii_out, xi_out


def _jackknife_autocorrelation(
    field: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    n_blocks_y: int = 4,
    n_blocks_x: int = 4,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Spatial jackknife estimate of the autocorrelation and its 1-sigma error.

    Divides the grid into ``n_blocks_y x n_blocks_x`` rectangular blocks.
    Each jackknife sample masks out one block and recomputes the
    autocorrelation.  The variance is estimated with the standard
    delete-one jackknife formula.

    Parameters
    ----------
    field : (ny, nx)
        Scalar field (0 where masked).
    mask : (ny, nx), bool
        True for valid pixels.
    n_blocks_y, n_blocks_x:
        Block grid dimensions.

    Returns
    -------
    radii : 1-D array
        Lag in pixels.
    xi : 1-D array
        Full-sample autocorrelation.
    xi_err : 1-D array
        1-sigma jackknife uncertainty per lag bin.
    """
    radii, xi_full = _autocorrelation_fft(field, mask)
    n_lags = len(radii)

    ny, nx = field.shape
    y_edges = np.linspace(0, ny, n_blocks_y + 1, dtype=int)
    x_edges = np.linspace(0, nx, n_blocks_x + 1, dtype=int)

    n_blocks = n_blocks_y * n_blocks_x
    xi_samples = np.zeros((n_blocks, n_lags))

    idx = 0
    for iy in range(n_blocks_y):
        for ix in range(n_blocks_x):
            jk_mask = mask.copy()
            jk_mask[y_edges[iy] : y_edges[iy + 1], x_edges[ix] : x_edges[ix + 1]] = False
            jk_field = field * jk_mask
            _, xi_jk = _autocorrelation_fft(jk_field, jk_mask)
            # Pad or trim to match full-sample length
            n = min(n_lags, len(xi_jk))
            xi_samples[idx, :n] = xi_jk[:n]
            idx += 1

    # Delete-one jackknife variance: var = (N-1)/N * sum((xi_i - xi_mean)^2)
    xi_mean = xi_samples.mean(axis=0)
    xi_err = np.sqrt((n_blocks - 1) / n_blocks * np.sum((xi_samples - xi_mean) ** 2, axis=0))

    return radii, xi_full, xi_err


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


_CorrelationResult = tuple[
    npt.NDArray[np.float64],  # radii_px
    npt.NDArray[np.float64],  # xi
    npt.NDArray[np.float64],  # xi_err
]


@docs_figure("correlation-plot.png")
def _build_correlation_figure(
    results: dict[str, _CorrelationResult],
    arcmin_per_px: float,
    max_lag_arcmin: float,
) -> Figure:
    """Build the 2x2 autocorrelation figure with jackknife error bands.

    Parameters
    ----------
    results:
        Mapping of field key -> (radii_px, xi, xi_err).
    arcmin_per_px:
        Pixel scale in arcminutes.
    max_lag_arcmin:
        Maximum angular lag to display.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.09, top=0.97, wspace=0.25, hspace=0.08)

    fmt = LogFormatterMathtext()

    for idx, (key, _ylabel, label) in enumerate(FIELD_SPECS):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        radii_px, xi, xi_err = results[key]
        radii_arcmin = radii_px * arcmin_per_px

        keep = (radii_arcmin > 0) & (radii_arcmin <= max_lag_arcmin) & (xi > 0)
        r_plot = radii_arcmin[keep]
        xi_plot = xi[keep]
        err_plot = xi_err[keep]

        ax.loglog(r_plot, xi_plot, color="0.15", linewidth=1.2, label=label)
        lo = np.maximum(xi_plot - err_plot, xi_plot * 1e-4)
        hi = xi_plot + err_plot
        ax.fill_between(r_plot, lo, hi, color="0.15", alpha=0.15)

        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        configure_axes(ax)
        ax.legend(loc="upper right", frameon=False)

        # Correlation length annotation (1/e crossing)
        e_fold = 1.0 / np.e
        below = np.where(xi_plot < e_fold)[0]
        if len(below) > 0:
            r_corr = r_plot[below[0]]
            ax.axvline(r_corr, color="0.15", linewidth=0.8, linestyle=":")
            ax.text(
                r_corr * 1.15,
                0.85,
                f"$\\theta_{{corr}}$ = {r_corr:.1f}'",
                transform=ax.get_xaxis_transform(),
                fontsize=9,
                color="0.15",
            )

        ax.set_ylabel(r"$\xi(\theta)$")

    for ax in axes[1]:
        ax.set_xlabel("Angular separation (arcmin)")

    return fig


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@click.command("correlation-plot")
@click.option(
    "--survey",
    default=SURVEY_DEFAULT,
    show_default=True,
    help="Survey name (must have cached decompositions).",
)
@click.option(
    "--fits-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="FITS cube for pixel-scale header (CDELT2).",
)
@click.option(
    "--max-lag",
    default=MAX_LAG_ARCMIN_DEFAULT,
    show_default=True,
    type=float,
    help="Maximum angular lag in arcminutes.",
)
@click.option("--cache-dir", default=CACHE_DIR, show_default=True, help="Cache directory.")
def correlation_plot(
    survey: str,
    fits_file: str,
    max_lag: float,
    cache_dir: str,
) -> None:
    """Two-point autocorrelation of decomposition scalar fields."""
    console.print(f"Loading [bold]{survey}[/bold] decomposition data ...", style="bold cyan")
    table = _load_decomposition_table(survey, cache_dir)

    console.print("Building scalar grids ...", style="bold cyan")
    grids = _build_scalar_grids(table)

    with fits.open(fits_file) as hdul:
        cdelt = abs(float(hdul[0].header["CDELT2"]))  # pylint: disable=no-member
    arcmin_per_px = cdelt * 60.0
    console.print(f"  Pixel scale: {arcmin_per_px:.3f} arcmin/px", style="green")

    results: dict[str, _CorrelationResult] = {}
    for key, _, label in FIELD_SPECS:
        console.print(f"  Computing autocorrelation for {label} ...", style="cyan")
        field, mask = grids[key]
        results[key] = _jackknife_autocorrelation(field, mask)

    console.print("Building figure ...", style="bold cyan")
    _build_correlation_figure(results, arcmin_per_px, max_lag)
    console.print("\nDone.", style="bold green")
