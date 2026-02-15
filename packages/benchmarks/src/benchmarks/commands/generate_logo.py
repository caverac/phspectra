"""``benchmarks generate-logo`` -- regenerate logo.svg and favicon.svg.

Builds a noisy three-peak spectrum, overlays persistence-homology
annotations (water level, emerged-peak markers, persistence bars),
and writes both SVG files to the docs static image directory.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
from benchmarks._console import console
from benchmarks._constants import DOCS_IMG_DIR

# ---------------------------------------------------------------------------
# SVG layout
# ---------------------------------------------------------------------------
_VIEWBOX = 512
_CIRCLE_R = 240
_CX, _CY = 256, 256
_X_LO, _X_HI = 80.0, 432.0
_Y_BASELINE = 400.0
_Y_CEIL = 80.0
_PLOT_W = _X_HI - _X_LO  # 352
_PLOT_H = _Y_BASELINE - _Y_CEIL  # 320

# ---------------------------------------------------------------------------
# Signal defaults
# ---------------------------------------------------------------------------
_N_PTS = 350
_NOISE_SIGMA = 0.045
_SEED = 42
_WATER_SIGNAL = 0.42  # normalised threshold between peaks B and A


def _build_signal(
    rng: np.random.Generator,
    noise_sigma: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(t, signal_noisy)`` in [0, 1] normalised units.

    Three Gaussians (amplitudes 1.0, 0.76, 0.35) plus additive noise.
    """
    t = np.linspace(0, 1, _N_PTS, dtype=np.float64)
    clean = (
        0.76 * np.exp(-0.5 * ((t - 0.355) / 0.035) ** 2)
        + 0.35 * np.exp(-0.5 * ((t - 0.57) / 0.025) ** 2)
        + 1.00 * np.exp(-0.5 * ((t - 0.75) / 0.032) ** 2)
    )
    noisy: npt.NDArray[np.float64] = np.clip(
        clean + rng.normal(0, noise_sigma, _N_PTS),
        0,
        None,
    )
    return t, noisy


def _to_svg(
    t: npt.NDArray[np.float64],
    signal: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Map normalised ``(t, signal)`` to SVG pixel coordinates."""
    x = _X_LO + t * _PLOT_W
    y = _Y_BASELINE - signal * _PLOT_H
    return x, y


def _path_d(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    *,
    closed: bool = False,
) -> str:
    """Build an SVG path ``d`` attribute from coordinate arrays."""
    parts = [f"M{x[0]:.1f},{y[0]:.1f}"]
    for i in range(1, len(x)):
        parts.append(f"L{x[i]:.1f},{y[i]:.1f}")
    if closed:
        parts.append("Z")
    return " ".join(parts)


def _find_peak(noisy: npt.NDArray[np.float64], lo_frac: float, hi_frac: float) -> int:
    lo_i = int(lo_frac * _N_PTS)
    hi_i = int(hi_frac * _N_PTS)
    return lo_i + int(np.argmax(noisy[lo_i:hi_i]))


def _render_svg(
    noisy: npt.NDArray[np.float64],
    x_svg: npt.NDArray[np.float64],
    y_svg: npt.NDArray[np.float64],
) -> str:
    """Return the complete SVG document as a string."""
    # Fill path (closed polygon under the curve)
    fill_x = np.concatenate([[x_svg[0]], x_svg, [x_svg[-1]]])
    fill_y = np.concatenate([[_Y_BASELINE], y_svg, [_Y_BASELINE]])
    fill_d = _path_d(fill_x, fill_y, closed=True)

    stroke_d = _path_d(x_svg, y_svg)

    water_y = _Y_BASELINE - _WATER_SIGNAL * _PLOT_H

    # Emerged peaks (A and C above water; B submerged)
    peak_a = _find_peak(noisy, 0.28, 0.43)
    peak_c = _find_peak(noisy, 0.68, 0.83)
    pa_x, pa_y = float(x_svg[peak_a]), float(y_svg[peak_a])
    pc_x, pc_y = float(x_svg[peak_c]), float(y_svg[peak_c])

    return f"""\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_VIEWBOX} {_VIEWBOX}">
  <defs>
    <linearGradient id="specFill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#6C5CE7" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="#6C5CE7" stop-opacity="0.02"/>
    </linearGradient>
    <radialGradient id="dot">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </radialGradient>
    <clipPath id="circleClip">
      <circle cx="{_CX}" cy="{_CY}" r="{_CIRCLE_R}"/>
    </clipPath>
  </defs>

  <!-- Background circle -->
  <circle cx="{_CX}" cy="{_CY}" r="{_CIRCLE_R}" fill="#0F0E17" stroke="#2D2B55" stroke-width="4"/>

  <!-- Diffuse glowing dots -->
  <circle cx="120" cy="120" r="6" fill="url(#dot)"/>
  <circle cx="400" cy="100" r="5" fill="url(#dot)"/>
  <circle cx="160" cy="85" r="4" fill="url(#dot)"/>
  <circle cx="380" cy="150" r="4.5" fill="url(#dot)"/>
  <circle cx="280" cy="78" r="3.5" fill="url(#dot)"/>
  <circle cx="440" cy="200" r="4" fill="url(#dot)"/>
  <circle cx="100" cy="180" r="3" fill="url(#dot)"/>
  <circle cx="320" cy="95" r="4.5" fill="url(#dot)"/>
  <circle cx="200" cy="100" r="3" fill="url(#dot)"/>
  <circle cx="450" cy="280" r="3.5" fill="url(#dot)"/>
  <circle cx="90" cy="300" r="4" fill="url(#dot)"/>

  <!-- Subtle axis lines -->
  <line x1="{_X_LO:.0f}" y1="{_Y_BASELINE:.0f}" x2="{_X_HI:.0f}" y2="{_Y_BASELINE:.0f}"
        stroke="#2D2B55" stroke-width="1.5"/>
  <line x1="{_X_LO:.0f}" y1="{_Y_BASELINE:.0f}" x2="{_X_LO:.0f}" y2="{_Y_CEIL:.0f}"
        stroke="#2D2B55" stroke-width="1.5"/>

  <!-- Water fill above threshold (clipped to circle) -->
  <g clip-path="url(#circleClip)">
    <rect x="0" y="{_Y_CEIL:.0f}" width="{_VIEWBOX}" height="{water_y - _Y_CEIL:.1f}"
          fill="#4A90D9" opacity="0.08"/>
  </g>

  <!-- Water level line -->
  <line x1="{_X_LO:.0f}" y1="{water_y:.1f}" x2="{_X_HI:.0f}" y2="{water_y:.1f}"
        stroke="#4A90D9" stroke-width="1.2" stroke-dasharray="8,5" opacity="0.45"/>

  <!-- Spectrum fill -->
  <path d="{fill_d}" fill="url(#specFill)" stroke="none"/>

  <!-- Spectrum stroke (noisy polyline) -->
  <path d="{stroke_d}" fill="none" stroke="#A29BFE" stroke-width="2.5"
        stroke-linecap="round" stroke-linejoin="round"/>

  <!-- Persistence bars (peak to water level) -->
  <line x1="{pc_x:.1f}" y1="{pc_y:.1f}" x2="{pc_x:.1f}" y2="{water_y:.1f}"
        stroke="#FF6F61" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.7"/>
  <line x1="{pa_x:.1f}" y1="{pa_y:.1f}" x2="{pa_x:.1f}" y2="{water_y:.1f}"
        stroke="#FF6F61" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.7"/>

  <!-- Peak markers -->
  <circle cx="{pc_x:.1f}" cy="{pc_y:.1f}" r="5" fill="#FF6F61" opacity="0.9"/>
  <circle cx="{pa_x:.1f}" cy="{pa_y:.1f}" r="5" fill="#FF6F61" opacity="0.9"/>
</svg>
"""


def _write_if_changed(path: Path, content: str) -> bool:
    """Write *content* to *path* only if it differs.  Return True if written."""
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


@click.command("generate-logo")
@click.option("--seed", default=_SEED, show_default=True)
@click.option("--noise-sigma", default=_NOISE_SIGMA, show_default=True)
def generate_logo(seed: int, noise_sigma: float) -> None:
    """Regenerate logo.svg and favicon.svg with noisy spectrum + persistence."""
    rng = np.random.default_rng(seed)
    t, noisy = _build_signal(rng, noise_sigma)
    x_svg, y_svg = _to_svg(t, noisy)
    svg = _render_svg(noisy, x_svg, y_svg)

    img_dir = Path(DOCS_IMG_DIR).parent  # DOCS_IMG_DIR is static/img/results
    for name in ("logo.svg", "favicon.svg"):
        path = img_dir / name
        if _write_if_changed(path, svg):
            console.print(f"  Saved [blue]{path}[/blue]")
        else:
            console.print(f"  Unchanged [dim]{path}[/dim]")

    console.print("\nDone.", style="bold green")
