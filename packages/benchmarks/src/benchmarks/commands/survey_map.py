"""``benchmarks survey-map`` — 2x2 survey visualisation from full-field decomposition.

Panel layout
------------
+-------------------------------+-------------------------------+
| (a) Velocity RGB composite    | (b) Topological complexity    |
|                               |                               |
| Three velocity bins mapped    | Number of Gaussian components |
| to R, G, B from the decomp-  | detected per pixel.  Lights   |
| osed (not raw) Gaussians.     | up at cloud boundaries,       |
| Sharper than moment-0 RGB     | outflows, and shock fronts —  |
| because noise is removed by   | a quantity unique to persist-  |
| the fit.                      | ent-homology decomposition.   |
+-------------------------------+-------------------------------+
| (c) Dominant velocity field   | (d) Amplitude–velocity        |
|                               |     bivariate colormap         |
| Centroid velocity of the      |                               |
| brightest component per       | 2-D perceptual colormap where |
| pixel.  Reveals bulk gas      | hue encodes centroid velocity  |
| motions hidden by moment-1    | and luminance encodes peak     |
| blending when multiple clouds | amplitude.  Every pixel        |
| overlap along the LOS.        | communicates two physical      |
|                               | quantities simultaneously.     |
+-------------------------------+-------------------------------+
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from benchmarks._console import console
from benchmarks._plotting import docs_figure


def _configure_axes(ax: Axes) -> None:
    """Apply the shared tick/grid style."""
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="minor", length=3, color="gray", direction="in")
    ax.tick_params(which="major", length=6, direction="in")
    ax.tick_params(top=True, right=True, which="both")


@docs_figure("survey-map.png")
def _build_figure() -> Figure:
    """Construct the 2x2 placeholder figure."""
    fig: Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(
        left=0.08, right=0.96, bottom=0.06, top=0.94, wspace=0.20, hspace=0.22,
    )

    labels = [
        "(a) Velocity RGB composite",
        "(b) Topological complexity",
        "(c) Dominant velocity field",
        r"(d) Amplitude–velocity bivariate",
    ]

    for ax, label in zip(axes.ravel(), labels):
        ax.text(
            0.5, 0.5, "placeholder",
            transform=ax.transAxes, ha="center", va="center",
            color="0.6",
        )
        ax.set_title(label)
        ax.set_xlabel("Galactic longitude (px)")
        ax.set_ylabel("Galactic latitude (px)")
        _configure_axes(ax)

    return fig


@click.command("survey-map")
def survey_map() -> None:
    """Generate 2x2 survey visualisation from full-field decomposition."""
    console.print("Generating survey map placeholder ...", style="bold cyan")
    _build_figure()
    console.print("\nDone.", style="bold green")
