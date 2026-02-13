"""Shared dataclasses and type aliases for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from astropy.io.fits import Header


class FitsPrimaryHDU(Protocol):
    """Structural type for an astropy PrimaryHDU.

    Astropy's own type stubs are incomplete, so Pylance and Pylint
    cannot resolve ``.header`` and ``.data`` on the real class.  This
    protocol captures the two attributes we actually use.
    """

    header: Header
    data: npt.NDArray[np.float64]


class _AngleAttr(Protocol):
    """An astropy angle-like attribute with a ``.deg`` property."""

    @property
    def deg(self) -> npt.NDArray[np.float64]: ...


class GalacticFrame(Protocol):
    """Structural type for an astropy Galactic coordinate frame."""

    @property
    def l(self) -> _AngleAttr: ...  # noqa: E741

    @property
    def b(self) -> _AngleAttr: ...


class ICRSFrame(Protocol):
    """Structural type for an astropy ICRS coordinate frame."""

    @property
    def ra(self) -> _AngleAttr: ...

    @property
    def dec(self) -> _AngleAttr: ...


@dataclass
class Component:
    """Gaussian component in channel space."""

    amplitude: float
    mean: float
    stddev: float


@dataclass
class ComparisonResult:
    """One spectrum with both decompositions + timing."""

    pixel: tuple[int, int]
    signal: npt.NDArray[np.float64]
    gp_comps: list[Component]
    ph_comps: list[Component]
    ph_rms: float
    gp_rms: float
    ph_time: float
    gp_time: float


@dataclass
class BetaSweepResult:
    """Result of a single beta value in the sweep."""

    beta: float
    f1: float
    precision: float
    recall: float
    n_correct: int
    n_true: int
    n_guessed: int
    time_s: float
    mean_ph_rms: float
    mean_gp_rms: float
    n_ph_wins: int


@dataclass
class SyntheticSpectrum:
    """A synthetic spectrum with known ground-truth components."""

    category: str
    index: int
    signal: npt.NDArray[np.float64]
    components: list[Component]
