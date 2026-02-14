"""Matplotlib-based interactive viewer for training set curation."""

from __future__ import annotations

import string
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
from train_gui._loader import ComparisonData
from train_gui._state import TrainingSet

# Visual constants.
_ALPHA_COMP = 0.6
_LW_NORMAL = 1.5
_LW_SELECTED = 3.5
_COLOR_PH = "#CF2A2A"
_COLOR_GP = "#415493"
_COLOR_SELECTED = "#1b6311"
_COLOR_SIGNAL = "#888888"
_COLOR_SIGNAL_ORIG = "#cccccc"
_COLOR_MAN = "#d4890e"

# Key labels: a–z, excluding matplotlib defaults and app reserved keys.
# Matplotlib: f(fullscreen), g(grid), h(home), k(xscale), l(yscale),
#             o(zoom), p(pan), q(quit), r(reset), s(save), v(forward)
# App: c(clear)
_RESERVED = set("cfghklopqrsv")
_KEYS = [k for k in string.ascii_lowercase if k not in _RESERVED]


def _gaussian(
    x: npt.NDArray[np.float64],
    amp: float,
    mean: float,
    stddev: float,
) -> npt.NDArray[np.float64]:
    return amp * np.exp(-0.5 * ((x - mean) / max(stddev, 1e-10)) ** 2)


class SpectrumViewer:
    """Interactive matplotlib figure for curating training components.

    Parameters
    ----------
    data : ComparisonData
        Loaded comparison data from ``benchmarks compare``.
    training_set : TrainingSet
        Mutable training set container.
    start_index : int
        Initial pixel index to display.
    """

    def __init__(  # noqa: D107
        self,
        data: ComparisonData,
        training_set: TrainingSet,
        start_index: int = 0,
    ) -> None:
        self._data = data
        self._ts = training_set
        self._index = max(0, min(start_index, len(data) - 1))

        # Keyed component list for the current pixel.
        self._keyed_comps: list[dict[str, Any]] = []

        # Build figure: spectrum axes + text panel below.
        self._fig: Figure
        self._fig, (self._ax, self._ax_list) = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        self._fig.canvas.manager.set_window_title("train-gui")  # type: ignore[union-attr]

        # Connect events.
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)  # type: ignore[arg-type]

        # Span selector for manual Gaussian fitting on the residual.
        self._span = SpanSelector(
            self._ax,
            self._on_span,
            "horizontal",
            useblit=True,
            props={"alpha": 0.25, "facecolor": _COLOR_MAN},
            interactive=False,
        )

        self._draw_pixel()

    def _draw_pixel(self) -> None:  # pylint: disable=too-many-branches
        ax = self._ax
        ax.clear()

        pd = self._data.pixel_data[self._index]
        pixel = pd.pixel
        signal = pd.signal
        x = np.arange(len(signal), dtype=np.float64)

        # Build keyed component list: GP+ first, then PHS, then manual.
        self._keyed_comps = list(pd.gp_components) + list(pd.ph_components)
        entry = self._ts.get_entry(pixel)
        if entry is not None:
            for c in entry["components"]:
                if c.get("source") == "manual":
                    self._keyed_comps.append(c)

        # Compute residual: subtract all *selected* components from signal.
        residual = signal.copy()
        for comp in self._keyed_comps:
            if self._ts.has_component(pixel, comp):
                residual = residual - _gaussian(x, comp["amplitude"], comp["mean"], comp["stddev"])

        # Original signal as thin dashed line for context.
        ax.step(
            x,
            signal,
            where="mid",
            color=_COLOR_SIGNAL_ORIG,
            linewidth=0.6,
            linestyle="--",
        )
        # Residual as the main step plot.
        ax.step(x, residual, where="mid", color=_COLOR_SIGNAL, linewidth=0.8)

        n_sel_ph = 0
        n_sel_gp = 0
        n_sel_man = 0

        # First pass: draw curves and collect label positions.
        label_info: list[tuple[float, float, str, str]] = []
        for i, comp in enumerate(self._keyed_comps):
            if i >= len(_KEYS):
                break
            selected = self._ts.has_component(pixel, comp)
            src = comp["source"]
            if selected:
                if src == "phspectra":
                    n_sel_ph += 1
                elif src == "gausspyplus":
                    n_sel_gp += 1
                else:
                    n_sel_man += 1
            base_color = _COLOR_PH if src == "phspectra" else _COLOR_GP if src == "gausspyplus" else _COLOR_MAN
            color = _COLOR_SELECTED if selected else base_color
            lw = _LW_SELECTED if selected else _LW_NORMAL
            self._plot_component(ax, x, comp, color, lw)
            label_color = _COLOR_SELECTED if selected else base_color
            label_info.append((comp["mean"], comp["amplitude"], _KEYS[i], label_color))

        # Second pass: place labels, offsetting vertically when peaks overlap.
        amp_range = float(np.nanmax(signal) - np.nanmin(signal)) if len(signal) else 1.0
        offset_step = amp_range * 0.06
        placed: list[tuple[float, float]] = []
        for mean, amp, key, color in label_info:
            y = amp
            for pm, py in placed:
                if abs(mean - pm) < 5.0 and abs(y - py) < offset_step:
                    y = py + offset_step
            placed.append((mean, y))
            ax.text(mean, y, key, ha="center", va="bottom", color=color, fontweight="bold")

        # Legend.
        n_ph = len(pd.ph_components)
        n_gp = len(pd.gp_components)
        n_man = sum(1 for c in self._keyed_comps if c["source"] == "manual")
        handles = [
            Line2D([], [], color=_COLOR_SIGNAL_ORIG, linewidth=0.6, linestyle="--", label="Signal"),
            Line2D([], [], color=_COLOR_SIGNAL, linewidth=0.8, label="Residual"),
            Line2D([], [], color=_COLOR_PH, linewidth=_LW_NORMAL, label=f"PHS: {n_sel_ph}/{n_ph}"),
            Line2D([], [], color=_COLOR_GP, linewidth=_LW_NORMAL, label=f"GP+: {n_sel_gp}/{n_gp}"),
            Line2D([], [], color=_COLOR_MAN, linewidth=_LW_NORMAL, label=f"MAN: {n_sel_man}/{n_man}"),
            Line2D([], [], color=_COLOR_SELECTED, linewidth=_LW_SELECTED, label="Selected"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False)

        # Global training set totals label.
        entries = self._ts._entries  # pylint: disable=protected-access
        total_ph = sum(1 for e in entries for c in e.get("components", []) if c.get("source") == "phspectra")
        total_gp = sum(1 for e in entries for c in e.get("components", []) if c.get("source") == "gausspyplus")
        total_man = sum(1 for e in entries for c in e.get("components", []) if c.get("source") == "manual")
        ax.text(
            0.01,
            0.97,
            f"Training set  PHS: {total_ph}  GP+: {total_gp}  MAN: {total_man}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.7},
        )

        # Title.
        curated = self._ts.has_pixel(pixel)
        check = " [curated]" if curated else ""
        n_total = len(self._data)
        ax.set_title(
            f"Pixel ({pixel[0]}, {pixel[1]}) - {self._index + 1}/{n_total}{check}",
        )
        ax.set_xlabel("Channel")
        ax.set_ylabel("Amplitude (K)")
        ax.grid(True, alpha=0.15)

        # Re-attach SpanSelector to the cleared axes.
        self._span = SpanSelector(
            ax,
            self._on_span,
            "horizontal",
            useblit=True,
            props={"alpha": 0.25, "facecolor": _COLOR_MAN},
            interactive=False,
        )

        self._draw_component_list(pixel)

        self._fig.canvas.draw_idle()

    def _draw_component_list(self, pixel: tuple[int, int]) -> None:
        """Draw the keyed component list in the bottom panel."""
        ax = self._ax_list
        ax.clear()
        ax.set_axis_off()

        lines: list[str] = []
        for i, comp in enumerate(self._keyed_comps):
            if i >= len(_KEYS):
                break
            key = _KEYS[i]
            selected = self._ts.has_component(pixel, comp)
            mark = "*" if selected else " "
            src_val = comp["source"]
            src = "PHS" if src_val == "phspectra" else "GP+" if src_val == "gausspyplus" else "MAN"
            lines.append(
                f"[{key}]{mark} {src}"
                f"  $A$={comp['amplitude']:7.3f}"
                f"  $\\mu$={comp['mean']:7.1f}"
                f"  $\\sigma$={comp['stddev']:6.2f}"
            )

        if not lines:
            lines.append("(no components)")

        text = "\n".join(lines)
        ax.text(
            0.02,
            0.95,
            text,
            transform=ax.transAxes,
            verticalalignment="top",
        )

    @staticmethod
    def _plot_component(
        ax: plt.Axes,  # type: ignore[name-defined]
        x: npt.NDArray[np.float64],
        comp: dict[str, Any],
        color: str,
        lw: float,
    ) -> None:
        """Plot a single Gaussian clipped to +/-5 sigma."""
        mean = comp["mean"]
        stddev = max(comp["stddev"], 1e-10)
        lo = max(mean - 5 * stddev, x[0])
        hi = min(mean + 5 * stddev, x[-1])
        mask = (x >= lo) & (x <= hi)
        xc = x[mask]
        yc = _gaussian(xc, comp["amplitude"], mean, stddev)
        ax.plot(xc, yc, color=color, linewidth=lw, alpha=_ALPHA_COMP)

    def _on_span(self, x_min: float, x_max: float) -> None:
        """Fit a single Gaussian to the residual within the selected span."""
        pd = self._data.pixel_data[self._index]
        pixel = pd.pixel
        signal = pd.signal
        x = np.arange(len(signal), dtype=np.float64)

        # Compute residual (same logic as _draw_pixel).
        residual = signal.copy()
        for comp in self._keyed_comps:
            if self._ts.has_component(pixel, comp):
                residual = residual - _gaussian(x, comp["amplitude"], comp["mean"], comp["stddev"])

        # Slice to the selected range.
        mask = (x >= x_min) & (x <= x_max)
        x_slice = x[mask]
        r_slice = residual[mask]

        if len(x_slice) < 3:
            return

        # Initial guesses.
        amp_guess = float(np.max(r_slice))
        mean_guess = (x_min + x_max) / 2.0
        std_guess = (x_max - x_min) / 4.0

        try:
            popt, _ = curve_fit(
                _gaussian,
                x_slice,
                r_slice,
                p0=[amp_guess, mean_guess, std_guess],
                maxfev=5000,
            )
        except RuntimeError:
            print("Gaussian fit did not converge — try a different span.")
            return

        amp_fit, mean_fit, std_fit = float(popt[0]), float(popt[1]), float(abs(popt[2]))
        comp = {
            "amplitude": amp_fit,
            "mean": mean_fit,
            "stddev": std_fit,
            "source": "manual",
        }
        self._ts.add_component(pixel, comp)
        self._ts.save()
        self._draw_pixel()
        print(f"Manual fit: A={amp_fit:.3f}  mu={mean_fit:.1f}  sigma={std_fit:.2f}")

    def _on_key(self, event: KeyEvent) -> None:
        key = event.key
        if key == "right":
            if self._index < len(self._data) - 1:
                self._ts.save()
                self._index += 1
                self._draw_pixel()
        elif key == "left":
            if self._index > 0:
                self._ts.save()
                self._index -= 1
                self._draw_pixel()
        elif key == "s":
            self._ts.save()
            print(f"Saved training set ({len(self._ts.curated_pixels)} curated pixels).")
        elif key == "c":
            pixel = self._data.pixel_data[self._index].pixel
            self._ts.clear_pixel(pixel)
            self._ts.save()
            self._draw_pixel()
            print(f"Cleared pixel {pixel}.")
        elif key == "q":
            self._ts.save()
            print(f"Saved and quitting ({len(self._ts.curated_pixels)} curated pixels).")
            plt.close(self._fig)
        elif key in _KEYS:
            idx = _KEYS.index(key)
            if idx < len(self._keyed_comps):
                comp = self._keyed_comps[idx]
                pixel = self._data.pixel_data[self._index].pixel
                if self._ts.has_component(pixel, comp):
                    self._ts.remove_component(pixel, comp)
                else:
                    self._ts.add_component(pixel, comp)
                self._ts.save()
                self._draw_pixel()

    def show(self) -> None:
        """Show the interactive figure (blocks until closed)."""
        plt.show()
