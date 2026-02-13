"""0-dimensional persistent homology peak detection for 1D signals.

Implements the "descending water level" algorithm: sweep a threshold from the
global maximum downward. Each time the threshold drops below a local maximum a
new connected component is *born*; when two components merge, the younger one
(smaller maximum) *dies*. The persistence of a peak is ``birth - death`` and
serves as a parameter-free measure of significance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PersistentPeak:
    """A peak identified by 0-dim persistent homology.

    Attributes
    ----------
    index : int
        Index of the local maximum in the original signal.
    birth : float
        Function value at which this component was born (the peak height).
    death : float
        Function value at which this component merged into an older one.
    persistence : float
        ``birth - death`` -- the significance of the peak.
    saddle_index : int
        Channel index of the saddle point where this component died
        (merged into an older component).  For the global maximum,
        which never dies, this is set to ``-1``.
    """

    index: int
    birth: float
    death: float
    persistence: float
    saddle_index: int = -1


def find_peaks_by_persistence(
    signal: NDArray[np.floating],
    *,
    min_persistence: float = 0.0,
) -> list[PersistentPeak]:
    """Detect peaks in a 1D signal using 0-dim persistent homology.

    The algorithm processes samples in order of *decreasing* function value
    (the "upper-level set filtration"). A union-find structure tracks connected
    components as the threshold descends.

    Parameters
    ----------
    signal:
        1-D array of function values.
    min_persistence:
        Discard peaks whose persistence is below this threshold.

    Returns
    -------
    list[PersistentPeak]
        Peaks sorted by persistence (most significant first).
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = len(signal)
    if n == 0:
        return []

    # Union-Find ---------------------------------------------------------------
    parent = np.arange(n)
    rank = np.zeros(n, dtype=np.intp)
    # The *representative* of each component is the index with the highest value
    rep = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> int:
        """Merge two components, returning the root of the merged tree."""
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1
        # The representative of the merged component is the one with the
        # higher function value (the "older" component).
        if signal[rep[b]] > signal[rep[a]]:
            rep[a] = rep[b]
        return a

    # Process indices in decreasing order of function value --------------------
    order = np.argsort(-signal)
    visited = np.zeros(n, dtype=bool)
    peaks: list[PersistentPeak] = []

    for idx in order:
        visited[idx] = True
        comp = idx  # starts as its own component

        for neighbor in (idx - 1, idx + 1):
            if neighbor < 0 or neighbor >= n or not visited[neighbor]:
                continue
            neighbor_root = find(neighbor)
            idx_root = find(comp)
            if idx_root == neighbor_root:
                continue

            # Determine which representative has the *lower* peak (younger).
            rep_ir = rep[idx_root]
            rep_nr = rep[neighbor_root]
            younger_rep = rep_ir if signal[rep_ir] < signal[rep_nr] else rep_nr

            # The younger component dies at the current threshold.
            death = signal[idx]
            birth = signal[younger_rep]
            persistence = birth - death
            if persistence > min_persistence:
                peaks.append(
                    PersistentPeak(
                        index=int(younger_rep),
                        birth=float(birth),
                        death=float(death),
                        persistence=float(persistence),
                        saddle_index=int(idx),
                    )
                )

            merged = union(find(comp), find(neighbor))
            comp = merged

    # The global maximum component never dies -- record it with infinite
    # persistence if it passes the threshold.
    global_max_idx = int(order[0])
    global_birth = float(signal[global_max_idx])
    if global_birth >= min_persistence:
        peaks.append(
            PersistentPeak(
                index=global_max_idx,
                birth=global_birth,
                death=0.0,
                persistence=global_birth,
            )
        )

    peaks.sort(key=lambda p: p.persistence, reverse=True)
    return peaks
