---
sidebar_position: 2
---

# API

## `fit_gaussians`

The main entry point. Decomposes a 1-D signal into Gaussian components.

```python
from phspectra import fit_gaussians

components = fit_gaussians(
    signal,              # 1-D NumPy array of flux values
    *,
    beta=3.5,            # persistence threshold in units of noise sigma
    max_refine_iter=3,   # maximum refinement iterations
    snr_min=1.5,         # minimum amplitude SNR
    mf_snr_min=5.0,      # minimum matched-filter SNR
    f_sep=1.2,           # blended-pair separation factor (FWHM units)
    neg_thresh=5.0,      # negative-dip threshold (RMS units)
)
```

**Parameters**

| Parameter         | Type                   | Default      | Description                                                                                                                          |
| ----------------- | ---------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `signal`          | `NDArray[np.floating]` | _(required)_ | 1-D spectrum (flux values)                                                                                                           |
| `beta`            | `float`                | `3.5`        | Persistence threshold in units of $\sigma_\mathrm{rms}$. The threshold is computed as $\beta \times \sigma_\mathrm{rms}$.            |
| `max_refine_iter` | `int`                  | `3`          | Maximum refinement iterations.                                                                                                       |
| `snr_min`         | `float`                | `1.5`        | Minimum amplitude $A / \sigma_\mathrm{rms}$ for a component to survive validation.                                                   |
| `mf_snr_min`      | `float`                | `5.0`        | Minimum matched-filter SNR: $\mathrm{SNR}_\mathrm{mf} = (A/\sigma_\mathrm{rms})\sqrt{\sigma}\;\pi^{1/4}$                             |
| `f_sep`           | `float`                | `1.2`        | Two components are considered blended when $\lvert\mu_i - \mu_j\rvert < f_\mathrm{sep} \cdot \min(\mathrm{FWHM}_i, \mathrm{FWHM}_j)$ |
| `neg_thresh`      | `float`                | `5.0`        | A negative residual dip deeper than $\texttt{neg\_thresh} \cdot \sigma_\mathrm{rms}$ triggers component splitting.                   |

**Returns** `list[GaussianComponent]` -- fitted Gaussians sorted by mean position. Returns an empty list when no peaks survive thresholding.

## `find_peaks_by_persistence`

Low-level peak detection via 0-dimensional persistent homology.

```python
from phspectra import find_peaks_by_persistence

peaks = find_peaks_by_persistence(
    signal,                # 1-D NumPy array
    *,
    min_persistence=0.0,   # discard peaks below this threshold
)
```

**Returns** `list[PersistentPeak]` -- peaks sorted by persistence (most significant first).

## `estimate_rms`

Signal-masked noise estimation following Riener et al. (2019, Sect 3.1.1).

```python
from phspectra import estimate_rms

rms = estimate_rms(
    signal,          # 1-D NumPy array
    *,
    mask_pad=2,      # padding around masked positive runs
    mad_clip=5.0,    # clipping threshold in sigma
)
```

**Returns** `float` -- estimated noise RMS ($\sigma_\mathrm{rms}$).

## Types

### `GaussianComponent`

A frozen dataclass representing a single fitted Gaussian.

```python
from phspectra import GaussianComponent
```

| Field       | Type    | Description                     |
| ----------- | ------- | ------------------------------- |
| `amplitude` | `float` | Peak height of the Gaussian     |
| `mean`      | `float` | Centre position (channel units) |
| `stddev`    | `float` | Standard deviation (width)      |

The FWHM of a component is $\mathrm{FWHM} = 2\sqrt{2\ln 2} \cdot \sigma \approx 2.3548 \cdot \sigma$.

### `PersistentPeak`

A frozen dataclass representing a peak detected by persistent homology.

```python
from phspectra.persistence import PersistentPeak
```

| Field          | Type    | Description                                                                                                         |
| -------------- | ------- | ------------------------------------------------------------------------------------------------------------------- |
| `index`        | `int`   | Index of the local maximum in the signal                                                                            |
| `birth`        | `float` | Function value at which this component was born (peak height)                                                       |
| `death`        | `float` | Function value at which it merged into an older component                                                           |
| `persistence`  | `float` | `birth - death` -- the significance of the peak                                                                     |
| `saddle_index` | `int`   | Channel index of the saddle point where this component died. Set to `-1` for the global maximum (which never dies). |

## Error handling

The library does not define custom exception classes. All parameters are keyword-only (except `signal`), so passing positional arguments raises a `TypeError`.

Both `fit_gaussians` and `estimate_rms` raise `ValueError` if the input signal contains NaN values. Callers must handle missing channels (e.g. `np.nan_to_num(signal, nan=0.0)`) before calling these functions.

Internally, curve fitting may encounter `RuntimeError` (non-convergence) or `numpy.linalg.LinAlgError` (singular Jacobian). These are caught and handled gracefully -- the function returns the best result available rather than raising. If no peaks survive thresholding, `fit_gaussians` returns an empty list.

The C extension may raise `ValueError` when the parameter vector exceeds its compiled limit. In this case the library falls back to SciPy's `curve_fit` automatically.
