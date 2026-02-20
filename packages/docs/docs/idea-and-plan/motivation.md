---
sidebar_position: 1
---

# Motivation

## The problem: spectral line decomposition

Along any line of sight, radio-astronomical spectra -- HI 21-cm, ${}^{13}$CO, and other tracers -- sample emission from multiple gas clouds at different radial velocities. Within each cloud, random thermal motions (Maxwell-Boltzmann) and small-scale turbulent motions both produce Gaussian velocity distributions. The observed linewidth of a single cloud is $\sigma = \sqrt{\sigma_\mathrm{th}^2 + \sigma_\mathrm{turb}^2}$, where $\sigma_\mathrm{th} = \sqrt{k_B T / m}$ is the thermal broadening and $\sigma_\mathrm{turb}$ captures non-thermal contributions. In the optically thin limit the brightness temperature at each velocity is proportional to the column density of emitting gas, so each cloud contributes a Gaussian profile centred on its bulk velocity. The observed spectrum is therefore a **linear superposition of Gaussians**, one per kinematically distinct component along the line of sight.

Recovering these individual components -- their amplitudes, centroid velocities, and widths -- is essential for understanding the structure and kinematics of the interstellar medium (ISM). This is a **blind decomposition** problem: given a noisy 1D signal, determine the number of components and fit their parameters without prior knowledge.

## Persistent homology for peak detection

**Persistent homology** detects peaks by analysing the **topology** of the function's upper-level sets as a threshold descends from the maximum.

The persistence of a peak -- the difference between its birth (height) and death (merge level) -- provides a natural measure of significance. Small-persistence features correspond to noise; large-persistence features are real peaks.

### The beta parameter

Raw persistence is measured in the same units as the signal, so a fixed threshold cannot distinguish noise from signal across datasets with different noise levels. The main tuning parameter is $\beta$, which sets the persistence threshold in units of noise:

$$
\pi_{\min} = \beta \times \sigma_{\mathrm{rms}}
$$

where $\sigma_{\rm rms}$ is estimated robustly from the data itself using signal-masked RMS estimation (following [Riener et al. 2019](https://arxiv.org/abs/1906.10506), Sect. 3.1.1): positive-channel runs are masked, a MAD-based clip removes outliers, and the final $\sigma$ is computed as the RMS of surviving noise channels.

### Component validation

After persistence selects candidate peaks and Gaussians are fitted, each component must pass three additional quality checks before it is accepted:

| Check                  | Criterion                                                                   | Default     |
| ---------------------- | --------------------------------------------------------------------------- | ----------- |
| **SNR floor**          | Amplitude $\geq$ `snr_min` $\times \sigma_\mathrm{rms}$                     | 1.5         |
| **Matched-filter SNR** | $(A_i / \sigma_\mathrm{rms})\,\sqrt{\sigma_i}\;\pi^{1/4} \geq$ `mf_snr_min` | 5.0         |
| **Minimum width**      | FWHM $\geq$ `fwhm_min_channels`                                             | 1.0 channel |

The matched-filter SNR is the optimal detection signal-to-noise for a Gaussian component in white noise. Because it scales as $\sqrt{\sigma}$, narrow peaks must have proportionally higher amplitude to survive -- this rejects noise spikes without an ad-hoc width threshold.

Components that survive validation are kept only if they improve the AICc of the overall model.

These thresholds have sensible physical defaults that work across the datasets we have tested. In practice, $\beta$ is the only parameter that meaningfully affects decomposition results -- the validation thresholds act as safety nets rather than tuning knobs.

## Benchmark: GaussPy

Several families of decomposition algorithms exist (see [Introduction](/) for a full overview). We benchmark phspectra against [GaussPy](https://arxiv.org/abs/1409.2840) / [GaussPy+](https://arxiv.org/abs/1906.10506) because both are open-source, fully automated, and target the same use case -- blind Gaussian decomposition of radio spectral cubes. Their derivative-based approach also provides a useful contrast with topology-based peak detection.

GaussPy uses **derivative spectroscopy**: it computes the second and fourth derivatives of the spectrum and locates their zero-crossings, which mark candidate peak positions and boundaries. Because finite-difference derivatives amplify noise, GaussPy regularizes them via **Total Variation (TV) regularization**, solving an optimization problem that balances data fidelity against a smoothness penalty:

$$
R[u] = \alpha \int \sqrt{(D_x u)^2 + \beta_{\mathrm{TV}}^2} \;+\; \int |A_x u - f|^2
$$

where $u$ is the regularized derivative, $A_x$ is the anti-derivative operator, $f$ is the observed spectrum, and $\alpha$ controls the regularization strength. A **two-phase decomposition** uses $\alpha_1$ for broad features and $\alpha_2$ for narrower residual structure. Both parameters must be **trained** on synthetic spectra via supervised gradient descent that maximizes $F_1$. GaussPy+ adds spatial coherence constraints for spectral cubes, but the dependence on trained regularization parameters remains.

The key difference is in how peaks are found: GaussPy smooths the signal and differentiates; phspectra reads the peak structure directly from the topology of the data. The table below summarizes the comparison:

|                       | GaussPy                                                     | phspectra                                              |
| --------------------- | ----------------------------------------------------------- | ------------------------------------------------------ |
| **Peak detection**    | TV-regularized 2nd/4th derivative zero-crossings            | Persistent homology (all scales simultaneously)        |
| **Tuning parameters** | $\alpha_1, \alpha_2$ (regularization strength) + SNR cutoff | $\beta$ (persistence threshold in noise units)         |
| **Training**          | Supervised gradient descent on synthetic spectra            | Not required -- default $\beta = 3.5$ generalizes      |
| **Peak significance** | Implicit (via regularization strength)                      | Explicit (topological persistence)                     |
| **Validation**        | SNR threshold (trained)                                     | SNR floor + matched-filter SNR + AICc (fixed defaults) |

## Why this matters

For large-scale surveys (e.g. [GALFA-HI](https://purcell.ssl.berkeley.edu/), [THOR](https://www2.mpia-hd.mpg.de/thor/Overview.html), [GASKAP](https://research.csiro.au/gaskap-hi/)), millions of spectra must be decomposed. A method that:

1. Has a **single tuning parameter** ($\beta$) with sensible fixed defaults for everything else,
2. Provides a principled significance measure grounded in topology,
3. Naturally handles multi-scale features without smoothing,
4. Is deterministic and reproducible,

could significantly improve both the **reliability** and **scalability** of spectral line decomposition.
