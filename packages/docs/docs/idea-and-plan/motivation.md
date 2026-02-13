---
sidebar_position: 1
---

# Motivation

## The problem: spectral line decomposition

Along any line of sight, radio-astronomical spectra — HI 21-cm, ${}^{13}$CO, and other tracers — sample emission from multiple gas clouds at different radial velocities. Within each cloud, random thermal motions (Maxwell–Boltzmann) and small-scale turbulent motions both produce Gaussian velocity distributions. The observed linewidth of a single cloud is $\sigma = \sqrt{\sigma_\mathrm{th}^2 + \sigma_\mathrm{turb}^2}$, where $\sigma_\mathrm{th} = \sqrt{k_B T / m}$ is the thermal broadening and $\sigma_\mathrm{turb}$ captures non-thermal contributions. In the optically thin limit the brightness temperature at each velocity is proportional to the column density of emitting gas, so each cloud contributes a Gaussian profile centred on its bulk velocity. The observed spectrum is therefore a **linear superposition of Gaussians**, one per kinematically distinct component along the line of sight.

Recovering these individual components — their amplitudes, centroid velocities, and widths — is essential for understanding the structure and kinematics of the interstellar medium (ISM). This is a **blind decomposition** problem: given a noisy 1D signal, determine the number of components and fit their parameters without prior knowledge.

## The current approach: GaussPy

[GaussPy](https://arxiv.org/abs/1409.2840) and its successor [GaussPy+](https://arxiv.org/abs/1906.10506) use **derivative spectroscopy** to identify features in spectra. The core idea is to compute the second and fourth derivatives of the spectrum and locate their zero-crossings, which mark candidate peak positions and boundaries.

Because finite-difference derivatives amplify noise, GaussPy regularizes them via **Total Variation (TV) regularization**. Instead of convolving with a smoothing kernel, it solves an optimization problem that balances data fidelity against a smoothness penalty:

$$
R[u] = \alpha \int \sqrt{(D_x u)^2 + \beta_{\mathrm{TV}}^2} \;+\; \int |A_x u - f|^2
$$

where $u$ is the regularized derivative, $A_x$ is the anti-derivative operator, $f$ is the observed spectrum, and $\alpha$ controls the strength of the regularization. When $\alpha \to 0$ the solution converges to the noisy finite-difference derivative; as $\alpha$ grows the derivative becomes increasingly smooth and piecewise-constant features are preserved.

GaussPy uses a **two-phase decomposition**: $\alpha_1$ sets the regularization for an initial pass that recovers broad features, and $\alpha_2$ is applied to the residual to catch narrower ones. Both parameters **must be trained** on synthetic spectra whose decomposition is known, using a supervised gradient-descent procedure that maximizes the $F_1$ score. The empirical relationship $\delta_\mathrm{chan} \approx 3.7 \times 1.8^{\log \alpha}$ links $\alpha$ to the minimum spatial scale preserved in the derivative.

GaussPy+ adds spatial coherence constraints for spectral cubes, but the fundamental dependence on trained regularization parameters remains.

## A topological alternative: persistent homology

**Persistent homology** offers a fundamentally different approach to peak detection. Instead of smoothing and differentiating, it analyses the **topology** of the function's upper-level sets as a threshold descends from the maximum.

The persistence of a peak — the difference between its birth (height) and death (merge level) — provides a natural measure of significance. Small-persistence features correspond to noise; large-persistence features are real peaks.

### The beta parameter

Raw persistence is measured in the same units as the signal, so a fixed threshold cannot distinguish noise from signal across datasets with different noise levels. The main tuning parameter is $\beta$, which sets the persistence threshold in units of noise:

$$
\pi_{\min} = \beta \times \sigma_{\mathrm{rms}}
$$

where $\sigma_{\rm rms}$ is estimated robustly from the data itself using signal-masked RMS estimation (following [Riener et al. 2019](https://arxiv.org/abs/1906.10506), Sect. 3.1.1): positive-channel runs are masked, a MAD-based clip removes outliers, and the final $\sigma$ is computed as the RMS of surviving noise channels.

### Component validation

After persistence selects candidate peaks and Gaussians are fitted, each component must pass three additional quality checks before it is accepted:

| Check | Criterion | Default |
|-------|-----------|---------|
| **SNR floor** | Amplitude $\geq$ `snr_min` $\times \sigma_\mathrm{rms}$ | 1.5 |
| **Significance** | Integrated flux per FWHM normalised by noise $\geq$ `sig_min` | 5.0 |
| **Minimum width** | FWHM $\geq$ `fwhm_min_channels` | 1.0 channel |

Components that survive validation are kept only if they improve the AICc of the overall model. During iterative refinement (residual peak search, negative-dip splitting, blended-pair merging) the significance threshold relaxes to `sig_min = 4.0`.

These thresholds have sensible physical defaults that work across the datasets we have tested. In practice, $\beta$ is the only parameter that meaningfully affects decomposition results — the validation thresholds act as safety nets rather than tuning knobs.

### Comparison with GaussPy

|                       | GaussPy                                                     | phspectra                                         |
| --------------------- | ----------------------------------------------------------- | ------------------------------------------------- |
| **Peak detection**    | TV-regularized 2nd/4th derivative zero-crossings            | Persistent homology (all scales simultaneously)   |
| **Tuning parameters** | $\alpha_1, \alpha_2$ (regularization strength) + SNR cutoff | $\beta$ (persistence threshold in noise units)    |
| **Training**          | Supervised gradient descent on synthetic spectra            | Not required — default $\beta = 4.0$ generalizes |
| **Peak significance** | Implicit (via regularization strength)                      | Explicit (topological persistence)                |
| **Validation**        | SNR threshold (trained)                                     | SNR floor + significance + AICc (fixed defaults)  |

$\beta = 4$ (a 4$\sigma$ persistence cut) is the default. Training $\beta$ on a labeled dataset is a simple 1-D optimization and automatically covers all scales.

## Why this matters

For large-scale surveys (e.g. [GALFA-HI](https://purcell.ssl.berkeley.edu/), [THOR](https://www2.mpia-hd.mpg.de/thor/Overview.html), [GASKAP](https://research.csiro.au/gaskap-hi/)), millions of spectra must be decomposed. A method that:

1. Has a **single tuning parameter** ($\beta$) with sensible fixed defaults for everything else,
2. Provides a principled significance measure grounded in topology,
3. Naturally handles multi-scale features,

could significantly improve both the **reliability** and **scalability** of spectral line decomposition.
