---
sidebar_position: 1
---

# Motivation

## The problem: spectral line decomposition

Radio-astronomical spectra — particularly HI 21-cm emission — are typically a superposition of multiple Gaussian components, each corresponding to a distinct gas cloud along the line of sight. Recovering these individual components is essential for understanding the structure and kinematics of the interstellar medium (ISM).

This is a **blind decomposition** problem: given a noisy 1D signal, determine the number of components and fit their parameters (amplitude, centre, width) without prior knowledge.

## The current approach: GaussPy

[GaussPy](https://arxiv.org/abs/1409.2840) and its successor [GaussPy+](https://arxiv.org/abs/1906.10506) use **derivative spectroscopy** to identify features in spectra:

1. Smooth the spectrum with a Gaussian kernel of width controlled by parameters $\alpha_1$ and $\alpha_2$ (two-phase decomposition: $\alpha_1$ finds broad features, $\alpha_2$ catches narrower ones in the residual).
2. Compute the second (and optionally fourth) derivative of the smoothed spectrum.
3. Identify zero-crossings as candidate peak locations.
4. Fit Gaussians to the identified peaks.

The key limitation is that $\alpha_1$ and $\alpha_2$ **must be trained** on synthetic spectra whose decomposition is known. The choice of $\alpha$ is a sensitivity/specificity trade-off:

- Too small: noise creates spurious peaks.
- Too large: real features are smoothed away.

GaussPy+ adds spatial coherence constraints for spectral cubes, but the fundamental dependence on a trained smoothing parameter remains.

## A topological alternative: persistent homology

**Persistent homology** offers a fundamentally different approach to peak detection. Instead of smoothing and differentiating, it analyses the **topology** of the function's upper-level sets as a threshold descends from the maximum.

The persistence of a peak — the difference between its birth (height) and death (merge level) — provides a natural measure of significance. Small-persistence features correspond to noise; large-persistence features are real peaks.

### The beta parameter

Raw persistence is measured in the same units as the signal, so a fixed threshold cannot distinguish noise from signal across datasets with different noise levels. We introduce a single free parameter $\beta$ that sets the persistence threshold in units of noise:

$$
\mathrm{min}_{\mathrm{persistence}} = \beta \times \sigma_{\mathrm{rms}}
$$

where $\sigma_{\rm rms}$ is estimated robustly from the data itself (via the median absolute deviation). This makes $\beta$ the **sole free parameter** of the model — it controls the sensitivity/specificity trade-off, analogous to a signal-to-noise cut.

### Comparison with GaussPy

|                       | GaussPy (derivatives).                                   | phspectra (persistence + $\beta$)               |
| --------------------- | -------------------------------------------------------- | ----------------------------------------------- |
| **Free parameters**   | $\alpha_1, \alpha_2$ (smoothing widths) + SNR thresholds | $\beta$ (persistence in noise units)            |
| **Training**          | Requires synthetic training set                          | Single scalar — grid search or 1-D optimization |
| **Peak significance** | Implicit (via smoothing)                                 | Explicit (persistence)                          |
| **Noise handling**    | Smoothing kernel                                         | MAD-based $\sigma$ estimate $ \times \beta$     |
| **Multi-scale**       | Single scale per $\alpha$                                | All scales simultaneously                       |

$\beta = 5$ (a 5$\sigma$ persistence cut) is a reasonable default. Training $\beta$ on a labeled dataset is a simple 1-D optimization, compared to GaussPy's multi-parameter search over synthetic spectra.

## Why this matters

For large-scale surveys (e.g. [GALFA-HI](https://purcell.ssl.berkeley.edu/), [THOR](https://www2.mpia-hd.mpg.de/thor/Overview.html), [GASKAP](https://research.csiro.au/gaskap-hi/)), millions of spectra must be decomposed. A method that:

1. Has a **single, interpretable** free parameter ($\beta$) instead of multiple trained smoothing scales,
2. Provides a principled significance measure grounded in topology,
3. Naturally handles multi-scale features,

could significantly improve both the **reliability** and **scalability** of spectral line decomposition.
