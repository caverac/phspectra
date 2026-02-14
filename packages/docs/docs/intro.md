---
slug: /
sidebar_position: 1
---

# PHSpectra

**Persistent homology for spectral line decomposition.**

This project explores a novel approach to decomposing astronomical spectra into individual Gaussian components. Instead of relying on derivative spectroscopy (as in [GaussPy](https://gausspy.readthedocs.io/) and [GaussPy+](https://arxiv.org/abs/1906.10506)), we use **persistent homology** -- a tool from topological data analysis -- to identify peaks and measure their significance with a single, intuitive parameter.

## The $\beta$ parameter

The main tuning parameter in phspectra is $\beta$, the persistence threshold in units of the estimated noise $\sigma_\mathrm{rms}$. A peak in the spectrum is retained as a candidate Gaussian component only if its topological persistence -- the height difference between its birth (peak value) and death (the level at which it merges with a taller neighbor) -- exceeds $\beta \cdot \sigma_\mathrm{rms}$. In other words, $\beta$ sets the minimum significance, in units of $\sigma$, for a peak to be considered real rather than noise.

The default value $\beta = 3.8$ works well across both real and synthetic data. Performance is remarkably insensitive to $\beta$: sweeping from 3.8 to 4.5 changes $F_1$ by less than 0.01 (see [Beta sensitivity](results/beta)). Because $\beta$ has a direct physical interpretation -- minimum peak significance in units of $\sigma$ -- a single default generalizes across surveys without a training step.

## What problem does this solve?

Radio-astronomical surveys (e.g. HI 21-cm emission) produce spectra that are often a superposition of multiple Gaussian components. Decomposing these spectra is critical for understanding the kinematics and structure of the interstellar medium.

Existing tools like GaussPy require trained smoothing parameters ($\alpha_1$, $\alpha_2$) and use derivatives to detect features. Our approach replaces this with 0-dimensional persistent homology, which naturally ranks peaks by their topological persistence -- no training step required.

## Quick links

- [Motivation](idea-and-plan/motivation) -- Why topology instead of derivatives?
- [Persistent Homology](idea-and-plan/persistent-homology-primer) -- Mathematics, algorithm, and integration in phspectra
- [Data Sources](idea-and-plan/data-sources) -- Surveys and catalogs used for benchmarking
