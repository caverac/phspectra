---
slug: /
sidebar_position: 1
---

# PHSpectra

**Persistent homology for spectral line decomposition.**

This project explores a novel approach to decomposing astronomical spectra into individual Gaussian components. Instead of relying on derivative spectroscopy (as in [GaussPy](https://gausspy.readthedocs.io/) and [GaussPy+](https://arxiv.org/abs/1906.10506)), we use **persistent homology** — a tool from topological data analysis — to identify peaks and measure their significance in a parameter-free way.

## What problem does this solve?

Radio-astronomical surveys (e.g. HI 21-cm emission) produce spectra that are often a superposition of multiple Gaussian components. Decomposing these spectra is critical for understanding the kinematics and structure of the interstellar medium.

Existing tools like GaussPy require a trained smoothing parameter (alpha) and use derivatives to detect features. Our approach replaces this with 0-dimensional persistent homology, which naturally ranks peaks by their topological persistence — no training step required.

## Quick links

- [Motivation](idea-and-plan/motivation) — Why topology instead of derivatives?
- [Persistent Homology](idea-and-plan/persistent-homology-primer) — Mathematics, algorithm, and integration in phspectra
- [Plan of Attack](idea-and-plan/plan-of-attack) — Roadmap and milestones
