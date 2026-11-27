---
title: 'ecgrecon: Reconstructing complete 12-lead ECGs from paper printouts'
tags:
  - ECG
  - electrocardiography
  - signal reconstruction
  - digital signal processing
  - Python
authors:
  - name: Bjørn-Jostein Singstad
    affiliation: "1, 2, 3, 4"
affiliations:
  - index: 1
    name: Department of Electrical Engineering and Computer Science, University of South-Eastern Norway, Horten, Norway
  - index: 2
    name: Vestfold Hospital Trust, Tønsberg, Norway
  - index: 3
    name: Akershus University Hospital, Lørenskog, Norway
  - index: 4
    name: Department of Informatics, University of Oslo, Oslo, Norway
date: 27 November 2026
bibliography: paper.bib
---

## Summary

Clinical electrocardiograms (ECGs) are frequently stored as paper printouts in digitized form (scanned images). Complete 12-lead ECGs contain substantial redundancy due to Einthoven's law—the derived limb leads (III, aVR, aVL, aVF) are exact linear combinations of the independent leads (I, II, V1–V6). `ecgrecon` is a Python package for reconstructing complete, physiologically consistent 12-lead ECGs from paper printouts where only partial leads are visible (typically a 2.5-second Cabrera format layout plus rhythm strips). The software employs a Bayesian ridge regression approach with patient-specific lead transforms and beat-locked residuals, achieving higher correlation and lower error than deep learning baselines on the PTB-XL dataset.

## Statement of Need

As ECG machines shifted from paper to digital systems, many clinical institutions archived their historical records as scanned paper images. Digitizing these archives requires recovering the full 12-lead signal from partial printouts. The `ecgrecon` package addresses this need by combining signal-processing theory (Dower matrix transformations, Einthoven constraints) with patient-level learning, making it suitable for historical data recovery and research applications in cardiology, epidemiology, and machine learning.

## Features

- **Physiologically consistent reconstruction**: Enforces Einthoven's law exactly; derived leads are always correct linear combinations
- **Patient-specific learning**: Bayesian ridge FIR lead transforms with population priors
- **Beat-locked residuals**: Leverages R-peak-aligned beat templates to capture non-dipolar components
- **Flexible evaluation**: Missing-sample correlation, RMSE, shape correlation, and artefact detection
- **Stratified data handling**: Supports patient-wise cross-validation on PTB-XL

## Implementation

The core algorithm reconstructs the signal $\mathbf{ecg}(t) = \mathbf{L} \mathbf{h}(t) + \mathbf{r}(t)$, where:
- $\mathbf{L}$ is a patient-specific Bayesian ridge lead transform
- $\mathbf{h}(t)$ are the independent leads (observed or partially observed)
- $\mathbf{r}(t)$ is a beat-locked non-dipolar residual

The method is implemented in pure Python with NumPy and SciPy, with no mandatory deep learning dependencies, making it lightweight and reproducible.

## Comparison with Related Work

A concurrent training-free method (Hassannia et al., CinC 2026) uses Einthoven recovery, per-lead beat templates, and null-space projection, scored on all PTB-XL records by signal-to-noise ratio and fiducial errors. Our method additionally learns patient-specific lead relationships and uses multiple reference leads (II, V1, V5), achieving higher correlation on the U-Net's test set, though a direct head-to-head comparison remains future work.

## Validation and Performance

On the PTB-XL dataset using stratified patient-wise validation:
- **Test set correlation**: median r = 0.95 (vs. U-Net r ≈ 0.90–0.93)
- **Test set RMSE**: median 0.26 mV (vs. U-Net 0.4–0.5 mV)
- **Einthoven consistency**: <0.001 mV error (U-Net ~0.055 mV)

The software has been tested on 5,567 PTB-XL records and handles artefacts, missing samples, and variable paper layouts robustly.

## Availability and Reproducibility

- **Repository**: https://github.com/[username]/ECGRecovery
- **Package**: Installable via pip; tested on Python 3.9+
- **Tests**: 47 unit and integration tests in `tests/`
- **Data**: Uses the public PTB-XL dataset (Wagner et al., 2020)
- **Documentation**: Included in README and docstrings

## AI Usage Disclosure

This software was developed with assistance from Claude (Anthropic), a large language model. The assistance included: code generation and refactoring for the core algorithms, test scaffolding, documentation drafting, and manuscript authoring. All outputs were reviewed, validated, and edited by the human author, who made all core design decisions. The human author is fully responsible for the accuracy, correctness, and licensing of the software.

## References
