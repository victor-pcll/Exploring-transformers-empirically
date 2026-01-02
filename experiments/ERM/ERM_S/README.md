### Synthetic Experiment: Unfactorized Attention (Teacher-Student)

This directory contains the framework for analyzing **Unfactorized (Full-Rank)** attention layers using random matrix theory.

## 1. Overview
This baseline study compares unstructured $d^2$ parameterizations against factorized models. It highlights the "complexity gap" and the limits of generalization in the absence of structural priors.

## 2. Mathematical Setup
* **Parameterization**: Both teacher $S^*$ and student $S$ are full-rank matrices with i.i.d. Gaussian entries $\mathcal{N}(0, 1/d)$.
* **Spectral Scaling**: Initialized as a real Ginibre ensemble where eigenvalues follow the Circular Law.
* **Complexity**: The task involves $d^2$ independent degrees of freedom, representing the highest possible task entropy.

## 3. Key Findings
* **Double Descent**: Like the Untied model, it exhibits a characteristic peak at $\alpha \approx 0.5$, but with a significantly higher error floor.
* **Spectral Evolution**: The student's spectrum evolves from a collapsed peak at zero to a semi-circular distribution (projected Circular Law), matching the teacher's spectral scale at $\alpha \approx 1.0$.
* **Entropy Barrier**: Persistent performance gaps in the $\alpha < 1$ regime confirm that without low-rank priors, the model prioritizes noise interpolation over structural recovery.