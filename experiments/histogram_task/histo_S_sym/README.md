### Experiment: Symmetric Unfactorized Attention on the Structured Discrete Histogram Task

This directory contains scripts for analyzing the impact of symmetry alone ($S = S^\top$) on the learning dynamics of the Histogram Task.

## 1. Overview
The goal is to determine if enforcing symmetry in a full-rank attention matrix is sufficient to induce generalization. This experiment compares the **Symmetric Unfactorized** model against the **Tied (PSD)** model to isolate the specific role of the Positive Semi-Definite constraint.

## 2. Task Definition
* **Objective**: Exact count prediction for discrete tokens.
* **Requirement**: The model must learn to attend to all positions $j$ where $x_j = x_i$.
* **Hierarchy**: Empirical results show a strict performance hierarchy where symmetry alone fails to match factorized models: Tied > Untied > Symmetric Unfactorized.

## 3. Architecture
* **Symmetry Constraint**: The interaction matrix is forced to be symmetric ($S = S^\top$), reducing the parameter count to $\approx 0.5d^2$.
* **Lack of PSD**: Unlike the $WW^\top$ parameterization, this model can sustain negative eigenvalues.



## 4. Key Findings: The Insufficiency of Symmetry
* **Persistent Negative Eigenvalues**: During training, the model fails to eliminate negative eigenvalues, which correspond to non-physical "repulsive" interactions.
* **Stagnation**: Performance typically stagnates at intermediate accuracy levels ($\approx 50\%$), failing to reach the semantic solution robustly.
* **Parameter Parsimony vs. Geometry**: Although this model has the fewest parameters among full-rank variants, it performs poorly, proving that parameter reduction without PSD geometry is insufficient.

## 5. Optimization Landscape: Parasitic Minima
* **Saliency Attractor**: Without the PSD constraint, the gradient dynamics are often insufficient to escape the basin of attraction of the Saliency Trap (vertical columns).
* **Instability**: The optimization landscape is bimodal; while the model can theoretically learn the task, it lacks the robustness of the Tied architecture.