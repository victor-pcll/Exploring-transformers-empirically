---

### 2. README pour l'expérience Symmetric Unfactorized

```markdown
# Experiment: Symmetric Unfactorized Attention on the Structured Discrete Histogram Task

This directory contains scripts for analyzing the impact of symmetry alone ($S = S^\top$) on the learning dynamics of the Histogram Task.

## 1. Overview
The goal is to determine if enforcing symmetry in a full-rank attention matrix is sufficient to induce generalization. [cite_start]This experiment compares the **Symmetric Unfactorized** model against the **Tied (PSD)** model to isolate the specific role of the Positive Semi-Definite constraint[cite: 276, 288, 864].

## 2. Task Definition
* [cite_start]**Objective**: Exact count prediction for discrete tokens[cite: 132, 136].
* [cite_start]**Requirement**: The model must learn to attend to all positions $j$ where $x_j = x_i$[cite: 154].
* [cite_start]**Hierarchy**: Empirical results show a strict performance hierarchy where symmetry alone fails to match factorized models: Tied > Untied > Symmetric Unfactorized[cite: 211, 287].

## 3. Architecture
* [cite_start]**Symmetry Constraint**: The interaction matrix is forced to be symmetric ($S = S^\top$), reducing the parameter count to $\approx 0.5d^2$[cite: 273, 275, 864].
* [cite_start]**Lack of PSD**: Unlike the $WW^\top$ parameterization, this model can sustain negative eigenvalues[cite: 864, 866, 869, 894].



## 4. Key Findings: The Insufficiency of Symmetry
* [cite_start]**Persistent Negative Eigenvalues**: During training, the model fails to eliminate negative eigenvalues, which correspond to non-physical "repulsive" interactions[cite: 277, 278, 866, 894].
* [cite_start]**Stagnation**: Performance typically stagnates at intermediate accuracy levels ($\approx 50\%$), failing to reach the semantic solution robustly[cite: 866].
* [cite_start]**Parameter Parsimony vs. Geometry**: Although this model has the fewest parameters among full-rank variants, it performs poorly, proving that parameter reduction without PSD geometry is insufficient[cite: 273].

## 5. Optimization Landscape: Parasitic Minima
* [cite_start]**Saliency Attractor**: Without the PSD constraint, the gradient dynamics are often insufficient to escape the basin of attraction of the Saliency Trap (vertical columns)[cite: 867, 869, 895].
* [cite_start]**Instability**: The optimization landscape is bimodal; while the model can theoretically learn the task, it lacks the robustness of the Tied architecture[cite: 896, 897].

## 6. Usage
To run the training script for the Symmetric Unfactorized architecture:
```bash
python train_histogram.py --arch sym_unfactorized --dim 100 --vocab 15 --seq_len 30 --lr 0.01