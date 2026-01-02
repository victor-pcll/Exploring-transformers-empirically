# Experiment: Unfactorized Attention on the Structured Discrete Histogram Task

This directory contains the implementation and experimental scripts for analyzing the learning dynamics of **Unfactorized Attention** ($S \in \mathbb{R}^{d \times d}$) on the Histogram Task.

## 1. Overview
This experiment serves as a baseline to evaluate the performance of an unstructured attention mechanism with $d^2$ independent degrees of freedom. We investigate why high-capacity models without structural priors fail to recover the algorithmic counting rule, despite achieving near-zero training loss.

## 2. Task Description: The Histogram Task
The objective is to predict token frequencies in a discrete sequence.
* **Input**: Sequences of length $T=30$ with a vocabulary size $L=15$.
* **Target**: An exact match of the count $y_i$ for every token $x_i$ in the sequence.
* **Complexity**: The task requires identifying pairwise token identities, which theoretically implies a low-rank similarity structure.

## 3. Architecture: Unfactorized Student
* **Attention Kernel**: A full-rank matrix $S$ initialized with i.i.d. Gaussian entries $\mathcal{N}(0, 1/d)$.
* **Capacity**: The model utilizes its full $d^2$ degrees of freedom to interpolate training data.
* **Readout**: Standard two-layer MLP to map context vectors to frequency classes.

## 4. Observations: Overfitting and Lack of Regularization
* **Full Rank Maintenance**: Unlike factorized models, the Unfactorized architecture maintains its initial full numerical rank ($r \approx 100$) throughout the entire optimization process.
* **Spectral Bulk**: The eigenvalue distribution remains diffuse, failing to isolate a clear signal from the noise bulk.
* **Noise Interpolation**: The model successfully minimizes training loss ($< 10^{-4}$) but fails to generalize, prioritizing noise interpolation over structural learning.

## 5. Functional Failure: The Saliency Trap
* **Morphology**: Attention maps consistently exhibit vertical striations (columns), indicating a complete loss of contextual relevance.
* **Global Attractor**: The model settled into a 'Saliency' minimum, where attention is driven by global token frequencies rather than pairwise context.
* **High Variance**: Optimization is unstable across seeds, often collapsing into different suboptimal attractors.

## 6. Usage
To run the training script for the Unfactorized architecture:
```bash
python train_histogram.py --arch unfactorized --dim 100 --vocab 15 --seq_len 30 --lr 0.01