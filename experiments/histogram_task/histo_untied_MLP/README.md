# Experiment: Untied Attention on the Structured Discrete Histogram Task

This directory contains the implementation and experimental scripts for analyzing the learning dynamics of **Untied (Factorized) Attention** ($S = W_Q W_K^\top$) on the Histogram Task.

## 1. Overview
The goal of this experiment is to evaluate the learning behavior of factorized attention without an explicit symmetry or PSD constraint. We investigate how the product parameterization influences optimization despite the model possessing a higher parameter count ($2d^2$) compared to other architectures.

## 2. Task Description: The Histogram Task
The objective is to perform exact counting of discrete tokens within a sequence.
* **Input**: Sequences of length $T=30$ with tokens from a vocabulary $L=15$.
* **Target**: Predict the total count $y_i$ of each token $x_i$ present in the sequence.
* **Data Properties**: The task resides in a "dense" regime where token repetitions are statistically significant ($P(X \ge 4) \ge 0.1$).

## 3. Architecture: Untied (Factorized) Student
* **Attention Kernel**: Parameterized as $S = \frac{1}{\sqrt{rd}}W_Q W_K^\top$ with $W_Q \ne W_K$.
* **Capacity**: In the $\rho=1$ setting, the model possesses full-rank capacity.
* **Readout**: Context vectors are processed by a two-layer MLP to produce frequency class logits.

## 4. Implicit Bias: Spontaneous Rank Reduction
A key finding in this experiment is the **spontaneous rank reduction** during optimization[cite: 220, 824].
* **Observation**: Although initialized at full rank ($r=d$), $W_Q$ and $W_K$ synchronously compress their numerical rank as training progresses.
* **Mechanism**: Minimizing the $L_2$ norm of individual factors approximates the minimization of the nuclear norm of the resulting product $S$, promoting low-rank solutions.
* **Conclusion**: This confirms that low-rank structure is an emergent property of the factorization rather than a hard architectural constraint.

## 5. Failure Mode: The Saliency Trap
Unlike the Tied architecture, the Untied model oftensettles into a suboptimal performance plateau (~70% accuracy).
* **Saliency Trap**: The attention maps exhibit a hybrid morphology where semantic diagonal blocks are accompanied by residual vertical columns.
* **Spectral Collapse**: The model converges to a regime where attention weights depend solely on the statistical salience of the key token $x_j$, becoming context-independent regarding the query $x_i$.
* **Inductive Bias Gap**: The absence of an explicit PSD constraint leaves the model vulnerable to these non-generalizable attractors.