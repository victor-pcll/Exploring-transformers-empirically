# Experiment: Untied Attention on the Structured Discrete Histogram Task

This directory contains the implementation and experimental scripts for analyzing the learning dynamics of **Untied (Factorized) Attention** ($S = W_Q W_K^\top$) on the Histogram Task.

## 1. Overview
[cite_start]The goal of this experiment is to evaluate the learning behavior of factorized attention without an explicit symmetry or PSD constraint[cite: 766]. [cite_start]We investigate how the product parameterization influences optimization despite the model possessing a higher parameter count ($2d^2$) compared to other architectures[cite: 271, 275].

## 2. Task Description: The Histogram Task
[cite_start]The objective is to perform exact counting of discrete tokens within a sequence[cite: 132, 136].
* [cite_start]**Input**: Sequences of length $T=30$ with tokens from a vocabulary $L=15$[cite: 266, 725].
* [cite_start]**Target**: Predict the total count $y_i$ of each token $x_i$ present in the sequence[cite: 137, 138].
* [cite_start]**Data Properties**: The task resides in a "dense" regime where token repetitions are statistically significant ($P(X \ge 4) \ge 0.1$)[cite: 686, 698].

## 3. Architecture: Untied (Factorized) Student
* [cite_start]**Attention Kernel**: Parameterized as $S = \frac{1}{\sqrt{rd}}W_Q W_K^\top$ with $W_Q \ne W_K$[cite: 164, 766].
* [cite_start]**Capacity**: In the $\rho=1$ setting, the model possesses full-rank capacity[cite: 764, 765].
* [cite_start]**Readout**: Context vectors are processed by a two-layer MLP to produce frequency class logits[cite: 175, 177].



## 4. Implicit Bias: Spontaneous Rank Reduction
[cite_start]A key finding in this experiment is the **spontaneous rank reduction** during optimization[cite: 220, 824].
* [cite_start]**Observation**: Although initialized at full rank ($r=d$), $W_Q$ and $W_K$ synchronously compress their numerical rank as training progresses[cite: 824, 825].
* [cite_start]**Mechanism**: Minimizing the $L_2$ norm of individual factors approximates the minimization of the nuclear norm of the resulting product $S$, promoting low-rank solutions[cite: 828].
* [cite_start]**Conclusion**: This confirms that low-rank structure is an emergent property of the factorization rather than a hard architectural constraint[cite: 221, 823].

## 5. Failure Mode: The Saliency Trap
[cite_start]Unlike the Tied architecture, the Untied model oftensettles into a suboptimal performance plateau (~70% accuracy)[cite: 218].
* [cite_start]**Saliency Trap**: The attention maps exhibit a hybrid morphology where semantic diagonal blocks are accompanied by residual vertical columns[cite: 792].
* [cite_start]**Spectral Collapse**: The model converges to a regime where attention weights depend solely on the statistical salience of the key token $x_j$, becoming context-independent regarding the query $x_i$[cite: 796].
* [cite_start]**Inductive Bias Gap**: The absence of an explicit PSD constraint leaves the model vulnerable to these non-generalizable attractors[cite: 219, 793].

## 6. Usage
To run the training script for the Untied architecture:
```bash
python train_histogram.py --arch untied --rank_ratio 1.0 --dim 100 --vocab 15 --seq_len 30 --lr 0.01