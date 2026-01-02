# Experiment: Tied Attention on the Structured Discrete Histogram Task

This directory contains the implementation and experimental scripts for analyzing the learning dynamics and inductive bias of **Tied (Factorized) Attention** applied to a structured discrete counting task.

## 1. Overview
The primary objective of this experiment is to investigate how the **Positive Semi-Definite (PSD) geometry** inherent to tied attention architectures promotes data efficiency and semantic learning. By enforcing a $S = \frac{1}{\sqrt{rd}}WW^\top$ parameterization, we evaluate the model's ability to recover a counting rule from discrete sequences.

## 2. Task Definition: The Histogram Task
The network must perform exact counting operations on discrete sequences.
* **Input**: Sequences of length $T=30$ composed of discrete tokens from a vocabulary $L=15$.
* **Target**: For each position $i$, the model predicts the total frequency of the token $x_i$ in the sequence: $y_i = \sum_{j=1}^{T} \mathbb{1}(x_i = x_j)$.
* **Complexity**: To ensure non-triviality, data is generated via a controlled random composition procedure that guarantees significant token repetitions ($P(X \ge 4) \ge 0.1$).

## 3. Architecture
The student model consists of a three-stage pipeline:
1. **Embedding Layer**: Maps discrete tokens to a continuous vector space $E \in \mathbb{R}^{L \times d}$.
2. **Tied Attention**: A single-head mechanism where the interaction matrix is factorized as $S = \frac{1}{\sqrt{rd}}WW^\top$. A zero-diagonal constraint is applied to match the generative teacher framework.
3. **MLP Readout**: A two-layer MLP that processes context vectors to output logits over $C=T$ frequency classes.



## 4. Optimization & Hyperparameters
* [cite_start]**Loss Function**: Multi-class Cross-Entropy loss[cite: 189, 197].
* [cite_start]**Regularization**: Global $L_2$ weight decay penalty $\lambda$ selected via cross-validation in the range $[10^{-5}, 10^{-1}]$[cite: 191, 267].
* [cite_start]**Optimizer**: Adam with learning rate $\eta=0.01$ and tolerance $10^{-5}$[cite: 192, 267].
* [cite_start]**Dimensions**: $d=100$, $d_{MLP}=100$, and sequence length $T=30$[cite: 266].

## 5. Key Results & Spectral Signatures
* [cite_start]**Efficiency**: The Tied model achieves high sequence accuracy (~90%) at low sample complexity ($\alpha \le 1$)[cite: 212].
* [cite_start]**Spectral Bulk**: Training leads to the progressive formation of a distinct spectral bulk in the singular value distribution of $S$, representing the recovery of the semantic signal[cite: 737, 738, 754].
* [cite_start]**Semantic Solution**: Visualizations confirm that the model learns a dot-product similarity metric ($A_{ij} \propto \mathbb{1}(x_i = x_j)$) rather than memorizing positional artifacts[cite: 681, 758].
* [cite_start]**Stability**: This architecture shows low variance across training seeds and avoids the "Saliency Trap" (vertical striations) prevalent in unstructured models[cite: 215, 850].

## 6. Usage
To run the training for the Tied architecture:
```bash
python train_histogram.py --arch tied --rank_ratio 1.0 --dim 100 --vocab 15 --seq_len 30