# Synthetic Experiment: Tied Attention (Teacher-Student)

This directory implements the synthetic high-dimensional analysis for **Tied (Factorized)** attention mechanisms.

## 1. Overview
The goal is to analyze the generalization performance of a symmetric Positive Semi-Definite (PSD) prior in a controlled setting. We study the student's ability to recover a teacher's interaction matrix $S^*$ under Gaussian inputs.

## 2. Mathematical Setup
* **High-Dimensional Limit**: We operate where $d \to \infty$ and $n \propto d^2$, with a fixed sample complexity $\alpha = n/d^2$.
* **Teacher/Student Model**: Both are parameterized as $S = \frac{1}{\sqrt{rd}}WW^\top$, enforcing symmetry and PSD geometry.
* **Input Distribution**: Independent standard Gaussian tokens $x_a^\mu \sim \mathcal{N}(0, \mathbb{I}_d)$.
* **Target Generation**: Scores are computed as centered quadratic forms $h_{ab}^\mu$ with additive noise $\Delta$.

## 3. Key Findings
* **Learning Acceleration**: The PSD constraint significantly shifts the interpolation threshold to a lower sample complexity ($\alpha \approx 0.2$) compared to asymmetric models.
* **Data Efficiency**: Enforcing the $WW^\top$ structure reduces the effective model capacity, leading to superior data efficiency in the low-$\alpha$ regime.
* **Double Descent**: Under noisy Empirical Risk Minimization (ERM), a distinct generalization peak emerges around $\alpha \approx 0.2$ when regularization is weak.