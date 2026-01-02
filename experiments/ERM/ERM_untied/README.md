### 2. Synthetic Experiment: Untied Attention (Teacher-Student)

This directory focuses on the learning dynamics of **Untied (Asymmetric Factorized)** attention mechanisms in high dimensions.

## 1. Overview
We evaluate the performance of factorized query-key interactions ($W_Q W_K^\top$). This setting investigates how low-rank structural priors influence generalization without the explicit symmetry of the Tied model.

## 2. Mathematical Setup
* **Parameterization**: The student approximates the teacher using $S = \frac{1}{\sqrt{rd}}W_Q W_K^\top$.
* **Over-parameterization**: We specifically test the regime where the student's width ratio $\rho$ exceeds the teacher's rank ratio $\rho^*$ ($\rho > \rho^*$).
* **Targets**: Generated via a factorized asymmetric teacher to isolate the optimization dynamics of the $QK^\top$ structure.

## 3. Key Findings
* **Synchronized Peak**: The Untied architecture exhibits a generalization peak (cusp) at $\alpha \in [0.5, 0.6]$, stable across various width ratios.
* **Robustness**: The model shows remarkable robustness to over-parameterization; performance does not degrade even when $\rho = 1.5$.
* **Implicit Regularization**: Factorization acts as a powerful prior, outperforming the Unfactorized baseline by biasing the search toward generalizable low-rank solutions.