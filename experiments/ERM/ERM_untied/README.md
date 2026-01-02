---

### 2. README pour le cas Synthetic : Untied Attention

```markdown
# Synthetic Experiment: Untied Attention (Teacher-Student)

This directory focuses on the learning dynamics of **Untied (Asymmetric Factorized)** attention mechanisms in high dimensions.

## 1. Overview
[cite_start]We evaluate the performance of factorized query-key interactions ($W_Q W_K^\top$)[cite: 63, 75]. [cite_start]This setting investigates how low-rank structural priors influence generalization without the explicit symmetry of the Tied model[cite: 26, 79].

## 2. Mathematical Setup
* [cite_start]**Parameterization**: The student approximates the teacher using $S = \frac{1}{\sqrt{rd}}W_Q W_K^\top$[cite: 75, 77].
* [cite_start]**Over-parameterization**: We specifically test the regime where the student's width ratio $\rho$ exceeds the teacher's rank ratio $\rho^*$ ($\rho > \rho^*$)[cite: 81, 568].
* [cite_start]**Targets**: Generated via a factorized asymmetric teacher to isolate the optimization dynamics of the $QK^\top$ structure[cite: 79, 80].

## 3. Key Findings
* [cite_start]**Synchronized Peak**: The Untied architecture exhibits a generalization peak (cusp) at $\alpha \in [0.5, 0.6]$, stable across various width ratios[cite: 118, 122, 123].
* [cite_start]**Robustness**: The model shows remarkable robustness to over-parameterization; performance does not degrade even when $\rho = 1.5$[cite: 590, 592].
* [cite_start]**Implicit Regularization**: Factorization acts as a powerful prior, outperforming the Unfactorized baseline by biasing the search toward generalizable low-rank solutions[cite: 285, 593].

## 4. Usage
To run the synthetic untied experiment:
```bash
python run_synthetic.py --arch untied --rho_teacher 0.5 --rho_student 1.0 --noise 0.5