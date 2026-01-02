### 3. README pour le cas Synthetic : Unfactorized Attention

```markdown
# Synthetic Experiment: Unfactorized Attention (Teacher-Student)

This directory contains the framework for analyzing **Unfactorized (Full-Rank)** attention layers using random matrix theory.

## 1. Overview
[cite_start]This baseline study compares unstructured $d^2$ parameterizations against factorized models[cite: 9, 26]. [cite_start]It highlights the "complexity gap" and the limits of generalization in the absence of structural priors[cite: 126, 127].

## 2. Mathematical Setup
* **Parameterization**: Both teacher $S^*$ and student $S$ are full-rank matrices with i.i.d. [cite_start]Gaussian entries $\mathcal{N}(0, 1/d)$[cite: 63, 69, 75, 89].
* [cite_start]**Spectral Scaling**: Initialized as a real Ginibre ensemble where eigenvalues follow the Circular Law[cite: 434, 435, 628].
* [cite_start]**Complexity**: The task involves $d^2$ independent degrees of freedom, representing the highest possible task entropy[cite: 119, 127].

## 3. Key Findings
* [cite_start]**Double Descent**: Like the Untied model, it exhibits a characteristic peak at $\alpha \approx 0.5$, but with a significantly higher error floor[cite: 119, 618, 619].
* [cite_start]**Spectral Evolution**: The student's spectrum evolves from a collapsed peak at zero to a semi-circular distribution (projected Circular Law), matching the teacher's spectral scale at $\alpha \approx 1.0$[cite: 641, 642, 643].
* [cite_start]**Entropy Barrier**: Persistent performance gaps in the $\alpha < 1$ regime confirm that without low-rank priors, the model prioritizes noise interpolation over structural recovery[cite: 285, 292].

## 4. Usage
To run the synthetic unfactorized experiment:
```bash
python run_synthetic.py --arch unfactorized --dim 100 --alpha_range 0.01 1.0 --noise 0.5