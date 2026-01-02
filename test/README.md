# Tests and Mathematical Validations

This directory contains utility scripts and notebooks used to validate the theoretical foundations of the project and ensure the stability of the implementation before large-scale deployment.

## 1. Histogram Data Generation & Statistics (`.ipynb`)
These tests focus on validating the "Structured Discrete" dataset generator to ensure the task is non-trivial and follows the theoretical constraints described in Appendices F and G.

* **Controlled Composition Validation**: Verification of the random integer partitioning and shuffling procedure used to ensure that position carries no information about token counts.
* **Binomial Count Distribution**: Empirical check that the number of occurrences $X$ of a token follows the expected Binomial distribution $Binomial(n=T, p=1/L)$.
* **Hardness Criterion ($P(X \ge 4)$)**: Numerical evaluation of the probability that a token appears at least 4 times. This ensures the dataset resides in the "dense" regime ($T/L=2$) where the model must learn genuine counting rather than simple identity mapping.

## 2. Cluster Validation Script (`test_cluster.py`)
A lightweight script designed to test the compute environment (SLURM/Local Cluster) before launching intensive training runs.

* **Resource Check**: Verifies the availability of CPU/GPU resources and ensures the environment is correctly configured for the high-dimensional scaling limit ($d \to \infty$).
* **Optimizer Stability**: Performs a short training loop to validate the Adam optimizer settings ($\eta$, $tol$) and the weight initialization scales ($Var(S_{ij}) \approx 1/d$).
* **Seed Reproducibility**: Confirms that weight initialization and data generation are consistent across independent seeds to ensure statistical significance.

## 3. Directory Structure
* `notebook.ipynb`: Notebook for visualizing spectral distributions and quadratic form moments.
* `test_cluster.py`: Entry point for cluster resource and environment validation.
* `test_cluster.sbatch`: Cluster validation.