# ERM-Tied Student–Teacher Experiment

## 1. Overview

This experiment implements a tied-weight student–teacher framework in high dimension.
A randomly initialized teacher network generates labels for Gaussian inputs, and a student network is trained to recover both:
	•	the output behavior of the teacher,
	•	and the structural correlation matrix of its weight tensor.

The model is a simplified, linearized attention mechanism with a single weight matrix W \in \mathbb{R}^{R \times D}, where:
	•	D: input dimension
	•	R: hidden dimension
	•	L: number of tokens
	•	\rho = R/D: width ratio
	•	\alpha = N/D^2: sample density

The goal is to study how the student reconstructs the teacher as a function of:
	•	sampling load \alpha,
	•	width mismatch \rho, \rho_\*,
	•	regularization \lambda,
	•	and noise level \Delta_\text{in}.

⸻

## 2. Model

Both networks follow the same architecture:

Forward computation

Given input x \in \mathbb{R}^{N \times L \times D}:
	1.	Linear mapping
xW^\top / \sqrt{D}
	2.	Attention score
A_{ij} = \frac{x_i W^\top W x_j^\top}{\sqrt{R}} -
\frac{\|W\|^2}{\sqrt{R} D^2} \delta_{ij}
	3.	Optional symmetric Gaussian noise controlled by \Delta_\text{in}
	4.	Softmax at temperature \beta

⸻

## 3. Training Procedure

The student minimizes:

\mathcal{L} =
\|y_{\text{student}} - y_{\text{teacher}}\|^2 +
\lambda_\text{eff} \|W_{\text{student}}\|^2,
\qquad
\lambda_\text{eff} = \frac{\lambda}{\sqrt{\rho}}.

Training stops early when the loss variation drops below a tolerance.

⸻

## 4. Metrics

The following quantities are recorded for every (\alpha, \lambda):

(a) Structural reconstruction

We compute:

S(W) = \frac{W^\top W}{\sqrt{R D}}

and record:

\text{MSE} = \frac{1}{D} \| S_\text{student} - S_\text{teacher} \|^2.

(b) Label errors
	•	without teacher noise
\|y_s - y_t\|^2 / D^2
	•	with teacher noise
\|y_s - y_{t,\text{noisy}}\|^2 / D^2

(c) Training losses
	•	mean empirical loss
	•	mean regularization loss
	•	total loss

(d) Complete student weights

Stored as .pkl files for later analysis.

⸻

## 5. Output Files

Each experiment run produces:
results/run_<JOB_ID>/
    logs_<run_index>.csv
    summary.csv
    config.csv
    experiment_<run_index>_<JOB_ID>.log
    W_runs_<run_index>.pkl