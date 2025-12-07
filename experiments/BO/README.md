# Bayes-Optimal Student–Teacher Experimentation

## 1. Overview

This experiment studies a student–teacher framework in a high-dimensional setting, where a student network learns to approximate the representations produced by a randomly initialized teacher.
The objective is to analyze:
	•	the structural reconstruction of the teacher weights,
	•	generalization performance,
	•	the effect of the dimension ratio \rho = R/D,
	•	the effect of the sampling density \alpha = N / D^2,
	•	and the role of regularization and input noise.

The setup allows us to identify phase transitions in wide linearized attention models, especially through the comparison of correlation matrices derived from student and teacher weights.

⸻

## 2. Model

Both the teacher and the student use a single-layer attention-like architecture:
	•	Input: x \in \mathbb{R}^{N \times L \times D}
	•	Weights: W \in \mathbb{R}^{R \times D}

The attention scores are computed as:

A_{ij}
=
\frac{1}{\sqrt{R}} x_i W^\top W x_j^\top
-
\frac{\|W\|^2}{\sqrt{R D^2}} \delta_{ij},

and the output distribution is:

y = \mathrm{Softmax}(\beta A).

The teacher network is sampled once for each pair (\alpha,\rho), while the student is trained through gradient descent to match the teacher’s outputs.

⸻

## 3. Data Generation

For each experiment:
	•	We generate N = \alpha D^2 training samples.
	•	Inputs x are i.i.d. standard normal tensors.
	•	Labels y are produced by the teacher, with optional input noise controlled by \Delta_{\text{in}}.

⸻

## 4. Student Training

The student minimizes the objective:

\mathcal{L}
=
\underbrace{\sum (y_{\text{student}} - y_{\text{teacher}})^2}_{\text{data loss}}
+
\underbrace{\lambda \|W_{\text{student}}\|^2}_{\text{regularization}}.

Training stops early when the loss variation is below a predefined tolerance.

⸻

## 5. Metrics

For each combination of (\alpha, \rho), multiple runs are performed. The following quantities are recorded:

a) Structural reconstruction error

We compare the normalized correlation matrices:

S = \frac{W^\top W}{\sqrt{R D}},

via the mean squared error:

\text{MSE}(S_\text{student}, S_\text{teacher}).

b) Prediction error

On an independent test set:
	•	Noise-free teacher:

\|y_{\text{student}} - y_{\text{teacher}}\|^2 / D^2
	•	Noisy teacher:

\|y_{\text{student}} - y_{\text{teacher noisy}}\|^2 / D^2

c) Training losses
	•	mean training data loss
	•	mean regularization loss
	•	mean total loss

d) Final weights

All student weight matrices W from each run are saved for later analysis.

⸻

## 6. Output Files

For each run (indexed via SLURM or manually), the script produces:
	•	logs_<run_index>.csv — numerical metrics for each (\alpha, \rho),
	•	W_runs_<run_index>.pkl — list of all student weight matrices,
	•	config.csv — global experiment configuration,
	•	summary.csv — aggregated results across runs,
	•	experiment.txt — detailed log of the run.
