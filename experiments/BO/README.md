# Bayes-Optimal Student–Teacher Experimentation

## 1. Overview

This experiment studies a student–teacher framework in a high-dimensional setting, where a student network learns to approximate the outputs of a randomly initialized teacher.  

**Main objectives:**

- Structural reconstruction of the teacher weights.
- Generalization performance on independent test data.
- The effect of the dimension ratio $\rho = R/D$.
- The effect of the sampling density $\alpha = N / D^2$.
- The role of regularization ($\lambda$) and input noise ($\Delta_{\text{in}}$).

This setup allows the identification of phase transitions in wide linearized attention models by comparing the correlation matrices derived from student and teacher weights.

---

## 2. Model

Both the teacher and student use a **single-layer attention-like architecture**:

- **Input:** $x \in \mathbb{R}^{N \times L \times D}$  
- **Weights:** $W \in \mathbb{R}^{R \times D}$

The attention scores are computed as:

$$
A_{ij} = \frac{1}{\sqrt{R}} x_i W^\top W x_j^\top - \frac{\|W\|^2}{\sqrt{R D^2}} \delta_{ij},
$$

and the output distribution is

$$
y = \mathrm{Softmax}(\beta A).
$$

- The **teacher network** is sampled once per pair $(\alpha, \rho)$.  
- The **student network** is trained via gradient descent to match the teacher outputs.

---

## 3. Data Generation

For each experiment:

- Generate $N = \alpha D^2$ training samples.  
- Inputs $x$ are i.i.d. standard normal tensors.  
- Labels $y$ are produced by the teacher, optionally including input noise $\Delta_{\text{in}}$.

---

## 4. Student Training

The student minimizes the regularized mean-squared error objective:

$$
\mathcal{L} = 
\underbrace{\sum (y_{\text{student}} - y_{\text{teacher}})^2}_{\text{data loss}} 
+ 
\underbrace{\lambda \|W_{\text{student}}\|^2}_{\text{regularization}}.
$$

- Optimization uses **Adam** or standard gradient descent.  
- Early stopping occurs if the loss variation falls below a predefined tolerance threshold.

---

## 5. Metrics

For each $(\alpha, \rho)$ combination, multiple independent runs are performed. Recorded quantities include:

1. **Structural reconstruction error**  
   - Normalized correlation matrices:  
     $$
     S = \frac{W^\top W}{\sqrt{R D}}
     $$  
   - Measured via mean squared error: $\text{MSE}(S_\text{student}, S_\text{teacher})$.

2. **Prediction error**  
   - On an independent test set:  
     - Noise-free teacher: $\|y_{\text{student}} - y_{\text{teacher}}\|^2 / D^2$  
     - Noisy teacher: $\|y_{\text{student}} - y_{\text{teacher noisy}}\|^2 / D^2$

3. **Training losses**  
   - Mean training data loss  
   - Mean regularization loss  
   - Mean total loss

4. **Final weights**  
   - All student weight matrices $W$ from each run are saved for downstream analysis.

---

## 6. Output Files

Each run (indexed via **SLURM** or manually) produces:

- `logs_<run_index>.csv` — numerical metrics for each $(\alpha, \rho)$  
- `W_runs_<run_index>.pkl` — list of all student weight matrices  
- `config.csv` — global experiment configuration  
- `summary.csv` — aggregated results across runs  
- `experiment.txt` — detailed log of the run

---

## 7. Notes

- The experiment supports reproducibility by storing all seeds, configurations, and student weights.  
- Input noise ($\Delta_{\text{in}}$) and weight regularization ($\lambda$) can be tuned to explore different learning regimes.  
- Phase transitions are typically observed as sharp changes in MSE or prediction error with respect to $\alpha$ and $\rho$.  
- For further reference, see [Boncoraglio et al., 2025](https://arxiv.org/abs/...) for the original Bayes-optimal study.

---

## 8. Quick Start

Clone the repository:

```bash
git clone https://github.com/yourusername/tpiv-simulations.git
sbatch tpiv-simulations/experiments/BO/job.sbatch
```
Results will be saved in ./results/run_<timestamp>/ with:
- CSV logs (logs_<run_index>.csv)
- Pickled weight matrices (W_runs_<run_index>.pkl)
- Configuration file (config.csv)
- Summary CSV (summary.csv)
- Detailed log (experiment.txt)

You can adjust experiment parameters by modifying src/main.py or the configuration dictionary within.

⸻

9. Requirements
- Python 3.8+
- PyTorch 2.x
- NumPy
- Pandas