# ADR 004 — Running Experiments with SLURM

## Status

Accepted

## Context

Our experiment pipeline for TPIV simulations must run both locally (for debugging and development) and on SLURM-based HPC clusters (for large-scale sweeps over hyperparameters such as α, ρ, λ, etc.).

Running on SLURM introduces constraints:
	•	Jobs must support array execution.
	•	The experiment must automatically detect:
	•	the run index (via command-line or SLURM environment variable),
	•	the job identifier (SLURM_JOB_ID),
	•	Results must be stored in isolated folders per job and per run index.
	•	Deterministic reproducibility is required across local and SLURM runs.

To ensure consistency, we adopt a unified way of executing experiments and storing outputs.

⸻

## Decision

We standardize the execution workflow as follows:
	1.	The experiment script (run_experiment.py) accepts an optional run_index argument.
	•	If omitted, the script automatically falls back to:
	•	SLURM_ARRAY_TASK_ID if running under SLURM.
	•	0 otherwise.
	2.	Each SLURM job creates its own results directory:
  ```
  results/
  run_<timestamp_or_jobid>/
  ```
	3.	Each run inside a SLURM array produces its own output files:
  ```
  logs_<run_index>.csv
  summary.csv
  config.csv
  experiment.txt
  W_runs_<run_index>.pkl
  config_used.pkl
  results.pkl
  ```
  4.	Reproducibility is enforced by fixing:
	•	NumPy seed → 42
	•	PyTorch seed → 42
	•	Deterministic generation of teacher weights
	5.	The SLURM submit script (run_bo.sbatch) exposes two execution modes:
	•	Single job
	•	Job array (e.g. sweeping over 15 values of α)

This decision ensures stable execution across environments, automatic job indexing, safe parallelization, and high reproducibility.

⸻

## Details

### A. Running Locally

Command:
```
python experiments/BO/run_experiment.py <run_index>
```
	•	run_index is an integer identifying this run.
	•	If omitted, it defaults to:
	•	SLURM_ARRAY_TASK_ID (if running on SLURM),
	•	0 locally.

Example:
```
python experiments/BO/run_experiment.py 0
```
Output location (local machine):
```
./results/run_<timestamp>/
```

### B. Running on SLURM

Single job
```
sbatch run_bo.sbatch
```
Array of experiments
```
sbatch --array=0-14 run_bo.sbatch
```
The SLURM script automatically retrieves:
	•	SLURM_ARRAY_TASK_ID → used as run_index
	•	SLURM_JOB_ID → used to name the output directory

Output location (SLURM cluster):
```
/home/<user>/tpiv-simulations/results/run_<SLURM_JOB_ID>/
```

### C. Output Structure

Each run writes:
```
results/
  run_<JOB_ID or timestamp>/
    ├── logs_<run_index>.csv
    ├── summary.csv
    ├── config.csv
    ├── experiment.txt
    ├── W_runs_<run_index>.pkl
    ├── config_used.pkl
    ├── results.pkl
```

### D. Reproducing an Experiment

Reproducibility is ensured by:
	•	Fixing seeds for NumPy and PyTorch (42)
	•	Regenerating teacher weights deterministically
	•	Saving:
	•	Full configuration (config.csv)
	•	Used configuration (config_used.pkl)
	•	Final results (results.pkl)
	•	Learned weights (W_runs_*.pkl)

Given any run directory, the full experiment can be replayed exactly.

⸻

## Consequences

Pros
	•	Consistent workflow across local and SLURM environments
	•	Automatic run indexing
	•	Full reproducibility
	•	Traceability via saved configs and logs
	•	Safe parallel execution using SLURM arrays

Cons
	•	Requires maintaining consistent folder structure
	•	Requires SLURM-specific environment variables when running on HPC

⸻

## Alternatives Considered
	1.	Separate scripts for local and SLURM execution
→ Rejected due to duplication and risk of divergence.
	2.	Manual naming of run folders
→ Rejected because SLURM_JOB_ID is safer, collision-free, and traceable.
	3.	Not using job arrays
→ Rejected since sweeping over α, ρ, λ becomes inefficient.