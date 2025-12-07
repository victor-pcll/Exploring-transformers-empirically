# ADR 004 - How to run with SLURM

This experiment is designed to be executed either locally or on a SLURM cluster.
All parameters are configured inside the main() function or provided via command-line arguments / SLURM environment variables.

A. Running locally
python experiments/BO/run_experiment.py <run_index>
	•	run_index is an integer identifying this run (useful when aggregating results).
	•	If omitted, it defaults to SLURM_ARRAY_TASK_ID or 0.

Example:
python experiments/BO/run_experiment.py 0
All outputs will be written to:
./results/run_<timestamp>/
B. Running on a SLURM cluster

To launch a single job:
sbatch run_bo.sbatch
To launch an array of experiments:
sbatch --array=0-14 run_bo.sbatch
The script automatically reads:
	•	SLURM_ARRAY_TASK_ID → used as run_index
	•	SLURM_JOB_ID → used to name the results directory

Output directory example:
/home/<user>/tpiv-simulations/results/run_<SLURM_JOB_ID>/

C. Output location

Each run produces:
results/
  run_<JOB_ID>/
    logs_<run_index>.csv
    summary.csv
    config.csv
    experiment.txt
    W_runs_<run_index>.pkl
    config_used.pkl
    results.pkl

D. Reproducing an experiment

To ensure reproducibility:
	•	Random seeds for NumPy and PyTorch are fixed to 42.
	•	Teacher weights are regenerated deterministically at each run.
	•	All configurations and results are stored in the run folder.