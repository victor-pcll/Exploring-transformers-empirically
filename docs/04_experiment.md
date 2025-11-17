# Experiment Orchestration

The main experiment pipeline is defined in `run_experiment(config)`.

### What it does
- Loops over values of lambda (regularization parameter)
- For each value:
  - Runs multiple student trainings (`samples`)
  - Evaluates test error
  - Stores:
    - attention matrices
    - learned weights
    - sample sequences
    - histogram predictions

### Logging
A logger is created for each run:
- writes logs to a file
- prints summary in console

### Saved artifacts
- `logs_<run_index>.csv`: training metrics
- `attn_run<index>.pkl`: sample attention matrices
- `test_attn_run<index>.pkl`: aggregated test results
- `W_runs_<index>.pkl`: weight matrices
- `summary.csv`: full experiment summary across all runs
- `config.csv`: parameters used for each run