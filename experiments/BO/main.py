import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from datetime import datetime

from src.utils.seed import init_torch
from src.utils.config_utils import convert_numeric_config
from src.utils.run_dir import create_run_dir, get_job_id
from src.utils.logger import init_logging, log_header, log_footer
from src.utils.saver import save_pickle
from src.experiment import run_experiment


def update_summary_csv(run_dir, run_idx):
    summary_csv = os.path.join(run_dir, "summary.csv")
    logs_csv = os.path.join(run_dir, f"logs_{run_idx}.csv")

    if not os.path.isfile(logs_csv):
        logging.warning(f"{logs_csv} not found")
        return

    try:
        df = pd.read_csv(logs_csv)
        df.to_csv(
            summary_csv,
            mode='a' if os.path.isfile(summary_csv) else 'w',
            header=not os.path.isfile(summary_csv),
            index=False
        )
        logging.info(f"Updated summary CSV: {summary_csv}")
    except Exception as e:
        logging.warning(f"Failed to update summary CSV: {e}")

def update_config_csv(run_dir, config, run_idx):
    config_csv = os.path.join(run_dir, "configs.csv")

    # On ajoute le run index dans la config
    flat = {**config, "run_index": run_idx}

    df = pd.DataFrame([flat])

    try:
        df.to_csv(
            config_csv,
            mode="a" if os.path.isfile(config_csv) else "w",
            header=not os.path.isfile(config_csv),
            index=False
        )
        logging.info(f"Updated config CSV: {config_csv}")
    except Exception as e:
        logging.warning(f"Failed to update config CSV: {e}")


if __name__ == "__main__":

    # ----- Arguments -----
    run_idx = int(sys.argv[1]) if len(sys.argv) > 1 else int(
        os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    )
    run_dir = sys.argv[2] if len(sys.argv) > 2 else create_run_dir()
    job_id = get_job_id()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- CONFIG -----
    config = {
        "D": 100,
        "L": 2,
        "beta": 1.0,
        "lam": 0.0,     
        "Delta_in": 0.0,
        "samples": 16,
        "T": 10000,
        "lr": 0.01,
        "norm_init": 1.0,
        "tol": 1e-6,
        "N_test": 2000,
        "rho_list": [0.2, 0.3, 0.4, 0.5, 1.0],
        "alpha_start": 0.1,
        "alpha_end": 0.4,
        "alpha_steps": 15,
        "base_dir": run_dir,
        "verbose": True,
    }

    # ----- Seed & numeric conversion -----
    init_torch(42, verbose=config["verbose"])
    config = convert_numeric_config(config, verbose=config["verbose"])

    # ----- Alpha selection for this job -----
    alpha_list = np.linspace(config["alpha_start"], config["alpha_end"], config["alpha_steps"])
    alpha = float(alpha_list[run_idx])
    config["alpha_list"] = [alpha]
    config["alpha"] = alpha

    # ----- Logging -----
    os.makedirs(run_dir, exist_ok=True)
    init_logging(run_dir, verbose=config["verbose"])
    log_header(run_idx, job_id, device, run_dir, config)

    # ----- EXPERIMENT -----
    results = run_experiment(config, run_index=run_idx)

    update_config_csv(run_dir, config, run_idx)
    save_pickle(results, f"{run_dir}/results_{run_idx}.pkl")

    # Save logs CSV
    logs_csv = os.path.join(run_dir, f"logs_{run_idx}.csv")
    if isinstance(results, pd.DataFrame):
        results.to_csv(logs_csv, index=False)
    else:
        df_results = pd.DataFrame(results)
        df_results.to_csv(logs_csv, index=False)

    # ----- UPDATE SUMMARY -----
    update_summary_csv(run_dir, run_idx)

    log_footer(run_dir, run_idx)