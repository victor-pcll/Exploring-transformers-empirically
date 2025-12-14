from datetime import datetime
import torch
import numpy as np
import sys
import os
import pickle
from src.utils.seed import init_torch
from src.utils.config_utils import convert_numeric_config
from src.utils.run_dir import create_run_dir, get_job_id
from src.utils.logger import init_logging, log_header, log_footer
from src.utils.saver import save_pickle, update_summary_csv
from src.experiment.run_experiment import run_experiment


if __name__ == "__main__":

    # ----- Arguments -----
    run_idx = int(sys.argv[1]) if len(sys.argv) > 1 else int(
        os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    )
    run_dir = sys.argv[2] if len(sys.argv) > 2 else create_run_dir()
    job_id = get_job_id()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- CONFIG (tu la gardes entièrement ici) -----
    config = {
        "verbose": False,
        "alpha_start": 0.01,
        "alpha_end": 1.0,
        "alpha_steps": 15,
        "d": 100,
        "L": 2,
        "beta": 1.0,
        "lmbda": [0.1, 0.001, 2*0.0001, 5*0.0001, 0.0001],
        "Delta_in": 0.5,
        "Delta_list": [0.0],
        "samples": 16,
        "T": 10000,
        "lr": 0.1,
        "norm_init": 1.0,
        "tol": 1e-6,
        "n_test": 2000,
    }

    # Seed & numeric conversion
    init_torch(42, verbose=config["verbose"])
    config = convert_numeric_config(config, verbose=config["verbose"])
    alpha_list = np.linspace(config["alpha_start"], config["alpha_end"], config["alpha_steps"])

    # ----- Logging -----
    init_logging(run_dir, verbose=config["verbose"])
    log_header(run_idx, job_id, device, run_dir, config)

    # ----- EXPERIMENT -----
    results, alphas = run_experiment(
        base_dir=run_dir,
        D=config["d"],
        L=config["L"],
        beta=config["beta"],
        lam_list=config["lmbda"],
        Delta_list=config["Delta_list"],
        Delta_in=config["Delta_in"],
        samples=config["samples"],
        T=config["T"],
        learning_rate=config["lr"],
        norm_init=config["norm_init"],
        tol=config["tol"],
        N_test=config["n_test"],
        verbose=config["verbose"],
        alpha_list=alpha_list,
        run_index=run_idx,
    )

    # ----- SAVE -----
    save_pickle(config, f"{run_dir}/config_used.pkl")
    save_pickle(results, f"{run_dir}/results.pkl")
    update_summary_csv(run_dir, run_idx)

    log_footer(run_dir, run_idx)