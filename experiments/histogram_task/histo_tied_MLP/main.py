from datetime import datetime
import torch
import numpy as np
import pandas as pd
import sys
import os

from src.utils.seed import init_torch
from src.utils.logger import setup_logger
from src.experiment.run_experiment import run_experiment

if __name__ == "__main__":
    run_index = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = sys.argv[2] if len(sys.argv) > 2 else f"./results/run_{job_id}"
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger centralisé
    logger, log_file = setup_logger(run_dir, run_index, verbose=True)

    # Seeds
    init_torch(42)
    logger.info(f"🖥 Device: {device} | Run index: {run_index} | Job ID: {job_id}")

    # Alpha list
    alpha_start = 0.1
    alpha_end = 1.0
    alpha_steps = 15
    alpha_list = np.linspace(alpha_start, alpha_end, alpha_steps).tolist()
    alpha = alpha_list[run_index] if run_index < len(alpha_list) else alpha_list[-1]

    # Configuration
    config = {
        "D": 100,
        "L": 15,
        "T" : 30,
        "d_mlp_list": [10],
        "alpha": alpha,
        "rho": 1.0,
        "beta": 1.0,
        "lambda": 0.0001,
        "samples": 20,
        "max_iter": 5000,
        "max_fine_tune_iter": 1000,
        "learning_rate": 0.01,
        "learning_rate_fine_tune": 0.01,
        "norm_init": 1.0,
        "tol": 1e-5,
        "N_test": 2000,
        "N_valid": 2000,
        "device": device,
        "logger": logger,
        "run_dir": run_dir,
        "run_index": run_index,
        "seed": 42,
        "batch_size": 64,
    }

    config["R"] = int(config["rho"] * config["D"])
    config["N_train"] = int(config["alpha"] * config["D"])  # j'ai mis n = alpha * d
    config["N_total"] = config["N_train"] + config["N_test"] + config["N_valid"]

    # --- Header log ---
    logger.info("========================================\n🧪 EXPERIMENT START\n----------------------------------------")
    df_results = run_experiment(config)
    config["run_index"] = run_index
    config_csv_path = os.path.join(run_dir, "config.csv")

    try:
        df_config = pd.DataFrame([config])
        if os.path.isfile(config_csv_path):
            df_config.to_csv(config_csv_path, mode='a', header=False, index=False)
        else:
            df_config.to_csv(config_csv_path, mode='a', header=True, index=False)
        logger.info(f"💾 Configuration saved as CSV: {config_csv_path}")
    except Exception as e:
        logger.warning(f"[ERR] Failed to save config.csv: {e}")

    try:
        logs_csv_path = os.path.join(run_dir, f"logs_{run_index}.csv")
        summary_csv_path = os.path.join(run_dir, "summary.csv")

        if os.path.isfile(logs_csv_path):
            df_logs = pd.read_csv(logs_csv_path)

            if os.path.isfile(summary_csv_path):
                df_logs.to_csv(summary_csv_path, mode="a", header=False, index=False)
            else:
                df_logs.to_csv(summary_csv_path, mode="w", header=True, index=False)

            logger.info(f"🧾 Summary CSV updated at {summary_csv_path}")
        else:
            logger.warning("[ERR] logs.csv not found — could not update summary.csv")
    except Exception as e:
        logger.warning(f"[ERR] Failed to update summary.csv: {e}")

    # --- Footer log ---
    footer_lines = [
        "----------------------------------------",
        "✅ Experiment finished successfully",
        f"📂 Results saved in: {run_dir}",
        f"📄 Logs: {log_file}",
        f"🧾 Summary: {summary_csv_path}",
        f"🧠 Config: {config_csv_path}",
        "----------------------------------------\n\n",
    ]
    logger.info("\n" + "\n".join(footer_lines))