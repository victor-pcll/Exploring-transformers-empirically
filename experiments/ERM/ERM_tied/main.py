import sys
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Importations locales
from src.utils import init_torch, get_logger
from src.experiment import run_experiment

if __name__ == "__main__":
    
    # --- Arguments & Environnement ---
    if len(sys.argv) > 1:
        run_index = int(sys.argv[1])
    else:
        run_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Dossier de résultats : argument 2 ou défaut
    run_dir = sys.argv[2] if len(sys.argv) > 2 else f"./results/run_{job_id}"
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Logger & Seed ---
    logger, log_file = get_logger(run_dir, run_index, verbose=True)
    init_torch(42)
    
    logger.info(f"🖥 Device: {device} | Run index: {run_index} | Job ID: {job_id}")

    # --- Configuration ---
    config = {
        "D": 100,
        "L": 2,
        "rho": 1.0,
        "rho_star": 0.5,
        "beta": 1.0,
        "lam_list": [0.1, 0.01, 0.001, 0.0001, 0.00001],
        "Delta_list": [0.0],
        "Delta_in": 0.5,
        "samples": 16,
        "T": 10000,
        "learning_rate": 0.1,
        "norm_init": 1.0,
        "tol": 1e-6,
        "N_test": 2000,
        "alpha_start": 0.01,
        "alpha_end": 1.0,
        "alpha_steps": 15
    }

    alpha_list = np.linspace(config["alpha_start"], config["alpha_end"], config["alpha_steps"])

    # --- Lancement de l'expérience ---
    logger.info("========================================\n🧪 EXPERIMENT START\n----------------------------------------")
    
    df_results = run_experiment(alpha_list=alpha_list,
                                base_dir=run_dir,
                                run_index=run_index,
                                D=config["D"],
                                L=config["L"],
                                rho=config["rho"],
                                rho_star=config["rho_star"],
                                beta=config["beta"],
                                lam_list=config["lam_list"],
                                Delta_list=config["Delta_list"],
                                Delta_in=config["Delta_in"],
                                samples=config["samples"],
                                T=config["T"],
                                learning_rate=config["learning_rate"],
                                norm_init=config["norm_init"],
                                tol=config["tol"],
                                N_test=config["N_test"],
                                device=device,
                                logger=logger)

    # --- Sauvegarde de la Configuration ---
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

    # --- Mise à jour du fichier Summary global ---
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

    # --- Fin ---
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