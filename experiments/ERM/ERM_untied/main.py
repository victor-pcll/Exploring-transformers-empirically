from datetime import datetime
import torch
import numpy as np
import os
import sys
import logging
import pandas as pd
import pickle

# Importation des fonctions depuis le paquet src
from src.utils import init_torch, convert_numeric_config, get_run_dir
from src.experiment import run_experiment

if __name__ == "__main__":

    # --- Gestion des arguments et de l'environnement SLURM ---
    run_idx = 0
    if len(sys.argv) > 1:
        # Tenter de lire l'index d'exécution
        try:
            run_idx = int(sys.argv[1])
        except ValueError:
            print(f"Avertissement: L'argument sys.argv[1] n'est pas un entier valide. Utilisation de 0.")
    else:
        # Lire l'index depuis SLURM ou utiliser 0
        run_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    # Tenter de lire le répertoire de l'exécution
    run_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    if run_dir is None:
        run_dir = get_run_dir() # Créer un nouveau répertoire si non fourni

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Configuration par défaut ---
    config = {
        "verbose": False,
        "alpha_start": 0.01,
        "alpha_end": 1.0,
        "alpha_steps": 15,
        "d": 100,
        "L": 2,
        "beta": 1.0,
        "lmbda": [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001],
        "Delta_in": 0.5,
        "Delta_list": [0.0],
        "samples": 16,
        "T": 10000,
        "lr": 0.1,
        "norm_init": 1.0,
        "tol": 1e-6,
        "n_test": 2000,
        "rho": 0.3,
        "rho_star": 0.5  
    }

    # Initialiser les graines
    init_torch(42, verbose=config.get("verbose", True))

    # Convertir les paramètres numériques (dans la copie)
    config = convert_numeric_config(config, verbose=config["verbose"])
    alpha_list = np.linspace(config["alpha_start"], config["alpha_end"], config["alpha_steps"])

    # --- Configurer logging ---
    log_file = os.path.join(run_dir, "experiment.txt")
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.INFO if config.get("verbose", True) else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    
    # --- Affichage et Log du Header ---
    header_lines = [
        "========================================",
        "        EXPERIMENT START",
        "----------------------------------------",
        f"Run index: {run_idx}",
        f"Job ID: {job_id}",
        f"Device: {device}",
        f"Results directory: {run_dir}",
        "========================================",
    ]
    header_msg = "\n".join(header_lines)
    print(header_msg)
    logging.info(header_msg)
    
    # --- Affichage et Log du Sommaire de la Configuration ---
    config_summary_lines = [
        "--- Configuration ---",
        f"D            : {config['d']}",
        f"L            : {config['L']}",
        f"alpha range  : {config['alpha_start']} → {config['alpha_end']} ({config['alpha_steps']} steps)",
        f"rho values   : {config['rho']}",
        f"learning rate: {config['lr']}",
        f"samples      : {config['samples']}",
        "---------------------",
    ]
    config_summary_msg = "\n".join(config_summary_lines)
    print(config_summary_msg)
    logging.info(config_summary_msg)

    logging.info(f"[INFO] Using device: {device}")
    logging.info(f"[INFO] Run index: {run_idx}")

    # --- Appel de l'Expérience ---
    results, alphas = run_experiment(base_dir=run_dir,
                                    D = config["d"],
                                    L = config["L"],
                                    rho = config["rho"],
                                    rho_star = config["rho_star"],
                                    beta = config["beta"],
                                    lam_list = config["lmbda"],
                                    Delta_list = config["Delta_list"],
                                    Delta_in = config["Delta_in"],
                                    samples = config["samples"],
                                    T = config["T"],
                                    learning_rate = config["lr"],
                                    norm_init = config["norm_init"],
                                    tol = config["tol"],
                                    N_test = config["n_test"],
                                    verbose = config["verbose"],
                                    alpha_list= alpha_list,
                                    run_index=run_idx)
    
    # --- Affichage et Log du Footer ---
    footer_lines = [
        "----------------------------------------",
        "✅ Experiment finished successfully",
        f"Results saved in: {run_dir}",
        f"Logs CSV: logs_{run_idx}.csv",
        "Config CSV: config.csv",
        f"Pickle: W_Q_runs_{run_idx}.pkl, W_K_runs_{run_idx}.pkl",
        "---------------------------------------",
        "",
        "",
    ]
    footer_msg = "\n".join(footer_lines)
    print(footer_msg)
    logging.info(footer_msg)

    # --- Sauvegarder la config et les résultats complets ---
    config["run_index"] = run_idx
    config_path = os.path.join(run_dir, "config_used.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    results_path = os.path.join(run_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    logging.info("[INFO] Configuration and results saved")

    # --- Créer un Summary CSV dans le répertoire d'exécution ---
    summary_csv_path = os.path.join(run_dir, "summary.csv")
    logs_csv_path = os.path.join(run_dir, f"logs_{run_idx}.csv")
    if os.path.isfile(logs_csv_path):
        try:
            df = pd.read_csv(logs_csv_path)

            # Ajouter des lignes si le fichier existe déjà
            if os.path.isfile(summary_csv_path):
                df.to_csv(summary_csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(summary_csv_path, mode='w', header=True, index=False)

            logging.info(f"[INFO] Summary CSV updated at {summary_csv_path}")
        except Exception as e:
            logging.warning(f"[ERR] Failed to update summary CSV: {e}")
    else:
        logging.warning("[ERR] logs.csv not found to update summary.csv")