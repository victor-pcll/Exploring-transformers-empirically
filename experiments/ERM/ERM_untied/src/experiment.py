import torch
import numpy as np
import pandas as pd
import os
import pickle
import logging
import sys

from src.model import Net
from src.training import train_student_on_data, S_MSE, compute_S_from_W 

def run_experiment(alpha_idx=0, D=100, L=2, rho=1.0, rho_star=0.5, beta=1.0,
                   lam_list=[0.1, 0.01, 0.001, 0.0001, 0.00001], Delta_list=[0.0], Delta_in=0.5,
                   samples=8, T=10000, learning_rate=0.1, norm_init=1.0,
                   tol=1e-6, N_test=2000, base_dir="./results", verbose=False, alpha_list=np.linspace(0.005, 0.5, 10),
                   run_index=None):
    """
    Exécute une série d'expériences Teacher-Student pour différents paramètres.
    """
    all_results = []
    
    if run_index is not None and alpha_list is not None:
        alpha_list = [alpha_list[run_index]]

    for alpha_idx, alpha in enumerate(alpha_list):
        R = int(rho * D)        
        R_star = int(rho_star * D)
        beta_star = beta
        os.makedirs(base_dir, exist_ok=True)

        for lam_cur in lam_list:
            for Delta_cur in Delta_list:
                N = int(alpha * D**2)
                with torch.no_grad():
                    teacher = Net(D, R_star, L, norm=1.0, beta=beta_star)
                W_Q_teacher = teacher.W_Q.weight.detach().cpu().numpy()
                W_K_teacher = teacher.W_K.weight.detach().cpu().numpy()

                MSE_runs, label_err_runs, label_err_runs_noise = [], [], []
                train_data_runs, train_reg_runs, total_loss_runs = [], [], []
                W_Q_runs, W_K_runs = [], []

                for i in range(samples):
                    x_train = torch.normal(0, 1, (N, L, D))
                    with torch.no_grad():
                        y_train = teacher(x_train, delta_in=Delta_in)

                    W_Q_last, W_K_last, data_loss_i, reg_loss_i = train_student_on_data(
                        D, L, R, beta, lam_cur, x_train, y_train,
                        rho=rho, T=T, learning_rate=learning_rate, norm_init=norm_init, tol=tol
                    )
                    W_Q_runs.append(W_Q_last)
                    W_K_runs.append(W_K_last)

                    mse_i = S_MSE(W_Q_last, W_K_last, W_Q_teacher, W_K_teacher, R, R_star, D)
                    MSE_runs.append(mse_i)

                    # --- Tests ---
                    x_test = torch.normal(0, 1, (N_test, L, D))
                    with torch.no_grad():
                        y_test_teacher = teacher(x_test, delta_in=0.0)
                        y_test_teacher_noise = teacher(x_test, delta_in=Delta_in)

                    student_eval = Net(D, R, L, norm=0.0, beta=beta)
                    with torch.no_grad():
                        student_eval.W_Q.weight.copy_(torch.tensor(W_Q_last))
                        student_eval.W_K.weight.copy_(torch.tensor(W_K_last))
                        y_test_student = student_eval(x_test, delta_in=0.0)
                        
                        label_err_i = 1/N_test * torch.sum((y_test_student - y_test_teacher) ** 2).item()
                        label_err_i_noise = 1/N_test * torch.sum((y_test_student - y_test_teacher_noise) ** 2).item()

                    label_err_runs.append(label_err_i)
                    label_err_runs_noise.append(label_err_i_noise)
                    train_data_runs.append(data_loss_i)
                    train_reg_runs.append(reg_loss_i)
                    total_loss_runs.append(data_loss_i + reg_loss_i)

                results = {
                    "alpha": alpha,
                    "alpha_idx": alpha_idx,
                    "lam": lam_cur,
                    "rho": rho,
                    "MSE_mean": float(np.mean(MSE_runs)),
                    "MSE_std": float(np.std(MSE_runs, ddof=1)) if len(MSE_runs) > 1 else 0.0,
                    "label_err_mean": float(np.mean(label_err_runs)),
                    "label_err_std": float(np.std(label_err_runs, ddof=1)) if len(label_err_runs) > 1 else 0.0,
                    "label_err_mean_noise": float(np.mean(label_err_runs_noise)),
                    "label_err_std_noise": float(np.std(label_err_runs_noise, ddof=1)) if len(label_err_runs_noise) > 1 else 0.0,
                    "train_data_mean": float(np.mean(train_data_runs)/D**2),
                    "train_reg_mean": float(np.mean(train_reg_runs)/D**2),
                    "train_total_mean": float(np.mean(total_loss_runs)/D**2),
                    "W_Q_runs": W_Q_runs,
                    "W_K_runs": W_K_runs
                }
                all_results.append(results)


    # --- Sauvegarde des résultats et logs ---
    df_results = pd.DataFrame([{k: v for k, v in res.items() if k != "W_Q_runs" and k != "W_K_runs"} for res in all_results])
    logs_csv_path = os.path.join(base_dir, f"logs_{run_index}.csv" if run_index is not None else "logs.csv")
    df_results.to_csv(logs_csv_path, index=False)

    # --- Sauvegarde de la config (uniquement les paramètres simples) ---
    config_dict = {
        "alpha": alpha, # Stocker le dernier alpha traité
        "D": D,
        "L": L,
        "rho": rho,
        "rho_star": rho_star,
        "beta": beta,
        "lam": lam_cur, # Stocker le dernier lambda traité
        "Delta_in": Delta_in,
        "samples": samples,
        "T": T,
        "learning_rate": learning_rate,
        "norm_init": norm_init,
        "tol": tol,
        "N_test": N_test,
        "base_dir": base_dir
    }
    config_df = pd.DataFrame([config_dict])
    config_csv_path = os.path.join(base_dir, "config.csv")
    if os.path.isfile(config_csv_path):
        config_df.to_csv(config_csv_path, mode='a', header=False, index=False)
    else:
        config_df.to_csv(config_csv_path, mode='a', header=True, index=False)

    # Sauvegarde des listes de W_Q_runs et W_K_runs dans des fichiers pickle
    W_Q_runs_all = [res["W_Q_runs"] for res in all_results]
    pickle_path_Q = os.path.join(base_dir, f"W_Q_runs_{run_index}.pkl")
    with open(pickle_path_Q, "wb") as f:
        pickle.dump(W_Q_runs_all, f)

    W_K_runs_all = [res["W_K_runs"] for res in all_results]
    pickle_path_K = os.path.join(base_dir, f"W_K_runs_{run_index}.pkl")
    with open(pickle_path_K, "wb") as f:
        pickle.dump(W_K_runs_all, f)
        
    print(f"[INFO] Sauvegarde effectuée dans {os.path.abspath(base_dir)}")

    return df_results, alpha_list