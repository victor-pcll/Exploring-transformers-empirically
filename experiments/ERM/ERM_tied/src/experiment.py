import torch
import numpy as np
import pandas as pd
import os
import pickle
from src.model import Net
from src.training import train_student_on_data, S_MSE

def run_experiment(alpha_list, base_dir, run_index, D, L, rho, rho_star, beta, lam_list, Delta_list, Delta_in,
                   samples, T, learning_rate, norm_init, tol, N_test, device, logger):
    
    all_results = []

    # Filtrage de la liste d'alpha si un index spécifique est donné
    if run_index is not None and alpha_list is not None:
        try:
            alpha_list = [alpha_list[run_index]]
        except IndexError:
            logger.error(f"Run index {run_index} is out of bounds for alpha_list of size {len(alpha_list)}")
            return pd.DataFrame()

    for alpha_idx, alpha in enumerate(alpha_list):
        R = int(rho * D)
        R_star = int(rho_star * D)
        beta_star = beta
        os.makedirs(base_dir, exist_ok=True)

        for lam_cur in lam_list:
            for Delta_cur in Delta_list:
                
                # --- Teacher ---
                N = int(alpha * D**2)
                with torch.no_grad():
                    teacher = Net(D, R_star, L, norm=1.0, beta=beta_star, device=device)
                W_teacher = teacher.fc1.weight.detach().cpu().numpy()

                # --- Storage ---
                MSE_runs, label_err_runs, label_err_runs_noise = [], [], []
                train_data_runs, train_reg_runs, total_loss_runs, W_runs = [], [], [], []

                for i in range(samples):
                    x_train = torch.normal(0, 1, (N, L, D), device=device)
                    with torch.no_grad():
                        y_train = teacher(x_train, delta_in=Delta_in)

                    # --- Training Student ---
                    student_trained, data_loss_i, reg_loss_i = train_student_on_data(
                        D, L, R, beta, lam_cur, x_train, y_train,
                        rho=rho, T=T, learning_rate=learning_rate,
                        norm_init=norm_init, tol=tol, device=device
                    )
                    
                    # Sauvegarde des poids (CPU)
                    W_runs.append(student_trained.fc1.weight.detach().cpu())
                    
                    # Calcul MSE
                    mse_i = S_MSE(student_trained.fc1.weight.detach().cpu(), W_teacher, R, R_star, D)
                    MSE_runs.append(mse_i)

                    # --- Testing ---
                    x_test = torch.normal(0, 1, (N_test, L, D), device=device)
                    with torch.no_grad():
                        y_test_teacher = teacher(x_test, delta_in=0.0)
                        y_test_teacher_noise = teacher(x_test, delta_in=Delta_in)

                        student_eval = student_trained
                        student_eval.eval()
                        y_test_student = student_eval(x_test, delta_in=0.0)

                        label_err_i = 1/N_test * torch.sum((y_test_student - y_test_teacher)**2).item()
                        label_err_i_noise = 1/N_test * torch.sum((y_test_student - y_test_teacher_noise)**2).item()

                    label_err_runs.append(label_err_i)
                    label_err_runs_noise.append(label_err_i_noise)
                    train_data_runs.append(data_loss_i)
                    train_reg_runs.append(reg_loss_i)
                    total_loss_runs.append(data_loss_i + reg_loss_i)

                # --- Résultats agrégés ---
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
                    "W_runs": W_runs # Liste de tenseurs
                }

                all_results.append(results)
                logger.debug(f"[alpha={alpha:.4f}, lambda={lam_cur:.4f}] MSE={results['MSE_mean']:.6f}")

    # --- Sauvegarde CSV et Pickle ---
    # DataFrame sans la colonne lourde 'W_runs'
    df_results = pd.DataFrame([{k:v for k,v in res.items() if k != "W_runs"} for res in all_results])
    logs_csv_path = os.path.join(base_dir, f"logs_{run_index}.csv")
    df_results.to_csv(logs_csv_path, index=False)

    # Sauvegarde des poids dans un pickle séparé (conversion explicite en numpy)
    W_runs_all = [[w.numpy() for w in res["W_runs"]] for res in all_results]
    pickle_path = os.path.join(base_dir, f"W_runs_{run_index}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(W_runs_all, f)

    logger.info(f"💾 Results saved for run_index={run_index}")
    return df_results