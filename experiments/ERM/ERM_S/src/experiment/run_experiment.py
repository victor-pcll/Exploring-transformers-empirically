import os
import pickle
import numpy as np
import pandas as pd
import torch
from src.models.Net_S import Net
from src.training.train_student import train_student_on_data
from src.utils.S_MSE import S_MSE

def run_experiment(alpha_idx=0, D=100, L=2, rho=1.00, rho_star=0.5, beta=1.0,
                   lam_list=[0.1, 0.01, 0.001, 0.0001, 0.00001], Delta_list=[0.0], Delta_in=0.5,
                   samples=8, T=10000, learning_rate=0.1, norm_init=1.0,
                   tol=1e-6, N_test=2000, base_dir="./results", verbose=False,  alpha_list = np.linspace(0.005, 0.5, 10),
                   run_index=None):

    all_results = []

    R_star = int(rho_star * D)

    if run_index is not None and alpha_list is not None:
        alpha_list = [alpha_list[run_index]]

    for alpha_idx, alpha in enumerate(alpha_list):
        beta_star = beta
        os.makedirs(base_dir, exist_ok=True)

        for lam_cur in lam_list:
            for Delta_cur in Delta_list:
                # --- Étape d'entraînement du teacher ---
                N = int(alpha * D**2)
                with torch.no_grad():
                    teacher = Net(D, L, norm=1.0, beta=beta_star)
                    teacher.init_teacher(R_star)
                S_teacher = teacher.S.detach().cpu().numpy()

                MSE_runs, label_err_runs, label_err_runs_noise = [], [], []
                train_data_runs, train_reg_runs, total_loss_runs = [], [], []
                S_runs = []

                for _ in range(samples):
                    x_train = torch.normal(0, 1, (N, L, D))
                    with torch.no_grad():
                        y_train = teacher(x_train, delta_in=Delta_in)

                    S_last, data_loss_i, reg_loss_i = train_student_on_data(
                        D, L, beta, lam_cur, x_train, y_train,
                        rho=rho, T=T, learning_rate=learning_rate, norm_init=norm_init, tol=tol
                    )
                    S_runs.append(S_last)

                    mse_i = S_MSE(S_last, S_teacher, D)
                    MSE_runs.append(mse_i)

                    # --- Tests ---
                    x_test = torch.normal(0, 1, (N_test, L, D))
                    with torch.no_grad():
                        y_test_teacher = teacher(x_test, delta_in=0.0)
                        y_test_teacher_noise = teacher(x_test, delta_in=Delta_in)

                    student_eval = Net(D, L, norm=0.0, beta=beta)
                    with torch.no_grad():
                        student_eval.S.data.copy_(torch.tensor(S_last))
                        y_test_student = student_eval(x_test, delta_in=0.0)
                        label_err_i = torch.sum((y_test_student - y_test_teacher) ** 2).item()
                        label_err_i_noise = torch.sum((y_test_student - y_test_teacher_noise) ** 2).item()

                    label_err_runs.append(label_err_i)
                    label_err_runs_noise.append(label_err_i_noise)
                    train_data_runs.append(data_loss_i)
                    train_reg_runs.append(reg_loss_i)
                    total_loss_runs.append(data_loss_i + reg_loss_i)

                # --- Stockage des résultats ---
                results = {
                    "alpha": alpha,
                    "alpha_idx": alpha_idx,
                    "lam": lam_cur,
                    "rho": rho,
                    "MSE_mean": float(np.mean(MSE_runs)),
                    "MSE_std": float(np.std(MSE_runs, ddof=1)) if len(MSE_runs) > 1 else 0.0,
                    "label_err_mean": float(np.mean(label_err_runs)/D**2),
                    "label_err_std": float(np.std(label_err_runs, ddof=1)/D**2) if len(label_err_runs) > 1 else 0.0,
                    "label_err_mean_noise": float(np.mean(label_err_runs_noise)/D**2),
                    "label_err_std_noise": float(np.std(label_err_runs_noise, ddof=1)/D**2) if len(label_err_runs_noise) > 1 else 0.0,
                    "train_data_mean": float(np.mean(train_data_runs)/D**2),
                    "train_reg_mean": float(np.mean(train_reg_runs)/D**2),
                    "train_total_mean": float(np.mean(total_loss_runs)/D**2),
                    "S_runs": S_runs
                }
                all_results.append(results)

    # Save logs and W_runs per alpha_idx to avoid overwriting
    df_results = pd.DataFrame([{k: v for k, v in res.items() if k != "S_runs"} for res in all_results])
    logs_csv_path = os.path.join(base_dir, f"logs_{run_index}.csv" if run_index is not None else "logs.csv")
    df_results.to_csv(logs_csv_path, index=False)

    # --- Save config as CSV ---
    config_dict = {
        "alpha": alpha,
        "D": D,
        "L": L,
        "rho": rho,
        "rho_star": rho_star,
        "beta": beta,
        "lam": lam_cur,
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

    # Sauvegarde de la liste de tous les W_Q_runs dans un fichier pickle
    S_runs_all = [res["S_runs"] for res in all_results]
    pickle_path = os.path.join(base_dir, f"S_runs_{run_index}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(S_runs_all, f)    

    return df_results, alpha_list