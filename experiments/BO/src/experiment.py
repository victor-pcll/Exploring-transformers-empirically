import os
import torch
import pandas as pd
import numpy as np
import pickle
from src.training import train_student_on_data
from src.data import generate_teacher_student_data
from src.models import S_MSE
from src.models import Net

def run_experiment(config, run_index=None):

    D = config['D']; L = config['L']; beta = config['beta']; lam = config['lam']
    Delta_in = config['Delta_in']; samples = config['samples']; T = config['T']
    learning_rate = config['lr']; norm_init = config['norm_init']; tol = config['tol']
    N_test = config['N_test']; base_dir = config['base_dir']
    rho_list = config['rho_list']; alpha_list = config['alpha_list']
    alpha = config['alpha']

    os.makedirs(base_dir, exist_ok=True)

    all_results = []

    N = int(alpha * D**2)

    for rho in rho_list:
        R = int(rho * D)
        R_star = R

        # ---- Build teacher the same way as your working version ----
        with torch.no_grad():
            teacher = Net(D, R_star, L, norm=1.0, beta=beta)
        W_teacher = teacher.fc1.weight.detach().cpu().numpy()

        # --- metric lists ---
        MSE_runs = []
        label_err_runs = []
        label_err_runs_noise = []
        train_data_runs = []
        train_reg_runs = []
        total_loss_runs = []
        W_runs = []

        # ----------------------------------------------------------
        #                     EXP RUNS
        # ----------------------------------------------------------
        for _ in range(samples):

            # --- Train data ---
            x_train = torch.normal(0, 1, (N, L, D))
            with torch.no_grad():
                y_train = teacher(x_train, delta_in=Delta_in)

            # --- Train student ---
            W_last, data_loss_i, reg_loss_i = train_student_on_data(
                D, L, R, beta, lam, x_train, y_train,
                rho=rho, T=T, learning_rate=learning_rate,
                norm_init=norm_init, tol=tol
            )

            W_runs.append(W_last)

            # --- Weight MSE ---
            mse_i = S_MSE(W_last, W_teacher, R, R_star, D)
            MSE_runs.append(mse_i)

            # --- Label error ---
            x_test = torch.normal(0, 1, (N_test, L, D))
            with torch.no_grad():
                y_test_teacher = teacher(x_test, delta_in=0.0)
                y_test_teacher_noise = teacher(x_test, delta_in=Delta_in)

            student_eval = Net(D, R, L, norm=0.0, beta=beta)
            with torch.no_grad():
                student_eval.fc1.weight.copy_(torch.tensor(W_last, dtype=student_eval.fc1.weight.dtype))
                y_test_student = student_eval(x_test, delta_in=0.0)

            # squared errors
            label_err_i = torch.sum((y_test_student - y_test_teacher) ** 2).item()
            label_err_i_noise = torch.sum((y_test_student - y_test_teacher_noise) ** 2).item()

            label_err_runs.append(label_err_i)
            label_err_runs_noise.append(label_err_i_noise)
            train_data_runs.append(data_loss_i)
            train_reg_runs.append(reg_loss_i)
            total_loss_runs.append(data_loss_i + reg_loss_i)

        # ----------------------------------------------------------
        #                STORE RESULTS
        # ----------------------------------------------------------
        results = {
            "alpha": alpha,
            "lam": lam,
            "rho": rho,

            "MSE_mean": float(np.mean(MSE_runs)),
            "MSE_std": float(np.std(MSE_runs, ddof=1)) if len(MSE_runs) > 1 else 0.0,

            "label_err_mean": float(np.mean(label_err_runs) / D**2),
            "label_err_std": float(np.std(label_err_runs, ddof=1) / D**2) if len(label_err_runs) > 1 else 0.0,

            "label_err_mean_noise": float(np.mean(label_err_runs_noise) / D**2),
            "label_err_std_noise": float(np.std(label_err_runs_noise, ddof=1) / D**2) if len(label_err_runs_noise) > 1 else 0.0,

            "train_data_mean": float(np.mean(train_data_runs) / D),
            "train_reg_mean": float(np.mean(train_reg_runs) / D),
            "train_total_mean": float(np.mean(total_loss_runs) / D),

            "W_runs": W_runs
        }

        all_results.append(results)

    # ----------------------------------------------------------
    # Save results CSV
    # ----------------------------------------------------------
    df_results = pd.DataFrame([
        {k: v for k, v in res.items() if k != 'W_runs'}
        for res in all_results
    ])

    df_results.to_csv(os.path.join(base_dir, f"logs_{run_index}.csv"), index=False)

    # ----------------------------------------------------------
    # Save W_runs pickle
    # ----------------------------------------------------------
    with open(os.path.join(base_dir, f"W_runs_{run_index}.pkl"), "wb") as f:
        pickle.dump([res["W_runs"] for res in all_results], f)

    return df_results