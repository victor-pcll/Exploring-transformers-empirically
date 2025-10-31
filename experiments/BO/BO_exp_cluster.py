import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import logging
import sys
import pickle
import pandas as pd

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_tokens, norm=1.0, beta=1.0):
        super(Net, self).__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.R = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc1.weight.data.normal_(0, norm)

    def forward(self, x, delta_in):
        x = self.fc1(x) / np.sqrt(self.D)
        attention_matrix = torch.einsum('nap,nbp->nab', x, x) / np.sqrt(self.R)
        trace_part = torch.norm(self.fc1.weight)**2 / np.sqrt(self.R * self.D**2)
        x = attention_matrix - trace_part * torch.eye(self.L, device=attention_matrix.device)
        if delta_in > 0.0:
            M = torch.full((self.L, self.L), 1.0/np.sqrt(2), device=x.device, dtype=x.dtype)
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device, dtype=x.dtype)
            i, j = torch.triu_indices(row=self.L, col=self.L, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]
            x = x + np.sqrt(delta_in) * eps * M
        x = nn.Softmax(dim=-1)(self.beta * x)
        return x


def init_torch(seed=42, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if verbose:
        print(f"Random seeds initialized to {seed}")


def convert_numeric_config(config, verbose=True):
    for k, v in config.items():
        if isinstance(v, str):
            try:
                config[k] = int(v)  # Essayer de convertir en int
            except ValueError:
                try:
                    config[k] = float(v)  # Essayer de convertir en float
                except ValueError:
                    pass  # Laisser tel quel si ce n’est ni int ni float
        elif isinstance(v, dict):
            return convert_numeric_config(v)

    print(f"[Config] Converted config: {config}")
    return config


def train_student_on_data(D, L, R, beta, lam, x_train, y_train, rho=1.0, T=1000, learning_rate=0.02, norm_init=1.0, tol=1e-8):
    student = Net(D, R, L, norm=norm_init, beta=beta)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    prev_total_loss = None
    for t in range(T):
        optimizer.zero_grad()
        lam_stud = lam / np.sqrt(rho)
        y_pred = student(x_train, delta_in=0.0)
        data_loss = torch.sum((y_pred - y_train)**2)
        reg_loss = lam_stud * torch.sum(student.fc1.weight ** 2)
        total_loss = data_loss + reg_loss
        total_loss.backward()
        optimizer.step()
        cur = float(total_loss.item())
        if prev_total_loss is not None and abs(cur - prev_total_loss) < tol and t > 100:
            break
        prev_total_loss = cur
    with torch.no_grad():
        lam_stud2 = lam / np.sqrt(rho)
        y_pred_final = student(x_train, delta_in=0.0)
        data_loss_final = torch.sum((y_pred_final - y_train)**2).item()
        reg_loss_final = (lam_stud2 * torch.sum(student.fc1.weight ** 2)).item()
    W_student = student.fc1.weight.detach().cpu().numpy()
    return W_student, data_loss_final, reg_loss_final


def compute_S_from_W(W, R, D):
    return (W.T @ W) / np.sqrt(R * D)


def S_MSE(W_student, W_teacher, R, R_star, D):
    S_stud = compute_S_from_W(W_student, R, D)
    S_teach = compute_S_from_W(W_teacher, R_star, D)
    return float(((S_stud - S_teach) ** 2).sum() / D)


def run_experiment(D=100, L=2, beta=1.0,
                   lam=0.0, Delta_in=0.0,
                   samples=8, T=10000, learning_rate=0.01, norm_init=1.0,
                   tol=1e-7, N_test=2000, base_dir="./results", rho_list=[0.5, 1.0, 5.0],
                   alpha_list = None, run_index=None, verbose=False):

    all_results = []

    # If run_index is provided, filter alpha_list to only this index
    if run_index is not None and alpha_list is not None:
        alpha_list = [alpha_list[run_index]]

    for alpha_idx in range(len(alpha_list)):
        alpha = float(alpha_list[alpha_idx])

        beta_star = beta

        os.makedirs(base_dir, exist_ok=True)

        for rho in rho_list:
            R = int(rho * D)
            R_star = R

            N = int(alpha * D**2)
            with torch.no_grad():
                teacher = Net(D, R_star, L, norm=1.0, beta=beta_star)
            W_teacher = teacher.fc1.weight.detach().cpu().numpy()

            MSE_runs = []
            label_err_runs = []
            label_err_runs_noise = []
            train_data_runs = []
            train_reg_runs = []
            total_loss_runs = []
            W_runs = []

            for i in range(samples):
                x_train = torch.normal(0, 1, (N, L, D))
                with torch.no_grad():
                    y_train = teacher(x_train, delta_in=Delta_in)

                W_last, data_loss_i, reg_loss_i = train_student_on_data(
                    D, L, R, beta, lam, x_train, y_train,
                    rho=rho, T=T, learning_rate=learning_rate, norm_init=norm_init, tol=tol
                )
                W_runs.append(W_last)

                mse_i = S_MSE(W_last, W_teacher, R, R_star, D)
                MSE_runs.append(mse_i)

                x_test = torch.normal(0, 1, (N_test, L, D))
                with torch.no_grad():
                    y_test_teacher = teacher(x_test, delta_in=0.0)
                    y_test_teacher_noise = teacher(x_test, delta_in=Delta_in)

                student_eval = Net(D, R, L, norm=0.0, beta=beta)
                with torch.no_grad():
                    student_eval.fc1.weight.copy_(torch.tensor(W_last, dtype=student_eval.fc1.weight.dtype))
                    y_test_student = student_eval(x_test, delta_in=0.0)
                    label_err_i = torch.sum((y_test_student - y_test_teacher) ** 2).item()
                    label_err_i_noise = torch.sum((y_test_student - y_test_teacher_noise) ** 2).item()

                label_err_runs.append(label_err_i)
                label_err_runs_noise.append(label_err_i_noise)
                train_data_runs.append(data_loss_i)
                train_reg_runs.append(reg_loss_i)
                total_loss_runs.append(data_loss_i + reg_loss_i)

            results = {
                "alpha": alpha,
                "rho": rho,
                "lam": lam,
                "MSE_mean": float(np.mean(MSE_runs)),
                "MSE_std": float(np.std(MSE_runs, ddof=1)) if len(MSE_runs) > 1 else 0.0,
                "label_err_mean": float(np.mean(label_err_runs)/D**2),
                "label_err_std": float(np.std(label_err_runs, ddof=1)/D**4) if len(label_err_runs) > 1 else 0.0,
                "label_err_mean_noise": float(np.mean(label_err_runs_noise)/D**2),
                "label_err_std_noise": float(np.std(label_err_runs_noise, ddof=1)/D**4) if len(label_err_runs_noise) > 1 else 0.0,
                "train_data_mean": float(np.mean(train_data_runs)/D),
                "train_reg_mean": float(np.mean(train_reg_runs)/D),
                "train_total_mean": float(np.mean(total_loss_runs)/D),
                "W_runs": W_runs
            }
            all_results.append(results)

            # Enhanced logging and printing after each alpha/rho
            msg = f"[alpha={alpha:.3f}, rho={rho:.3f}] Finished → MSE={results['MSE_mean']:.4f}, label_err={results['label_err_mean']:.4f}"
            print(msg)
            logging.info(msg)

    # Convert all_results to DataFrame excluding 'W_runs'
    df_results = pd.DataFrame([
        {k: v for k, v in res.items() if k != "W_runs"}
        for res in all_results
    ])
    logs_csv_path = os.path.join(base_dir, f"logs_{run_index}.csv")
    if os.path.isfile(logs_csv_path):
        df_results.to_csv(logs_csv_path, mode='a', header=False, index=False)
    else:
        df_results.to_csv(logs_csv_path, mode='a', header=True, index=False)

    # --- Save config as CSV ---
    config_dict = {
        "alpha": alpha,
        "D": D,
        "L": L,
        "rho": rho,
        "beta": beta,
        "lam": lam,
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

    # Sauvegarde de la liste de tous les W_runs dans un fichier pickle
    W_runs_all = [res["W_runs"] for res in all_results]
    pickle_path = os.path.join(base_dir, f"W_runs_{run_index}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(W_runs_all, f)
        
    print(f"[INFO] Sauvegarde effectuée dans {os.path.abspath(base_dir)}")

    return df_results, alpha_list


def main(run_index):
    run_dir = sys.argv[2] if len(sys.argv) > 2 else None

    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    if run_dir is None:
        run_dir = f"/home/peucelle/tpiv-simulations/results/run_{job_id}"
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Embedded configuration ---
    config = {
        "verbose": False,
        "alpha_start": 0.1,
        "alpha_end": 0.4,
        "alpha_steps": 15,
        "d": 100,
        "L": 2,
        "beta": 1.0,
        "lmbda": 0.0,
        "Delta_in": 0.0,
        "samples": 8,
        "T": 10000,
        "lr": 0.01,
        "norm_init": 1.0,
        "tol": 1e-6,
        "n_test": 2000,
        "rho": [0.1, 0.5, 1.0],
    }

    # --- Initialiser graines ---
    init_torch(42, verbose=config.get("verbose", True))

    # --- Convertir les paramètres numériques ---
    config = convert_numeric_config(config, verbose=config.get("verbose", True))

    alpha_list = np.linspace(config["alpha_start"], config["alpha_end"], config["alpha_steps"])

    # --- Configurer logging ---
    log_file = os.path.join(run_dir, "experiment.txt")
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.INFO if config.get("verbose", True) else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # Print and log header block
    header_lines = [
        "========================================",
        "        EXPERIMENT START",
        "----------------------------------------",
        f"Run index: {run_index}",
        f"Job ID: {job_id}",
        f"Device: {device}",
        f"Results directory: {run_dir}",
        "========================================",
    ]
    header_msg = "\n".join(header_lines)
    print(header_msg)
    logging.info(header_msg)

    # Print and log configuration summary
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
    logging.info(f"[INFO] Run index: {run_index}")

    # --- Lancer l'expérience ---
    results, alphas = run_experiment(
        D=config["d"],
        L=config["L"],
        beta=config["beta"],
        lam=config["lmbda"],
        Delta_in=config["Delta_in"],
        samples=config["samples"],
        T=config["T"],
        learning_rate=config["lr"],
        norm_init=config["norm_init"],
        tol=config["tol"],
        N_test=config["n_test"],
        base_dir=run_dir,
        rho_list=config["rho"],
        alpha_list=alpha_list,
        run_index=run_index,
        verbose=config.get("verbose", True),
    )

    # Print and log footer block
    footer_lines = [
        "----------------------------------------",
        "✅ Experiment finished successfully",
        f"Results saved in: {run_dir}",
        f"Logs CSV: logs_{run_index}.csv",
        "Config CSV: config.csv",
        f"Pickle: W_runs_{run_index}.pkl",
        "---------------------------------------",
        "",
        "",
    ]
    footer_msg = "\n".join(footer_lines)
    print(footer_msg)
    logging.info(footer_msg)

    # --- Sauvegarder config pour traçabilité ---
    config["run_index"] = run_index
    config_path = os.path.join(run_dir, "config_used.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    results_path = os.path.join(run_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    logging.info("[INFO] Configuration and results saved")

    # --- Create summary CSV in run_dir directly ---
    summary_csv_path = os.path.join(run_dir, "summary.csv")
    # Just copy the logs.csv to summary.csv as all results are in one single CSV already
    logs_csv_path = os.path.join(run_dir, f"logs_{run_index}.csv")
    if os.path.isfile(logs_csv_path):
        try:
            df = pd.read_csv(logs_csv_path)

            # Append new rows if file exists
            if os.path.isfile(summary_csv_path):
                df.to_csv(summary_csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(summary_csv_path, mode='w', header=True, index=False)

            logging.info(f"[INFO] Summary CSV updated at {summary_csv_path}")
        except Exception as e:
            logging.warning(f"[ERR] Failed to update summary CSV: {e}")
    else:
        logging.warning("[ERR] logs.csv not found to update summary.csv")


if __name__ == "__main__":
    # Try to get run index from command line argument, else from SLURM_ARRAY_TASK_ID, else default to 0
    if len(sys.argv) > 1:
        run_idx = int(sys.argv[1])
    else:
        run_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    main(run_idx)