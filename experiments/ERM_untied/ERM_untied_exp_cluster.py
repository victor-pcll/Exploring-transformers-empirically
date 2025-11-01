from datetime import datetime
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pickle
import sys
import logging
import os

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

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_tokens, norm=1.0, beta=1.0):
        super(Net, self).__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.R = hidden_dim
        self.W_Q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(input_dim, hidden_dim, bias=False)
        # Initialisation normale
        self.W_Q.weight.data.normal_(0, norm)
        self.W_K.weight.data.normal_(0, norm)

    def forward(self, x, delta_in):
        Q = self.W_Q(x) / np.sqrt(self.D)
        K = self.W_K(x) / np.sqrt(self.D)
        attention_matrix = torch.einsum('nap,nbp->nab', Q, K) / np.sqrt(self.R)
        trace_part = (
            torch.norm(self.W_Q.weight)**2 +
            torch.norm(self.W_K.weight)**2
        ) / np.sqrt(2 * self.R * self.D**2)
        x = attention_matrix # - trace_part * torch.eye(self.L, device=attention_matrix.device)
        if delta_in > 0.0:
            M = torch.full((self.L, self.L), 1.0 / np.sqrt(2), device=x.device, dtype=x.dtype)
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device, dtype=x.dtype)
            i, j = torch.triu_indices(row=self.L, col=self.L, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]
            x = x + np.sqrt(delta_in) * eps * M
        x = nn.Softmax(dim=-1)(self.beta * x)
        return x


def train_student_on_data(D, L, R, beta, lam, x_train, y_train, rho=1.0, T=1000, learning_rate=0.02, norm_init=1.0, tol=1e-8):
    student = Net(D, R, L, norm=norm_init, beta=beta)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    prev_total_loss = None
    for t in range(T):
        optimizer.zero_grad()
        lam_stud = lam / np.sqrt(rho)
        y_pred = student(x_train, delta_in=0.0)
        data_loss = torch.sum((y_pred - y_train)**2)
        reg_loss = lam_stud * (torch.sum(student.W_Q.weight ** 2) + torch.sum(student.W_K.weight ** 2))
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
        reg_loss_final = (lam_stud2 * (torch.sum(student.W_Q.weight ** 2) + torch.sum(student.W_K.weight ** 2))).item()
    W_Q_student = student.W_Q.weight.detach().cpu().numpy()
    W_K_student = student.W_K.weight.detach().cpu().numpy()
    return W_Q_student, W_K_student, data_loss_final, reg_loss_final


def compute_S_from_W(W_Q, W_K, R, D):
    return (W_Q.T @ W_K) / np.sqrt(R * D)


def S_MSE(W_Q_student, W_K_student, W_Q_teacher, W_K_teacher, R, R_star, D):
    S_stud = compute_S_from_W(W_Q_student, W_K_student, R, D)
    S_teach = compute_S_from_W(W_Q_teacher, W_K_teacher, R_star, D)
    return float(((S_stud - S_teach) ** 2).sum() / D)

def run_experiment(alpha_idx=0, D=100, L=2, rho=1.00, rho_star=0.5, beta=1.0,
                   lam_list=[0.1, 0.01, 0.001, 0.0001, 0.00001], Delta_list=[0.0], Delta_in=0.5,
                   samples=8, T=10000, learning_rate=0.1, norm_init=1.0,
                   tol=1e-6, N_test=2000, base_dir="./results", verbose=False, alpha_list = np.linspace(0.005, 0.5, 10),
                   run_index=None):

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
                # --- Étape d'entraînement du teacher ---
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
                    "label_err_std": float(np.std(label_err_runs, ddof=1)/D**4) if len(label_err_runs) > 1 else 0.0,
                    "label_err_mean_noise": float(np.mean(label_err_runs_noise)/D**2),
                    "label_err_std_noise": float(np.std(label_err_runs_noise, ddof=1)/D**4) if len(label_err_runs_noise) > 1 else 0.0,
                    "train_data_mean": float(np.mean(train_data_runs)/D**2),
                    "train_reg_mean": float(np.mean(train_reg_runs)/D**2),
                    "train_total_mean": float(np.mean(total_loss_runs)/D**2),
                    "W_Q_runs": W_Q_runs,
                    "W_K_runs": W_K_runs
                }
                all_results.append(results)



    # Save logs and W_runs per alpha_idx to avoid overwriting
    df_results = pd.DataFrame([{k: v for k, v in res.items() if k != "W_runs"} for res in all_results])
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
    W_Q_runs_all = [res["W_Q_runs"] for res in all_results]
    pickle_path = os.path.join(base_dir, f"W_Q_runs_{run_index}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(W_Q_runs_all, f)

    # Sauvegarde de la liste de tous les W_K_runs dans un fichier pickle
    W_K_runs_all = [res["W_K_runs"] for res in all_results]
    pickle_path = os.path.join(base_dir, f"W_K_runs_{run_index}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(W_K_runs_all, f)
        
    print(f"[INFO] Sauvegarde effectuée dans {os.path.abspath(base_dir)}")

    return df_results, alpha_list

def get_run_dir(base_path="/home/peucelle/tpiv-simulations/results"):
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{base_path}/run_{now_str}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


if __name__ == "__main__":

    if len(sys.argv) > 1:
        run_idx = int(sys.argv[1])
    else:
        run_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    run_dir = sys.argv[2] if len(sys.argv) > 2 else None

    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    if run_dir is None:
        run_dir = get_run_dir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger la configuration
    config = {
        "verbose": False,
        "alpha_start": 0.005,
        "alpha_end": 1.0,
        "alpha_steps": 10,
        "d": 100,
        "L": 2,
        "beta": 1.0,
        "lmbda": [0.1, 0.01, 0.001, 0.0001, 0.00001],
        "Delta_in": 0.5,
        "Delta_list": [0.0],
        "samples": 8,
        "T": 10000,
        "lr": 0.1,
        "norm_init": 1.0,
        "tol": 1e-6,
        "n_test": 2000,
        "rho": 0.75,
        "rho_star": 0.5  
    }

    # Initialiser les graines
    init_torch(42, verbose=config.get("verbose", True))

    # Convertir les paramètres numériques
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

    # Print and log header block
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
    logging.info(f"[INFO] Run index: {run_idx}")

    # Appel simple avec l’alpha numéro 5
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
    
    # Print and log footer block
    footer_lines = [
        "----------------------------------------",
        "✅ Experiment finished successfully",
        f"Results saved in: {run_dir}",
        f"Logs CSV: logs_{run_idx}.csv",
        "Config CSV: config.csv",
        f"Pickle: W_runs_{run_idx}.pkl",
        "---------------------------------------",
        "",
        "",
    ]
    footer_msg = "\n".join(footer_lines)
    print(footer_msg)
    logging.info(footer_msg)

    # --- Sauvegarder config pour traçabilité ---
    config["run_index"] = run_idx
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
    logs_csv_path = os.path.join(run_dir, f"logs_{run_idx}.csv")
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