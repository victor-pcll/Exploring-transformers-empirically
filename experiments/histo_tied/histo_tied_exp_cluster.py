from datetime import datetime
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from collections import Counter
import numpy as np
import pandas as pd
import torch.nn as nn
import pickle
import sys
import logging
import os

# -------------------------------
# Initialisation
# -------------------------------
def init_torch(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------------
# Logger helper
# -------------------------------
def get_logger(run_dir, run_index, verbose=True):
    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_file = os.path.join(run_dir, f"experiment_{run_index}_{job_id}.log")
    logger = logging.getLogger(f"logger_{run_index}_{job_id}")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(ch)

    return logger, log_file

def hist(s):
  c = Counter(s)
  c = {w: c[w] for w in c}
  return [c[w] for w in s]

class HistogramDataset(Dataset):
    def __init__(self, seq_len, T, n_samples, seed=42):
        self.seq_len = seq_len
        self.T = T
        self.n_samples = n_samples
        rs = np.random.RandomState(seed)
        self.X = rs.randint(0, T, (n_samples, seq_len))
        # self.X = np.unique(self.X, axis=0)
        self.y = np.empty_like(self.X)
        self.n_samples = self.X.shape[0]
        for i in range(self.n_samples):
          self.y[i] = hist(self.X[i])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx],dtype=torch.long), torch.tensor(self.y[idx],dtype=torch.long)

# -------------------------------
# Neural network
# -------------------------------
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_tokens, seq_len, norm=1.0, beta=1.0, device="cpu"):
        super(Net, self).__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.seq_len = seq_len
        self.R = hidden_dim
        self.device = device
        self.embed = nn.Embedding(number_tokens, input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc1.weight.data.normal_(0, norm)
        self.to(device)

    def forward(self, x, delta_in=0.0):
        x = x.to(self.device)
        sqrt_D = torch.sqrt(torch.tensor(self.D, device=x.device, dtype=x.dtype))
        sqrt_R = torch.sqrt(torch.tensor(self.R, device=x.device, dtype=x.dtype))
        x = self.embed(x)     #  x.shape = (N, seq_len, input_dim)
        x = self.fc1(x) / sqrt_D
        attention_matrix = torch.einsum('nap,nbp->nab', x, x) / sqrt_R
        trace_part = torch.norm(self.fc1.weight)**2 / (sqrt_R * sqrt_D**2)
        x = attention_matrix - trace_part * torch.eye(self.seq_len, device=x.device)
        if delta_in > 0.0:
            M = torch.full((self.L, self.L), 1.0/torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype)), device=x.device, dtype=x.dtype)
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device, dtype=x.dtype)
            i, j = torch.triu_indices(row=self.L, col=self.L, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]
            x = x + torch.sqrt(torch.tensor(delta_in, device=x.device, dtype=x.dtype)) * eps * M
        x = nn.Softmax(dim=-1)(self.beta * x)
        return x

# -------------------------------
# Training student
# -------------------------------
def train_student_on_data(D, L, R, beta, lam, train_dataset, seq_len, rho=1.0, 
                             T = 1000, learning_rate=0.1, norm_init=1.0, tol=1e-6, device="cpu"):
    """
    Full-batch training for student network when L is small (e.g., L=2).
    """
    # --- Initialisation du réseau étudiant ---
    student = Net(D, R, L, seq_len, norm=norm_init, beta=beta, device=device)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

    # Récupérer tout le dataset en full batch
    X_full = torch.stack([x for x, _ in train_dataset]).long().to(device)
    y_full = torch.stack([y for _, y in train_dataset]).to(device)
    y_full_norm = y_full / (y_full.sum(dim=-1, keepdim=True) + 1e-8)
    
    lam_stud = lam / np.sqrt(rho)
    prev_total_loss = None

    for _ in range(T):
        optimizer.zero_grad()
        
        y_pred = student(X_full, delta_in=0.0)
        y_pred_diag = torch.diagonal(y_pred, dim1=1, dim2=2)

        data_loss = torch.sum((y_pred_diag - y_full_norm) ** 2)
        reg_loss = lam_stud * torch.sum(student.fc1.weight ** 2)
        total_loss = data_loss + reg_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Critère d’arrêt
        total_loss_val = total_loss.item()
        if prev_total_loss is not None and abs(total_loss_val - prev_total_loss) < tol:
            break
        prev_total_loss = total_loss_val

    # --- Évaluation finale ---
    with torch.no_grad():
        y_pred_final = student(X_full, delta_in=0.0)
        y_pred_diag_final = torch.diagonal(y_pred_final, dim1=1, dim2=2)
        data_loss_final = torch.sum((y_pred_diag_final - y_full_norm) ** 2).item()
        reg_loss_final = (lam_stud * torch.sum(student.fc1.weight ** 2)).item()

    W_student = student.fc1.weight.detach().cpu().numpy()
    return W_student, data_loss_final, reg_loss_final

# -------------------------------
# S_MSE helper
# -------------------------------
def compute_S_from_W(W, R, D):
    return (W.T @ W) / np.sqrt(R * D)

def S_MSE(W_student, W_teacher, R, R_star, D):
    S_stud = compute_S_from_W(W_student, R, D)
    S_teach = compute_S_from_W(W_teacher, R_star, D)
    return float(((S_stud - S_teach)**2).sum() / D)

# -------------------------------
# Run experiment
# -------------------------------
def run_experiment(alpha_list, base_dir, run_index, D, L, rho, rho_star, beta, lam_list, Delta_list, Delta_in,
                   samples, T, learning_rate, norm_init, tol, N_test, device, logger):

    all_results = []

    # --- ce sont des hyperparametres globaux ---
    seq_len = 10
    # --------------------------------

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
                    teacher = Net(D, R_star, L, seq_len, norm=1.0, beta=beta_star, device=device)
                W_teacher = teacher.fc1.weight.detach().cpu().numpy()

                # Storage
                MSE_runs, label_err_runs, train_data_runs, train_reg_runs, total_loss_runs, W_runs = [], [], [], [], [], []

                for _ in range(samples):

                    num_samples = N + N_test
                    dataset = HistogramDataset(seq_len, L, num_samples)
                    num_unique = len(dataset)

                    if num_unique < 2:
                        logger.warning(f"🚨 Dataset trop petit ({num_unique} unique samples), skip.")
                        continue

                    N_train = min(N, num_unique)
                    N_test_adj = num_unique - N_train
                    train_dataset, test_dataset = random_split(dataset, [N_train, N_test_adj])

                    W_last, data_loss_i, reg_loss_i = train_student_on_data(
                        D, L, R, beta, lam_cur, train_dataset, seq_len,
                        rho=rho, T=T, learning_rate=learning_rate,
                        norm_init=norm_init, tol=tol, device=device
                    )

                    W_runs.append(W_last)
                    mse_i = S_MSE(W_last, W_teacher, R, R_star, D)
                    MSE_runs.append(mse_i)

                    student_eval = Net(D, R, L, seq_len, norm=0.0, beta=beta, device=device)
                    with torch.no_grad():
                        student_eval.fc1.weight.copy_(torch.tensor(W_last, dtype=student_eval.fc1.weight.dtype, device=device))
                    student_eval.eval()

                    test_loader = DataLoader(test_dataset)
                    for x_test, y_test_teacher in test_loader:
                        x_test = x_test.long().to(device)
                        y_test_teacher = y_test_teacher.to(device)
                        with torch.no_grad():
                            y_test_student_full = student_eval(x_test, delta_in=0.0)
                            y_test_student_diag = torch.diagonal(y_test_student_full, dim1=1, dim2=2)
                            label_err_i = torch.mean((y_test_student_diag - y_test_teacher)**2).item()

                        label_err_runs.append(label_err_i)
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
                    "label_err_mean": float(np.mean(label_err_runs)/D**2),
                    "label_err_std": float(np.std(label_err_runs, ddof=1)/D**2) if len(label_err_runs) > 1 else 0.0,
                    "train_data_mean": float(np.mean(train_data_runs)/D**2),
                    "train_reg_mean": float(np.mean(train_reg_runs)/D**2),
                    "train_total_mean": float(np.mean(total_loss_runs)/D**2),
                    "W_runs": W_runs
                }

                all_results.append(results)
                logger.info(f"🔹 [alpha={alpha:.4f}, lambda={lam_cur:.4f}] → MSE={results['MSE_mean']:.6f}")

    # Save CSV & pickle
    df_results = pd.DataFrame([{k:v for k,v in res.items() if k != "W_runs"} for res in all_results])
    logs_csv_path = os.path.join(base_dir, f"logs_{run_index}.csv")
    df_results.to_csv(logs_csv_path, index=False)

    W_runs_all = [res["W_runs"] for res in all_results]
    pickle_path = os.path.join(base_dir, f"W_runs_{run_index}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(W_runs_all, f)

    logger.info(f"💾 Results saved for run_index={run_index}")
    return df_results

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    run_index = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = sys.argv[2] if len(sys.argv) > 2 else f"./results/run_{job_id}"
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger centralisé
    logger, log_file = get_logger(run_dir, run_index, verbose=True)

    # Seeds
    init_torch(42)
    logger.info(f"🖥 Device: {device} | Run index: {run_index} | Job ID: {job_id}")

    # Configuration
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
        "alpha_start": 0.05,
        "alpha_end": 10.0,
        "alpha_steps": 15
    }

    alpha_list = np.linspace(config["alpha_start"], config["alpha_end"], config["alpha_steps"])

    # --- Header log ---
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

    # -------------------------------
    # Save configuration as CSV
    # -------------------------------
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

    # -------------------------------
    # Update / create summary.csv
    # -------------------------------
    try:
        logs_csv_path = os.path.join(run_dir, f"logs_{run_index}.csv")
        summary_csv_path = os.path.join(run_dir, "summary.csv")

        if os.path.isfile(logs_csv_path):
            df_logs = pd.read_csv(logs_csv_path)

            # Append new data if summary already exists
            if os.path.isfile(summary_csv_path):
                df_logs.to_csv(summary_csv_path, mode="a", header=False, index=False)
            else:
                df_logs.to_csv(summary_csv_path, mode="w", header=True, index=False)

            logger.info(f"🧾 Summary CSV updated at {summary_csv_path}")
        else:
            logger.warning("[ERR] logs.csv not found — could not update summary.csv")
    except Exception as e:
        logger.warning(f"[ERR] Failed to update summary.csv: {e}")

    # -------------------------------
    # Footer log
    # -------------------------------
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