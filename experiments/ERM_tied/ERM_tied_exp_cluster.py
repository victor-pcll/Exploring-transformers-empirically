from datetime import datetime
import torch
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

# -------------------------------
# Neural network
# -------------------------------
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_tokens, norm=1.0, beta=1.0, device="cpu"):
        super(Net, self).__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.R = hidden_dim
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False).to(device)
        self.fc1.weight.data.normal_(0, norm)

    def forward(self, x, delta_in=0.0):
        x = x.to(self.device)
        sqrt_D = torch.sqrt(torch.tensor(self.D, device=x.device, dtype=x.dtype))
        sqrt_R = torch.sqrt(torch.tensor(self.R, device=x.device, dtype=x.dtype))
        x = self.fc1(x) / sqrt_D
        attention_matrix = torch.einsum('nap,nbp->nab', x, x) / sqrt_R
        trace_part = torch.norm(self.fc1.weight)**2 / (sqrt_R * sqrt_D**2)
        x = attention_matrix - trace_part * torch.eye(self.L, device=x.device)
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
def train_student_on_data(D, L, R, beta, lam, x_train, y_train, rho=1.0, T=1000, learning_rate=0.02, norm_init=1.0, tol=1e-8, device="cpu"):
    student = Net(D, R, L, norm=norm_init, beta=beta, device=device)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    prev_total_loss = None

    for t in range(T):
        optimizer.zero_grad()
        lam_stud = lam / np.sqrt(rho)
        y_pred = student(x_train, delta_in=0.0)
        data_loss = torch.sum((y_pred - y_train)**2)
        reg_loss = lam_stud * torch.sum(student.fc1.weight**2)
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
        reg_loss_final = (lam_stud2 * torch.sum(student.fc1.weight**2)).item()

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

    for alpha_idx, alpha in enumerate(alpha_list):

        R = int(rho * D)
        R_star = int(rho_star * D)
        beta_star = beta
        os.makedirs(base_dir, exist_ok=True)

        for lam_cur in lam_list:
            for Delta_cur in Delta_list:

                N = int(alpha * D**2)
                with torch.no_grad():
                    teacher = Net(D, R_star, L, norm=1.0, beta=beta_star, device=device)
                W_teacher = teacher.fc1.weight.detach().cpu().numpy()

                # Storage
                MSE_runs, label_err_runs, label_err_runs_noise = [], [], []
                train_data_runs, train_reg_runs, total_loss_runs, W_runs = [], [], [], []

                for i in range(samples):
                    x_train = torch.normal(0, 1, (N, L, D), device=device)
                    with torch.no_grad():
                        y_train = teacher(x_train, delta_in=Delta_in)

                    W_last, data_loss_i, reg_loss_i = train_student_on_data(
                        D, L, R, beta, lam_cur, x_train, y_train,
                        rho=rho, T=T, learning_rate=learning_rate,
                        norm_init=norm_init, tol=tol, device=device
                    )
                    W_runs.append(W_last)
                    mse_i = S_MSE(W_last, W_teacher, R, R_star, D)
                    MSE_runs.append(mse_i)

                    x_test = torch.normal(0, 1, (N_test, L, D), device=device)
                    with torch.no_grad():
                        y_test_teacher = teacher(x_test, delta_in=0.0)
                        y_test_teacher_noise = teacher(x_test, delta_in=Delta_in)

                        student_eval = Net(D, R, L, norm=0.0, beta=beta, device=device)
                        student_eval.fc1.weight.copy_(torch.tensor(W_last, dtype=student_eval.fc1.weight.dtype, device=device))
                        y_test_student = student_eval(x_test, delta_in=0.0)

                        label_err_i = torch.sum((y_test_student - y_test_teacher)**2).item()
                        label_err_i_noise = torch.sum((y_test_student - y_test_teacher_noise)**2).item()

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
                    "label_err_mean": float(np.mean(label_err_runs)/D**2),
                    "label_err_std": float(np.std(label_err_runs, ddof=1)/D**4) if len(label_err_runs) > 1 else 0.0,
                    "label_err_mean_noise": float(np.mean(label_err_runs_noise)/D**2),
                    "label_err_std_noise": float(np.std(label_err_runs_noise, ddof=1)/D**4) if len(label_err_runs_noise) > 1 else 0.0,
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
        "samples": 8,
        "T": 10000,
        "learning_rate": 0.1,
        "norm_init": 1.0,
        "tol": 1e-6,
        "N_test": 2000,
        "alpha_start": 0.005,
        "alpha_end": 1.0,
        "alpha_steps": 15
    }

    alpha_list = np.linspace(config["alpha_start"], config["alpha_end"], config["alpha_steps"])

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

logger.info("✅ Experiment finished successfully")