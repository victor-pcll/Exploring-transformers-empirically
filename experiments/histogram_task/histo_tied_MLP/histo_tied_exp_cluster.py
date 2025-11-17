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





# -------------------------------
# Prepare dataset
# -------------------------------
def prepare_dataset(config):
    """
    Prépare le dataset pour le histogram task, et effectue le split train/test.
    
    Args:
        config: objet de configuration contenant au moins :
            - seq_len: longueur des séquences
            - L: taille de l'alphabet
            - N_train: nombre d'échantillons d'entraînement
            - N_test: nombre d'échantillons de test
        seed: int, pour reproductibilité
    
    Returns:
        train_dataset: torch Dataset pour l'entraînement
        test_dataset: torch Dataset pour le test
        full_dataset: torch Dataset complet
    """
    # --- Création du dataset complet ---
    full_dataset = HistogramDataset(config)
    
    # --- Calcul des tailles de split ---
    train_dataset, test_dataset = random_split(full_dataset, [config["N_train"], config["N_test"]],
                                               generator=torch.Generator().manual_seed(config["seed"]))
    
    return train_dataset, test_dataset









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
# Histogram dataset
# -------------------------------
def hist(s):
  c = Counter(s) # count occurrences in a dictionary
  c = {w: c[w] for w in c} # convert to normal dict
  return [c[w] for w in s]

class HistogramDataset(Dataset):
    def __init__(self, config):
        self.seq_len = config["seq_len"]
        self.L = config["L"]
        self.n_samples = config["N_total"]
        rs = np.random.RandomState(config["seed"])
        self.X = rs.randint(0, config["L"], (config["N_total"], config["seq_len"]))  # shape: (n_total, seq_len)
        # self.X = np.unique(self.X, axis=0)
        self.y = np.empty_like(self.X) # shape: (n_total, seq_len)  
        for i in range(self.n_samples):
          self.y[i] = hist(self.X[i]) 

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx],dtype=torch.long), torch.tensor(self.y[idx],dtype=torch.long)
    



# -------------------------------
# h to attn
# -------------------------------
def histogram_to_similarity_matrix_batch(y_batch):
    """
    y_batch: Tensor (N, seq_len)
    retourne M: Tensor (N, seq_len, seq_len)
    """
    y_batch = y_batch.float()
    N, seq_len = y_batch.shape

    # On étend pour broadcast
    Y_i = y_batch.unsqueeze(2)           # (N, seq_len, 1)
    Y_j = y_batch.unsqueeze(1)           # (N, 1, seq_len)

    # Création du masque
    eye = torch.eye(seq_len, dtype=torch.bool, device=y_batch.device).unsqueeze(0)  # (1, seq_len, seq_len)
    mask = (Y_i == Y_j) & (~eye)        # (N, seq_len, seq_len)

    # Initialisation
    M = torch.zeros((N, seq_len, seq_len), dtype=torch.float, device=y_batch.device)

    # On remplit **en broadcast** avec division
    M = torch.where(mask, 1.0 / Y_i, M)  # 1/Y_i broadcasté correctement sur la 3ème dimension

    return M




# -------------------------------
# h_pred
# -------------------------------
def compute_h_pred(attention_matrix):
    """
    Calcule h_pred par position pour correspondre à y_true.
    Args:
        attention_matrix: np.ndarray ou torch.Tensor de forme (N, seq_len, seq_len)
    Returns:
        h_pred: même forme que y_true -> (N, seq_len)
    """
    # Torch branch
    if isinstance(attention_matrix, torch.Tensor):
        # Calculer moyenne et std par colonne (pour chaque position de sortie)
        mean_col = attention_matrix.mean(dim=1, keepdim=True)  # (N, 1, seq_len)
        std_col = attention_matrix.std(dim=1, keepdim=True)    # (N, 1, seq_len)
        max_col = attention_matrix.amax(dim=1, keepdim=True)   # (N, 1, seq_len)
        min_col = attention_matrix.amin(dim=1, keepdim=True)   # (N, 1, seq_len)
        threshold = (max_col + min_col) / 2 + std_col          # (N, 1, seq_len)
        # Pour chaque colonne j, compter combien de valeurs (sur l'axe 1) dépassent le seuil
        h_pred = 1 + (attention_matrix >= threshold).sum(dim=1)  # (N, seq_len)
        return h_pred

    # Numpy branch
    else:
        mean_col = attention_matrix.mean(axis=1, keepdims=True)  # (N, 1, seq_len)
        std_col = attention_matrix.std(axis=1, keepdims=True)    # (N, 1, seq_len)
        max_col = attention_matrix.max(axis=1, keepdims=True)    # (N, 1, seq_len)
        min_col = attention_matrix.min(axis=1, keepdims=True)    # (N, 1, seq_len)
        threshold = (max_col + min_col) / 2 + std_col            # (N, 1, seq_len)
        h_pred = 1 + (attention_matrix >= threshold).sum(axis=1)  # (N, seq_len)
        return h_pred



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
        x = self.embed(x)     # x.shape = (N, seq_len, input_dim)
        x = self.fc1(x) / np.sqrt(self.D)  # x.shape = (N, seq_len, hidden_dim)
        attention_matrix = torch.einsum('nap,nbp->nab', x, x) / np.sqrt(self.R)  # attention_matrix.shape = (N, seq_len, seq_len)
        trace_part = torch.norm(self.fc1.weight)**2 / np.sqrt(self.R * self.D**2)
        x = attention_matrix - trace_part * torch.eye(self.seq_len, device=x.device) # x.shape = (N, seq_len, seq_len)
        if delta_in > 0.0:
            M = torch.full((self.seq_len, self.seq_len), 1.0/torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype)), device=x.device, dtype=x.dtype)
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device, dtype=x.dtype)
            i, j = torch.triu_indices(row=self.seq_len, col=self.seq_len, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]
            x = x + torch.sqrt(torch.tensor(delta_in, device=x.device, dtype=x.dtype)) * eps * M
        x = nn.Softmax(dim=-1)(self.beta * x)  # x.shape = (N, seq_len, seq_len)
        # h_pred = compute_h_pred(x)                # (N, seq_len)
        h_pred = x.sum(dim=-1)          # h_pred.shape = (N, seq_len)
        return x, h_pred







# -------------------------------
# Training student
# -------------------------------
def train_student_on_data(config, lmbda, train_dataset):
    """
    Full-batch training for student network.
    Returns:
        W_student: learned student weights
        data_loss_final: data loss
        reg_loss_final: regularization loss
        attn_matrix_final: attention matrix from last forward pass (on a sample)
        seq_sample: input sequence corresponding to attn_matrix_final
    """

    # --- Initialize the student network ---
    student = Net(config["D"], config["R"], config["L"], config["seq_len"], norm=config["norm_init"], beta=config["beta"], device=config["device"]) # init student network
    optimizer = torch.optim.Adam(student.parameters(), lr=config["learning_rate"]) # optimizer
    
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    X_full, y_full = next(iter(train_loader))
    X_full = X_full.long().to(config["device"])
    A_teacher = histogram_to_similarity_matrix_batch(X_full)
    y_true = y_full.to(config["device"])
    y_true_f = y_true.float()

    loss_prev = None
    attn_matrix_final = None
    seq_sample = None
    h_pred_sample = None
    y_true_sample = None
    grad_align_history = []

    # --- Training loop ---
    for t in range(config["T"]):
        optimizer.zero_grad()
        lam_stud = lmbda / np.sqrt(config["rho"])

        A_student, h_pred = student(X_full, delta_in=0.0)

        # --- Loss ---
        h_pred_f = h_pred.float()
        data_loss = torch.mean((A_teacher - A_student) ** 2)
        reg_loss = lam_stud * torch.sum(student.fc1.weight ** 2) * 0.0
        total_loss = data_loss  + reg_loss

        total_loss.backward()

        grad_vector = []
        for name, p in student.named_parameters():
            if p.grad is not None:
                grad_vector.append(p.grad.detach().flatten())

        # On stocke le gradient courant
        grad_t = torch.cat(grad_vector)

        # Initialisation
        if t == 0:
            grad_prev = grad_t.clone()
        else:
            # Cosine similarity entre deux gradients successifs
            cos_sim = torch.dot(grad_prev, grad_t) / (grad_prev.norm() * grad_t.norm() + 1e-12)

            if t % 500 == 0:
                print(f"[t={t}] Gradient alignment = {cos_sim.item():.4f}")

            grad_align_history.append(float(cos_sim.item()))
            grad_prev = grad_t.clone()
            
        optimizer.step()

        loss_cur = float(total_loss.item())
        if loss_prev is not None and abs(loss_cur - loss_prev) < config["tol"] and t > 100:
            break
        loss_prev = loss_cur

    # --- Evaluation after training ---
    with torch.no_grad():
        lam_stud2 = lmbda / np.sqrt(config["rho"])
        _, h_final = student(X_full, delta_in=0.0)
        h_final_f = h_final.float()
        data_loss_final = torch.sum((h_final_f - y_true_f) ** 2).item()
        reg_loss_final = (lam_stud2 * torch.sum(student.fc1.weight ** 2)).item()

        # Pick a sample to visualize
        seq_sample = X_full[0].detach().cpu().numpy()
        attn_out, h_pred_out = student(X_full[0].unsqueeze(0), delta_in=0.0)
        attn_matrix_final = attn_out[0].detach().cpu().numpy()
        h_pred_sample = h_pred_out[0].detach().cpu().numpy()
        y_true_sample = y_true[0].detach().cpu().numpy()

    return student, data_loss_final, reg_loss_final, attn_matrix_final, seq_sample, h_pred_sample, y_true_sample, grad_align_history








# -------------------------------
# Run experiment
# -------------------------------
def run_experiment(config):

    all_results = []                    # To store all results
    attn_runs = []                      # To store attention matrices
    seq_runs = []                       # To store sample sequences associated with attention matrices    

    # Pour stocker tous les h_pred_sample et y_true_sample pour tous les runs
    h_pred_samples_all = []
    y_true_samples_all = []
    grad_align_all = []

    os.makedirs(config["run_dir"], exist_ok=True)

    for lam_cur in config["lam_list"]:
        for _ in config["Delta_list"]:

            # Storage
            MSE_runs, label_err_runs, train_data_runs, train_reg_runs, total_loss_runs, W_runs = [], [], [], [], [], []
            attn_runs_samples = []
            seq_runs_samples = []

            # Pour stocker les runs de test pour ce set d'hyperparams
            test_seq_list_samples = []
            attn_student_list_samples = []

            # Pour stocker h_pred/y_true pour chaque sample de ce set d'hyperparams
            h_pred_samples = []
            y_true_samples = []

            train_dataset, test_dataset = prepare_dataset(config)

            for _ in range(config["samples"]):

                # Modification: récupère h_pred_sample et y_true_sample
                student_trained, data_loss_i, reg_loss_i, attn_matrix, seq_sample, h_pred_sample, y_true_sample, grad_align_hist = train_student_on_data(
                    config, lam_cur, train_dataset
                )
                W_runs.append(student_trained.fc1.weight.detach().cpu().numpy())
                attn_runs_samples.append(attn_matrix)
                seq_runs_samples.append(seq_sample)

                # Ajout: stocke h_pred_sample et y_true_sample
                h_pred_samples.append(h_pred_sample)
                y_true_samples.append(y_true_sample)
                grad_align_all.append(grad_align_hist)

                student_eval = student_trained
                student_eval.eval()

                # On récupère tout le batch test d'un coup (pour avoir tous les comptages)
                test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

                for X_test_full, y_test_full in test_loader:
                    X_test_full = X_test_full.long().to(config["device"])
                    y_test_full = y_test_full.cpu().numpy()  

                    with torch.no_grad():
                        attn_test, h_test = student_eval(X_test_full, delta_in=0.0)
                        y_counts_student = h_test.detach().cpu().numpy()

                    # Erreur quadratique moyenne
                    label_err_i = np.mean((y_counts_student - y_test_full) ** 2)

                # --- Enregistrement des résultats ---
                label_err_runs.append(label_err_i)
                train_data_runs.append(data_loss_i)
                train_reg_runs.append(reg_loss_i)
                total_loss_runs.append(data_loss_i + reg_loss_i)

                # Conversion en numpy pour stockage / pickle, suppression des dims inutiles
                seq_np = np.squeeze(X_test_full.cpu().numpy())
                attn_student_np = np.squeeze(attn_test.cpu().numpy())

                # Sauvegarde locale dans des listes pour analyse ultérieure
                test_seq_list_samples.append(seq_np)
                attn_student_list_samples.append(attn_student_np)

            # Ajout: stocke les samples pour tous les runs
            h_pred_samples_all.extend(h_pred_samples)
            y_true_samples_all.extend(y_true_samples)

            attn_runs.append(np.mean(attn_runs_samples, axis=0))  # shape: (seq_len, seq_len)
            seq_runs.append(seq_runs_samples[0])  # shape: (seq_len,)

            results = {
                "alpha": config["alpha"],
                "lam": lam_cur,
                "rho": config["rho"],
                "MSE_mean": float(np.mean(MSE_runs)),
                "MSE_std": float(np.std(MSE_runs, ddof=1)) if len(MSE_runs) > 1 else 0.0,
                "label_err_mean": float(np.mean(label_err_runs)/config["D"]**2),
                "label_err_std": float(np.std(label_err_runs, ddof=1)/config["D"]**2) if len(label_err_runs) > 1 else 0.0,
                "train_data_mean": float(np.mean(train_data_runs)/config["D"]**2),
                "train_reg_mean": float(np.mean(train_reg_runs)/config["D"]**2),
                "train_total_mean": float(np.mean(total_loss_runs)/config["D"]**2),
                "W_runs": W_runs
            }

            all_results.append(results)
            config["logger"].info(f"🔹 [alpha={config['alpha']:.4f}, lambda={lam_cur:.4f}] → MSE={results['MSE_mean']:.6f}")

    # Save CSV & pickle
    df_results = pd.DataFrame([{k:v for k,v in res.items() if k != "W_runs"} for res in all_results])
    logs_csv_path = os.path.join(config['run_dir'], f"logs_{config['run_index']}.csv")
    df_results.to_csv(logs_csv_path, index=False)

    W_runs_all = [res["W_runs"] for res in all_results]
    pickle_path = os.path.join(config['run_dir'], f"W_runs_{config['run_index']}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(W_runs_all, f)

    # Save attention matrices and sample sequences
    attn_save_path = os.path.join(config['run_dir'], f"attn_run{config['run_index']}.pkl")
    with open(attn_save_path, "wb") as f:
        pickle.dump({
            "attn_matrices": attn_runs,
            "seq_samples": seq_runs
        }, f)

    # Ajout: sauvegarde des h_pred_sample et y_true_sample pour chaque run dans un pickle
    h_pred_y_true_save_path = os.path.join(config['run_dir'], f"h_pred_y_true_run{config['run_index']}.pkl")
    with open(h_pred_y_true_save_path, "wb") as f:
        pickle.dump({
            "h_pred_samples": h_pred_samples_all,
            "y_true_samples": y_true_samples_all
        }, f)

    # Save grad_align_all to pickle
    grad_align_save_path = os.path.join(config["run_dir"], f"grad_align_run{config['run_index']}.pkl")
    with open(grad_align_save_path, "wb") as f:
        pickle.dump(grad_align_all, f)

    config["logger"].info(f"💾 Results saved for run_index={config['run_index']}")
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

    # Alpha list
    alpha_start = 0.05
    alpha_end = 1.0
    alpha_steps = 3
    alpha_list = np.linspace(alpha_start, alpha_end, alpha_steps).tolist()
    alpha = alpha_list[run_index] if run_index < len(alpha_list) else alpha_list[-1]

    # Configuration
    config = {
        "D": 100,
        "L": 15,
        "alpha": alpha,
        "rho": 2.0,
        "beta": 1.0,
        "lam_list": [0.01],
        "Delta_list": [0.0],
        "Delta_in": 0.5,
        "samples": 1,
        "T": 50000,
        "learning_rate": 0.1,
        "norm_init": 1.0,
        "tol": 1e-6,
        "N_test": 2000,
        "seq_len" : 30,
        "device": device,
        "logger": logger,
        "run_dir": run_dir,
        "run_index": run_index,
        'seed': 42,
    }

    config["R"] = int(config["rho"] * config["D"])
    config["N_train"] = int(config["alpha"] * config["D"]**2)
    config["N_total"] = config["N_train"] + config["N_test"]

    # --- Header log ---
    logger.info("========================================\n🧪 EXPERIMENT START\n----------------------------------------")
    df_results = run_experiment(config)
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

    # --- Footer log ---
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