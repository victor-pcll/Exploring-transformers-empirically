import os 
import torch
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from dataset.histogram import prepare_dataset
from training.train_student_histogram import train_student_on_data

def run_experiment(config):

    all_results = []                    # To store all results
    attn_runs = []                      # To store attention matrices
    seq_runs = []                       # To store sample sequences associated with attention matrices    

    os.makedirs(config["run_dir"], exist_ok=True)

    for lam_cur in config["lam_list"]:
        for _ in config["Delta_list"]:

            # Storage
            MSE_runs, label_err_runs, train_data_runs, train_reg_runs, total_loss_runs, W_runs = [], [], [], [], [], []
            attn_runs_samples = []
            seq_runs_samples = []

            # Pour stocker les runs de test pour ce set d'hyperparams
            test_seq_list_samples = []
            attn_teacher_list_samples = []
            attn_student_list_samples = []

            train_dataset, test_dataset = prepare_dataset(config)

            for _ in range(config["samples"]):

                student_trained, data_loss_i, reg_loss_i, attn_matrix, seq_sample = train_student_on_data(
                    config, lam_cur, train_dataset
                )
                W_runs.append(student_trained.fc1.weight.detach().cpu().numpy())
                attn_runs_samples.append(attn_matrix)
                seq_runs_samples.append(seq_sample)

                student_eval = student_trained
                student_eval.eval()

                # On récupère tout le batch test d'un coup (pour avoir tous les comptages)
                test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

                for X_test_full, y_test_full in test_loader:
                    X_test_full = X_test_full.long().to(config["device"])
                    y_test_full = y_test_full.cpu().numpy()  # fix important

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


    config["logger"].info(f"💾 Results saved for run_index={config['run_index']}")
    return df_results