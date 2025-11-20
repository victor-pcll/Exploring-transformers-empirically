import os 
import torch
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from src.dataset.split_dataset import prepare_dataset
from src.training.train_student_histogram import train_student_on_data
from src.training.Fine_tune_histogram_task import fine_tune_student
from src.utils.accuracy import accuracy

def run_experiment(config):

    all_results = []                    # To store all results
    attn_runs = []                      # To store attention matrices
    seq_runs = []                       # To store sample sequences associated with attention matrices    

    # Pour stocker tous les h_pred_sample et y_true_sample pour tous les runs
    student_pred_samples_all = []   # stores all student predictions across runs
    teacher_true_samples_all = []   # stores all ground-truth teacher outputs across runs

    os.makedirs(config["run_dir"], exist_ok=True)

    for lam_cur in config["lam_list"]:

        acc_runs = []
        rank_runs = []

        # Storage
        MSE_runs, label_err_runs, train_data_runs, train_reg_runs, total_loss_runs, W_runs = [], [], [], [], [], []
        attn_runs_samples = []
        seq_runs_samples = []

        # Pour stocker les runs de test pour ce set d'hyperparams
        test_seq_list_samples = []
        attn_student_list_samples = []

        train_dataset, valid_dataset, test_dataset, _ = prepare_dataset(config)

        for _ in range(config["samples"]):

            # Modification: récupère h_pred_sample et y_true_sample
            student_trained, data_loss_i, reg_loss_i = train_student_on_data(config, lam_cur, train_dataset)
            student_fine_tune, _, _ = fine_tune_student(config, lam_cur, valid_dataset)
            W_runs.append(student_fine_tune.W0.weight.detach().cpu().numpy())

            rank = np.linalg.matrix_rank(student_fine_tune.W0.weight.detach().cpu().numpy())
            rank_runs.append(rank)

            student_fine_tune.eval()

            # On récupère tout le batch test d'un coup (pour avoir tous les comptages)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

            for X_test_full, y_test_full in test_loader:
                X_test_full = X_test_full.long().to(config["device"])
                y_test_full = y_test_full.cpu().numpy()  

                with torch.no_grad():
                    _, y_student_test = student_fine_tune(X_test_full, delta_in=0.0)
                    y_counts_student = y_student_test.detach().cpu().numpy()

                # Erreur quadratique moyenne
                label_err_i = np.mean((y_counts_student - y_test_full) ** 2)

                # Save student and teacher outputs for this sample
                student_pred_samples_all.append(y_counts_student)
                teacher_true_samples_all.append(y_test_full)

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

            attn_runs_samples.append(attn_student_np)
            seq_runs_samples.append(seq_np)
            acc = accuracy(np.argmax(y_counts_student, axis=1), np.argmax(y_test_full, axis=1))
            acc_runs.append(acc)

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
            "acc_mean": float(np.mean(acc_runs)),
            "acc_std": float(np.std(acc_runs, ddof=1)) if len(acc_runs) > 1 else 0.0,
            "rank_mean": float(np.mean(rank_runs)),
            "rank_std": float(np.std(rank_runs, ddof=1)) if len(rank_runs) > 1 else 0.0,
            "W_runs": W_runs
        }

        all_results.append(results)
        config["logger"].info(f"🔹 [alpha={config['alpha']:.4f}, lambda={lam_cur:.4f}] → MSE={results['MSE_mean']:.6f}")

    # Save CSV & pickle
    df_results = pd.DataFrame([{k:v for k,v in res.items() if k != "W_runs"} for res in all_results])
    logs_csv_path = os.path.join(config['run_dir'], f"logs_{config['run_index']}.csv")
    df_results.to_csv(logs_csv_path, index=False)  # Save experiment summary CSV

    W_runs_all = [res["W_runs"] for res in all_results]
    pickle_path = os.path.join(config['run_dir'], f"W_runs_{config['run_index']}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(W_runs_all, f)  # Save all weight matrices

    # Save attention matrices and sample sequences
    attn_save_path = os.path.join(config['run_dir'], f"attn_run{config['run_index']}.pkl")
    with open(attn_save_path, "wb") as f:
        pickle.dump({
            "attn_matrices": attn_runs,
            "seq_samples": seq_runs
        }, f)  # Save attention matrices and corresponding sequences

    # Ajout: sauvegarde des student_pred_samples et teacher_true_samples pour chaque run dans un pickle
    h_pred_y_true_save_path = os.path.join(config['run_dir'], f"h_pred_y_true_run{config['run_index']}.pkl")
    with open(h_pred_y_true_save_path, "wb") as f:
        pickle.dump({
            "student_pred_samples": student_pred_samples_all,
            "teacher_true_samples": teacher_true_samples_all
        }, f)  # Save all student predictions and teacher ground truths

    config["logger"].info(f"💾 Results saved for run_index={config['run_index']}")
    return df_results