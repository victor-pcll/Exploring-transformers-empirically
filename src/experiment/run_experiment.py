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
from src.utils.clean_value import clean_value

def run_experiment(config):

    all_results = []                    # To store all results
    attn_runs_all = []
    y_runs_all = []
    seq_runs_all = []
    attn_runs = []                      # To store attention matrices
    seq_runs = []                       # To store sample sequences associated with attention matrices    

    # Pour stocker tous les h_pred_sample et y_true_sample pour tous les runs
    student_pred_samples_all = []   # stores all student predictions across runs
    teacher_true_samples_all = []   # stores all ground-truth teacher outputs across runs

    os.makedirs(config["run_dir"], exist_ok=True)

    for lam_cur in config["lam_list"]:

        acc_runs = []
        rank_before_runs = []
        rank_after_runs = []

        # Storage
        label_err_runs, train_data_runs, train_reg_runs, total_loss_runs, W_runs = [], [], [], [], []
        attn_runs_samples = []
        seq_runs_samples = []
        acc_train = []
        acc_last = []

        # Pour stocker les runs de test pour ce set d'hyperparams
        test_seq_list_samples = []
        attn_student_list_samples = []

        train_dataset, valid_dataset, test_dataset, _ = prepare_dataset(config)

        for _ in range(config["samples"]):

            # --- Entraînement du student ---
            student_trained, data_loss_i, reg_loss_i, acc = train_student_on_data(config, lam_cur, train_dataset)
            rank_before = np.linalg.matrix_rank(student_trained.W0.weight.detach().cpu().numpy())
            rank_before_runs.append(rank_before)
            acc_train.append(acc)
            last_acc = acc[-100:] if len(acc) >= 100 else acc   
            acc_last.append(np.mean(last_acc) if len(last_acc) > 0 else 0.0)

            # --- Fine-tuning du student ---
            student_fine_tune, _, _ = fine_tune_student(config, lam_cur, valid_dataset, student_trained)
            W_runs.append(student_fine_tune.W0.weight.detach().cpu().numpy())
            rank_after = np.linalg.matrix_rank(student_fine_tune.W0.weight.detach().cpu().numpy())
            rank_after_runs.append(rank_after)

            # --- Évaluation sur le test dataset ---
            student_fine_tune.eval()
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

            for X_test_full, y_test_full in test_loader:
                X_test_full = X_test_full.long().to(config["device"])
                y_test_full = y_test_full.cpu().numpy()  

                with torch.no_grad():
                    A_student_test, y_student_test = student_fine_tune(X_test_full, delta_in=0.0)
                    y_counts_student = y_student_test.detach().cpu().numpy()

                label_err_i = np.mean((y_counts_student - y_test_full) ** 2)

                student_pred_samples_all.append(y_counts_student)
                teacher_true_samples_all.append(y_test_full)

            # --- Enregistrement des résultats ---
            label_err_runs.append(label_err_i)
            train_data_runs.append(data_loss_i)
            train_reg_runs.append(reg_loss_i)
            total_loss_runs.append(data_loss_i + reg_loss_i)

            # Conversion en numpy pour stockage / pickle, suppression des dims inutiles
            seq_np = np.squeeze(X_test_full.cpu().numpy())
            attn_student_np = np.squeeze(A_student_test.cpu().numpy())
            y_counts_student_np = np.squeeze(y_counts_student)

            attn_runs_all.append(attn_student_np)
            y_runs_all.append(y_counts_student_np)
            seq_runs_all.append(seq_np)

            # Sauvegarde locale dans des listes pour analyse ultérieure
            test_seq_list_samples.append(seq_np)
            attn_student_list_samples.append(attn_student_np)

            attn_runs_samples.append(attn_student_np)
            seq_runs_samples.append(seq_np)
            acc_runs.append(accuracy(y_counts_student, y_test_full))

        # Clean values before computing means
        train_data_runs = [clean_value(x) for x in train_data_runs]
        train_reg_runs = [clean_value(x) for x in train_reg_runs]
        total_loss_runs = [clean_value(x) for x in total_loss_runs]
        label_err_runs = [clean_value(x) for x in label_err_runs]

        attn_runs.append(np.mean(attn_runs_samples, axis=0))  # shape: (seq_len, seq_len)
        seq_runs.append(seq_runs_samples[0])  # shape: (seq_len,)

        acc_train_clean = []
        for sub in acc_train:
            if isinstance(sub, (list, np.ndarray)):
                acc_train_clean.append([float(x) for x in sub])
            else:
                acc_train_clean.append([float(sub)])

        results = {
            "alpha": config["alpha"],
            "lam": lam_cur,
            "rho": config["rho"],
            "label_err_mean": float(np.mean(label_err_runs)/config["D"]**2 if len(label_err_runs) > 0 else 0.0),
            "label_err_std": float(np.std(label_err_runs, ddof=1)/config["D"]**2 if len(label_err_runs) > 1 else 0.0),
            "train_data_mean": float(np.mean(train_data_runs)/config["D"]**2 if len(train_data_runs) > 0 else 0.0),
            "train_reg_mean": float(np.mean(train_reg_runs) / config["D"]**2 if len(train_reg_runs) > 0 else 0.0),
            "train_total_mean": float(np.mean(total_loss_runs)/config["D"]**2 if len(total_loss_runs) > 0 else 0.0),
            "acc_test_mean": float(np.mean(acc_runs) if len(acc_runs) > 0 else 0.0),
            "acc_test_std": float(np.std(acc_runs, ddof=1) if len(acc_runs) > 1 else 0.0),
            "acc_train_last": float(np.mean(acc_last) if len(acc_last) > 0 else 0.0),
            "rank_before_mean": float(np.mean(rank_before_runs) if len(rank_before_runs) > 0 else 0.0),
            "rank_before_std": float(np.std(rank_before_runs, ddof=1) if len(rank_before_runs) > 1 else 0.0),
            "rank_after_mean": float(np.mean(rank_after_runs) if len(rank_after_runs) > 0 else 0.0),
            "rank_after_std": float(np.std(rank_after_runs, ddof=1) if len(rank_after_runs) > 1 else 0.0),
        }

        all_results.append(results)
        config["logger"].info(f"🔹 [alpha={config['alpha']:.4f}, lambda={lam_cur:.4f}] → label_err={results['label_err_mean']:.6f}")

    # --- 1. Save CSV with only scalar metrics ---
    df_results = pd.DataFrame(all_results)
    logs_csv_path = os.path.join(config['run_dir'], f"logs_{config['run_index']}.csv")
    df_results.to_csv(logs_csv_path, index=False)

    # --- 2. Save heavy objects (weights, attention matrices, sequences, accuracy) ---
    heavy_pickle_path = os.path.join(config['run_dir'], f"heavy_data_{config['run_index']}.pkl")
    with open(heavy_pickle_path, "wb") as f:
        pickle.dump({
            "W_runs": W_runs,
            "attn_matrices": attn_runs_all,
            "seq_samples": seq_runs_all,
            "y_samples": y_runs_all,
            "acc_train": acc_train_clean
        }, f)

    # --- 3. Save predictions separately ---
    preds_pickle_path = os.path.join(config['run_dir'], f"preds_{config['run_index']}.pkl")
    with open(preds_pickle_path, "wb") as f:
        pickle.dump({
            "student_pred_samples": student_pred_samples_all,
            "teacher_true_samples": teacher_true_samples_all
        }, f)

    config["logger"].info(f"💾 Results saved for run_index={config['run_index']}")
    return df_results