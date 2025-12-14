import os 
import torch
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from src.dataset.split_dataset import prepare_dataset
from src.training.train_student_histogram import train_student_on_data
from src.training.Fine_tune_histogram_task import fine_tune_student
from src.utils.conversion import convert_to_numpy, clean_list, clean_accuracy_list
from src.utils.statistics import safe_mean, safe_std
from src.utils.metrics import accuracy
from src.utils.evaluation import evaluate_student
from src.training.cross_validation import cross_validate_regularization

def run_experiment(config):
    """
    Run experiments for different MLP dimensions and log results.
    Args:
        config: dictionary containing hyperparameters and settings
    Returns:
        df_results: DataFrame containing results for each MLP dimension
    """

    device = config["device"]
    D = config["D"]
    alpha = config["alpha"]
    rho = config["rho"]
    run_dir = config["run_dir"]
    run_index = config["run_index"]
    logger = config["logger"]

    os.makedirs(run_dir, exist_ok=True)

    all_results = []

    student_predictions_all_runs = []
    teacher_true_outputs_all_runs = []
    test_sequences_all_runs = [] 

    for mlp_dim in config["d_mlp_list"]:
        config["MLP_dim"] = mlp_dim

        acc_test_runs = []
        rank_before_runs = []
        rank_after_runs = []

        label_err_runs = []
        train_data_loss_runs = []
        train_reg_loss_runs = []
        total_loss_runs = []
        weights_runs = []

        acc_train_runs = []
        acc_fine_tune_runs = []
        acc_train_last_runs = []

        train_dataset, valid_dataset, test_dataset, _ = prepare_dataset(config)
        config["lambda"] = cross_validate_regularization(config, train_dataset, config["k_folds"])

        for _ in range(config["samples"]):

            # --- Training the student ---
            student_trained, data_loss, reg_loss, acc_train = train_student_on_data(config, train_dataset)
            rank_before = np.linalg.matrix_rank(convert_to_numpy(student_trained.S))
            rank_before_runs.append(rank_before)
            acc_train_runs.append(acc_train)
            last_acc_values = acc_train[-100:] if len(acc_train) >= 100 else acc_train
            acc_train_last_runs.append(np.mean(last_acc_values) if len(last_acc_values) > 0 else 0.0)

            # --- Fine-tuning the student ---
            student_fine_tuned, _, _, acc_fine_tune = fine_tune_student(config, valid_dataset, student_trained)
            weights_runs.append(convert_to_numpy(student_fine_tuned.S))
            rank_after = np.linalg.matrix_rank(convert_to_numpy(student_fine_tuned.S))
            rank_after_runs.append(rank_after)
            last_acc_values = acc_fine_tune[-100:] if len(acc_fine_tune) >= 100 else acc_fine_tune
            acc_fine_tune_runs.append(np.mean(last_acc_values) if len(last_acc_values) > 0 else 0.0)

            # --- Evaluation on test dataset ---
            y_pred_logits, y_true, _, X_test = evaluate_student(student_fine_tuned, test_dataset, device)

            # Convert logits to integer labels by argmax over classes (last dimension)
            y_pred = np.argmax(y_pred_logits, axis=-1)

            # Compute label error as proportion of incorrect positions
            label_err = np.mean(y_pred != y_true)

            student_predictions_all_runs.append(y_pred)
            teacher_true_outputs_all_runs.append(y_true)
            test_sequences_all_runs.append(X_test) 

            label_err_runs.append(label_err)
            train_data_loss_runs.append(data_loss)
            train_reg_loss_runs.append(reg_loss)
            total_loss_runs.append(data_loss + reg_loss)

            # Accuracy on test set
            acc_test_runs.append(accuracy(y_pred, y_true))

        # Clean values before computing means
        train_data_loss_runs = clean_list(train_data_loss_runs)
        train_reg_loss_runs = clean_list(train_reg_loss_runs)
        total_loss_runs = clean_list(total_loss_runs)
        label_err_runs = clean_list(label_err_runs)

        acc_train_clean = clean_accuracy_list(acc_train_runs)

        results = {
            "alpha": alpha,
            "MLP_dim": mlp_dim,
            "rho": rho,
            "label_err_mean": safe_mean(label_err_runs),
            "label_err_std": safe_std(label_err_runs),
            "train_data_mean": safe_mean(train_data_loss_runs, divisor=D**2),
            "train_reg_mean": safe_mean(train_reg_loss_runs, divisor=D**2),
            "train_total_mean": safe_mean(total_loss_runs, divisor=D**2),
            "train_total_std": safe_std(total_loss_runs, divisor=D**2),
            "acc_test_mean": safe_mean(acc_test_runs),
            "acc_test_std": safe_std(acc_test_runs),
            "acc_fine_tune_mean": safe_mean(acc_fine_tune_runs),
            "acc_fine_tune_std": safe_std(acc_fine_tune_runs),
            "acc_train_mean": safe_mean(acc_train_last_runs),
            "acc_train_std": safe_std(acc_train_last_runs),
            "rank_before_mean": safe_mean(rank_before_runs),
            "rank_before_std": safe_std(rank_before_runs),
            "rank_after_mean": safe_mean(rank_after_runs),
            "rank_after_std": safe_std(rank_after_runs),
        }

        all_results.append(results)
        logger.info(f"🔹 [alpha={alpha:.4f}, MLP_dim={mlp_dim}] → label_err={results['label_err_mean']:.6f}")

    # --- Save CSV with scalar metrics ---
    df_results = pd.DataFrame(all_results)
    logs_csv_path = os.path.join(run_dir, f"logs_{run_index}.csv")
    df_results.to_csv(logs_csv_path, index=False)

    # --- Save heavy objects (weights, accuracy) ---
    heavy_pickle_path = os.path.join(run_dir, f"heavy_data_{run_index}.pkl")
    with open(heavy_pickle_path, "wb") as f:
        pickle.dump({
            "W_runs": weights_runs,
            "acc_train": acc_train_clean
        }, f)

    # --- Save predictions separately ---
    preds_pickle_path = os.path.join(run_dir, f"preds_{run_index}.pkl")
    with open(preds_pickle_path, "wb") as f:
        pickle.dump({
            "student_pred_samples": student_predictions_all_runs,
            "teacher_true_samples": teacher_true_outputs_all_runs,
            "sequence_samples": test_sequences_all_runs
        }, f)

    logger.info(f"💾 Results saved for run_index={run_index}")
    return df_results