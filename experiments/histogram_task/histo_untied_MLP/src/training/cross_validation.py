import torch
import numpy as np
from torch.utils.data import Subset
from src.utils.metrics import accuracy
from src.utils.evaluation import evaluate_student
from src.training.train_student_histogram import train_student_on_data

def cross_validate_regularization(config, train_dataset, k_folds=5):

    total_len = len(train_dataset)
    indices = torch.randperm(total_len).tolist()
    
    # 2. Calculer la taille d'un fold
    fold_size = total_len // k_folds
    
    results = {}

    for lam in config["lambda_candidates"]:
        val_accuracies = []
        config["lambda"] = lam
        
        for i in range(k_folds):

            val_idx_start = i * fold_size
            val_idx_end = (i + 1) * fold_size if i < k_folds - 1 else total_len
            
            val_indices = indices[val_idx_start:val_idx_end]
            train_indices = indices[:val_idx_start] + indices[val_idx_end:]
            
            fold_train_ds = Subset(train_dataset, train_indices)
            fold_val_ds = Subset(train_dataset, val_indices)
            
            student, _, _, _ = train_student_on_data(config, fold_train_ds)
            y_pred_logits, y_true, _, _ = evaluate_student(student, fold_val_ds, config["device"])
            
            y_pred = np.argmax(y_pred_logits, axis=-1)
            val_accuracies.append(accuracy(y_pred, y_true))
            
        results[lam] = np.mean(val_accuracies)

    return max(results, key=results.get)