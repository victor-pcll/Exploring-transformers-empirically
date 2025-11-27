import torch
from torch.utils.data import DataLoader
import src.model.Net_MLP as Net

def fine_tune_student(config, train_dataset, student_trained):
    """
    Fine-tune only the W0 matrix of the student network in mini-batches.
    Args:
        config: dictionary containing hyperparameters and dimensions
        train_dataset: PyTorch dataset for training
        student_trained: pre-trained student model to be fine-tuned
    Returns:
        student_trained: fine-tuned student model
        data_loss_final: final data loss across the entire dataset
        reg_loss_final: final regularisation loss
    """
    # --- Bloc W0 uniquement trainable ---
    for p in student_trained.parameters():
        p.requires_grad = False
    student_trained.W0.weight.requires_grad = True

    optimizer = torch.optim.Adam([student_trained.W0.weight], lr=config["learning_rate_fine_tune"])
    criterion = torch.nn.MSELoss()
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    loss_prev = None

    for t in range(config["max_fine_tune_iter"]):
        student_trained.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.long().to(config["device"])
            y_batch = y_batch.float().to(config["device"])
            
            optimizer.zero_grad()
            _, y_student = student_trained(X_batch, delta_in=0.0)
            
            data_loss = criterion(y_student, y_batch)
            
            reg_loss = config["lambda"] * torch.sum(student_trained.W0.weight**2) if config["lambda"] > 0 else 0.0
            
            total_loss = data_loss + reg_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item() * X_batch.size(0)
        
        # --- Early stopping basé sur la convergence ---
        loss_cur = epoch_loss / len(train_dataset)
        if loss_prev is not None and abs(loss_cur - loss_prev) < config["tol"] and t > 5:
            break
        loss_prev = loss_cur

    # --- Évaluation finale sur tout le dataset (via DataLoader, compatible Subset) ---
    student_trained.eval()
    full_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    with torch.no_grad():
        X_full, y_full = next(iter(full_loader))
        X_full = X_full.long().to(config["device"])
        y_full = y_full.float().to(config["device"])

        _, y_student_f = student_trained(X_full, delta_in=0.0)

        data_loss_final = torch.mean((y_student_f - y_full) ** 2).item()
        reg_loss_final = config["lambda"] * torch.sum(student_trained.W0.weight**2).item() if config["lambda"] > 0 else 0.0

    return student_trained, data_loss_final, reg_loss_final