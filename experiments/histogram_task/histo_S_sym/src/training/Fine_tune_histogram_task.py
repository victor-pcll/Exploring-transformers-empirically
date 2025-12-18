import torch
from torch.utils.data import DataLoader
import src.model.Net_MLP as Net
from src.utils.metrics import accuracy

def fine_tune_student(config, train_dataset, student_trained):

    # --- Seul S est trainable ---
    for p in student_trained.parameters():
        p.requires_grad = False

    # S est un nn.Parameter → pas de .weight
    student_trained.S.requires_grad = True

    optimizer = torch.optim.Adam([student_trained.S], lr=config["learning_rate_fine_tune"])
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    loss_prev = None
    acc_fine_tune = []

    for t in range(config["max_fine_tune_iter"]):
        student_trained.train()
        epoch_loss = 0.0
        acc_epoch = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.long().to(config["device"])
            y_batch = y_batch.long().to(config["device"])
            
            optimizer.zero_grad()
            logits = student_trained(X_batch, delta_in=0.0)
            
            N, T, n_classes = logits.shape
            logits_flat = logits.view(N*T, n_classes)
            y_flat = y_batch.view(N*T)
            
            data_loss = criterion(logits_flat, y_flat)
            
            # Idem : S sans .weight
            reg_loss = config["lambda"] * torch.sum(student_trained.S**2) if config["lambda"] > 0 else 0.0
            
            total_loss = data_loss + reg_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item() * X_batch.size(0)

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc_epoch.append(torch.mean((pred == y_batch).float()).item())

        loss_cur = epoch_loss / len(train_dataset)
        acc_fine_tune.append(sum(acc_epoch) / len(acc_epoch))

        if loss_prev is not None and abs(loss_cur - loss_prev) < config["tol"] and t > 5:
            break

        loss_prev = loss_cur

    # --- Evaluation finale ---
    student_trained.eval()
    full_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    with torch.no_grad():
        X_full, y_full = next(iter(full_loader))
        X_full = X_full.long().to(config["device"])
        y_full = y_full.long().to(config["device"])

        logits_full = student_trained(X_full)
        N_full, T_full, n_classes_full = logits_full.shape
        logits_full_flat = logits_full.view(N_full*T_full, n_classes_full)
        y_full_flat = y_full.view(N_full*T_full)

        data_loss_final = criterion(logits_full_flat, y_full_flat).item()

        # Idem ici
        reg_loss_final = config["lambda"] * torch.sum(student_trained.S**2).item() if config["lambda"] > 0 else 0.0

    return student_trained, data_loss_final, reg_loss_final, acc_fine_tune