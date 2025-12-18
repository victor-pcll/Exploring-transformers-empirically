import torch
from torch.utils.data import DataLoader
from src.model.Net_MLP import Net
from src.utils.metrics import accuracy

def train_student_on_data(config, train_dataset):
    """
    Mini-batch training of the student network.
    Args:
        config: dictionary containing hyperparameters and dimensions
        train_dataset: PyTorch dataset for training
    Returns:
        student: trained model
        data_loss_final: final data loss across the entire dataset
        reg_loss_final: final regularisation loss
    """
    # --- Initialisation du réseau et de l'optimiseur ---
    student = Net(
        config["D"], config["R"], config["MLP_dim"], config["L"], config["T"],
        norm=config["norm_init"], beta=config["beta"], device=config["device"]
    )

    optimizer = torch.optim.Adam(student.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    # --- DataLoader avec mini-batch ---
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    loss_prev = None
    acc_train = []

    # --- Boucle d'entraînement ---
    for t in range(config["max_iter"]):
        student.train()
        epoch_loss = 0.0
        acc_epoch = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.long().to(config["device"])
            y_batch = y_batch.long().to(config["device"])

            optimizer.zero_grad()
            logits = student(X_batch)

            N, T, n_classes = logits.shape
            logits_flat = logits.view(N * T, n_classes)
            y_flat = y_batch.view(N * T)

            # --- Perte de données ---
            data_loss = criterion(logits_flat, y_flat)

            # --- Régularisation : DIRECTEMENT sur les paramètres ---
            reg_loss = 0.0
            if config["lambda"] > 0:
                reg_loss = config["lambda"] * sum(p.pow(2).sum() for p in student.parameters())

            total_loss = data_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            # Accumulation de la perte
            epoch_loss += total_loss.item() * X_batch.size(0)

            # Accuracy batch
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc_epoch.append(torch.mean((pred == y_batch).float()).item())

        # --- Convergence (early stopping) ---
        loss_cur = epoch_loss / len(train_dataset)
        acc_train.append(sum(acc_epoch) / len(acc_epoch))

        if loss_prev is not None and abs(loss_cur - loss_prev) < config["tol"] and t > 5:
            break

        loss_prev = loss_cur

    # --- Évaluation finale complète ---
    student.eval()
    full_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    with torch.no_grad():
        X_full, y_full = next(iter(full_loader))
        X_full = X_full.long().to(config["device"])
        y_full = y_full.long().to(config["device"])

        logits_full = student(X_full)

        N_full, T_full, n_classes_full = logits_full.shape
        logits_full_flat = logits_full.view(N_full * T_full, n_classes_full)
        y_full_flat = y_full.view(N_full * T_full).long()

        # Perte finale
        data_loss_final = criterion(logits_full_flat, y_full_flat).item()

        # Régularisation finale (toujours sans .weight)
        reg_loss_final = (
            config["lambda"] * sum(torch.sum(p**2) for p in student.parameters())
            if config["lambda"] > 0 else 0.0
        )

    return student, data_loss_final, reg_loss_final, acc_train