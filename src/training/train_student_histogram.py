import torch
from torch.utils.data import DataLoader
from src.model.Net_MLP import Net
from src.utils.accuracy import accuracy

def train_student_on_data(config, lmbda, train_dataset):
    """
    Training du student network en mini-batch.

    Args:
        config: dictionnaire contenant les hyperparamètres et dimensions
        lmbda: coefficient de régularisation (si utilisé)
        train_dataset: dataset PyTorch pour l'entraînement

    Returns:
        student: modèle entraîné
        data_loss_final: perte de données finale sur tout le dataset
        reg_loss_final: perte de régularisation finale
    """
    # --- Initialisation du réseau et de l'optimiseur ---
    student = Net(config["D"], config["R"], config["MLP_dim"], config["L"], config["T"], norm=config["norm_init"], beta=config["beta"], device=config["device"])
    optimizer = torch.optim.Adam(student.parameters(), lr=config["learning_rate"])
    
    # --- DataLoader avec mini-batch ---
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    loss_prev = None
    acc_train = []

    # --- Boucle d'entraînement ---
    for t in range(config["max_iter"]):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.long().to(config["device"])
            y_batch = y_batch.to(config["device"])
            
            optimizer.zero_grad()
            _, y_student = student(X_batch, delta_in=0.0)
            
            # --- Calcul de la perte ---
            data_loss = torch.mean((y_student - y_batch) ** 2)
            
            reg_loss = 0.0
            if lmbda > 0:
                reg_loss = lmbda * sum(torch.sum(p**2) for p in student.parameters())
            
            total_loss = data_loss + reg_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item() * X_batch.size(0)  # somme pondérée par batch size

            acc_train.append(accuracy(y_student, y_batch))
        
        # --- Arrêt anticipé basé sur la convergence ---
        loss_cur = epoch_loss / len(train_dataset)
        if loss_prev is not None and abs(loss_cur - loss_prev) < config["tol"] and t > 1000:
            break
        loss_prev = loss_cur

    # --- Évaluation finale sur tout le dataset ---
    full_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    with torch.no_grad():
        X_full, y_full = next(iter(full_loader))
        X_full = X_full.long().to(config["device"])
        y_full = y_full.to(config["device"])
        _, y_student_f = student(X_full, delta_in=0.0)
        data_loss_final = torch.mean((y_student_f - y_full) ** 2).item()
        reg_loss_final = lmbda * sum(torch.sum(p**2) for p in student.parameters()) if lmbda > 0 else 0.0

    return student, data_loss_final, reg_loss_final, acc_train