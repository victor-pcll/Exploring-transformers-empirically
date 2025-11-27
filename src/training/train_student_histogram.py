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
    student = Net(config["D"], config["R"], config["MLP_dim"], config["L"], config["T"], norm=config["norm_init"], beta=config["beta"], device=config["device"])
    optimizer = torch.optim.Adam(student.parameters(), lr=config["learning_rate"])

    criterion = torch.nn.MSELoss()
    
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
            y_batch = y_batch.float().to(config["device"])
            
            optimizer.zero_grad()
            _, y_student = student(X_batch, delta_in=0.0)
            
            # --- Calcul de la perte ---
            data_loss = criterion(y_student, y_batch)
            
            reg_loss = 0.0
            if config["lambda"] > 0:
                reg_loss = config["lambda"] * sum(p.pow(2).sum() for p in student.parameters())

            total_loss = data_loss + reg_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item() * X_batch.size(0)  # somme pondérée par batch size

            with torch.no_grad():
                acc_epoch.append(accuracy(y_student, y_batch))
        
        # --- Arrêt anticipé basé sur la convergence ---
        loss_cur = epoch_loss / len(train_dataset)
        acc_train.append(sum(acc_epoch) / len(acc_epoch))
        if loss_prev is not None and abs(loss_cur - loss_prev) < config["tol"] and t > 5:
            break
        loss_prev = loss_cur

    # --- Évaluation finale sur tout le dataset (via DataLoader, compatible Subset) ---
    student.eval()
    full_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    with torch.no_grad():
        X_full, y_full = next(iter(full_loader))
        X_full = X_full.long().to(config["device"])
        y_full = y_full.float().to(config["device"])

        _, y_student_f = student(X_full, delta_in=0.0)

        data_loss_final = torch.mean((y_student_f - y_full) ** 2).item()
        reg_loss_final = config["lambda"] * sum(torch.sum(p**2) for p in student.parameters()) if config["lambda"] > 0 else 0.0

    return student, data_loss_final, reg_loss_final, acc_train