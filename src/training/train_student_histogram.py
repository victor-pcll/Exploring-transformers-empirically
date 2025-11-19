import torch
from torch.utils.data import DataLoader
import src.model.Net_MLP as Net
import numpy as np

def train_student_on_data(config, lmbda, train_dataset):
    """
    Full-batch training for student network.
    Returns:
        W_student: learned student weights
        data_loss_final: data loss
        reg_loss_final: regularization loss
        attn_matrix_final: attention matrix from last forward pass (on a sample)
        seq_sample: input sequence corresponding to attn_matrix_final
    """

    # --- Initialize the student network ---
    student = Net(config["D"], config["R"], config["L"], config["T"], norm=config["norm_init"], beta=config["beta"], device=config["device"]) # init student network
    optimizer = torch.optim.Adam(student.parameters(), lr=config["learning_rate"]) # optimizer
    
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    X_full, y_full = next(iter(train_loader))
    X_full = X_full.long().to(config["device"])
    y_full = y_full.to(config["device"])
    y_full = y_full.unsqueeze(-1).expand(-1, -1, config["T"])  # y_full.shape = (N, T, T)

    y_counts = y_full[:, :, 0].float()           # shape: (N, L)
    y_true = y_counts / y_counts.sum(dim=1, keepdim=True)  # normalisation par séquence

    loss_prev = None
    attn_matrix_final = None
    seq_sample = None

    # --- Training loop ---
    for t in range(config["max_iter"]):
        optimizer.zero_grad()
        lam_stud = lmbda / np.sqrt(config["rho"])

        _, h_pred = student(X_full, delta_in=0.0)

        # --- Loss ---
        data_loss = torch.mean((h_pred - y_true) ** 2)
        reg_loss = lam_stud * torch.sum(student.fc1.weight ** 2)
        total_loss = data_loss + reg_loss

        total_loss.backward()
        optimizer.step()

        loss_cur = float(total_loss.item())
        if loss_prev is not None and abs(loss_cur - loss_prev) < config["tol"] and t > 100:
            break
        loss_prev = loss_cur

    # --- Evaluation after training ---
    with torch.no_grad():
        lam_stud2 = lmbda / np.sqrt(config["rho"])
        _, h_final = student(X_full, delta_in=0.0)
        data_loss_final = torch.sum((h_final - y_true) ** 2).item()
        reg_loss_final = (lam_stud2 * torch.sum(student.fc1.weight ** 2)).item()

        # Pick a sample to visualize
        seq_sample = X_full[0].detach().cpu().numpy()
        attn_out, _ = student(X_full[0].unsqueeze(0), delta_in=0.0)
        attn_matrix_final = attn_out[0].detach().cpu().numpy()

    return student, data_loss_final, reg_loss_final, attn_matrix_final, seq_sample