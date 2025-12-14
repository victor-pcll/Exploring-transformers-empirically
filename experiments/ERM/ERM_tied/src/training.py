import torch
import numpy as np
from src.model import Net

def train_student_on_data(D, L, R, beta, lam, x_train, y_train, rho=1.0, T=1000, learning_rate=0.02, norm_init=1.0, tol=1e-8, device="cpu"):
    """Entraîne le student pour minimiser la perte MSE + régularisation L2."""
    student = Net(D, R, L, norm=norm_init, beta=beta, device=device)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    prev_total_loss = None

    for t in range(T):
        optimizer.zero_grad()
        lam_stud = lam / np.sqrt(rho)
        y_pred = student(x_train, delta_in=0.0)
        
        data_loss = torch.sum((y_pred - y_train)**2)
        reg_loss = lam_stud * torch.sum(student.fc1.weight**2)
        total_loss = data_loss + reg_loss
        
        total_loss.backward()
        optimizer.step()

        cur = float(total_loss.item())
        if prev_total_loss is not None and abs(cur - prev_total_loss) < tol and t > 100:
            break
        prev_total_loss = cur

    # Calcul final sans gradient
    with torch.no_grad():
        lam_stud2 = lam / np.sqrt(rho)
        y_pred_final = student(x_train, delta_in=0.0)
        data_loss_final = torch.sum((y_pred_final - y_train)**2).item()
        reg_loss_final = (lam_stud2 * torch.sum(student.fc1.weight**2)).item()

    return student, data_loss_final, reg_loss_final

def compute_S_from_W(W, R, D):
    """Calcule S = (W^T W) / sqrt(R*D) pour l'architecture symétrique."""
    # W est (R x D) -> W.t() @ W donne (D x D)
    S = (W.t() @ W) / torch.sqrt(torch.tensor(R * D, dtype=W.dtype, device=W.device))
    return S

def S_MSE(W_student, W_teacher, R, R_star, D):
    """Calcule le MSE entre les matrices S du student et du teacher."""
    # Conversion numpy -> tensor si nécessaire
    if isinstance(W_teacher, np.ndarray):
        W_teacher = torch.tensor(W_teacher, dtype=W_student.dtype, device=W_student.device)

    S_stud = compute_S_from_W(W_student, R, D)
    S_teach = compute_S_from_W(W_teacher, R_star, D)

    # MSE normalisé par D
    return float(((S_stud - S_teach)**2).sum().cpu().item() / D)