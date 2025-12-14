import torch
import numpy as np
from src.model import Net

def train_student_on_data(D, L, R, beta, lam, x_train, y_train, rho=1.0, T=1000, learning_rate=0.02, norm_init=1.0, tol=1e-8):
    """
    Entraîne le modèle Student sur les données générées par le Teacher.
    """
    student = Net(D, R, L, norm=norm_init, beta=beta)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    prev_total_loss = None
    
    for t in range(T):
        optimizer.zero_grad()
        
        # Ajustement du paramètre de régularisation selon rho (tel que dans le code original)
        lam_stud = lam / np.sqrt(rho)
        
        y_pred = student(x_train, delta_in=0.0)
        
        # Fonction de perte: MSE sur la sortie + Régularisation L2 sur les poids W_Q et W_K
        data_loss = torch.sum((y_pred - y_train)**2)
        reg_loss = lam_stud * (torch.sum(student.W_Q.weight ** 2) + torch.sum(student.W_K.weight ** 2))
        total_loss = data_loss + reg_loss
        
        total_loss.backward()
        optimizer.step()
        
        cur = float(total_loss.item())
        # Critère d'arrêt: la variation de la perte est inférieure à tol après 100 itérations
        if prev_total_loss is not None and abs(cur - prev_total_loss) < tol and t > 100:
            break
        prev_total_loss = cur
        
    # Calcul des pertes finales sans calcul de gradient
    with torch.no_grad():
        lam_stud2 = lam / np.sqrt(rho)
        y_pred_final = student(x_train, delta_in=0.0)
        data_loss_final = torch.sum((y_pred_final - y_train)**2).item()
        reg_loss_final = (lam_stud2 * (torch.sum(student.W_Q.weight ** 2) + torch.sum(student.W_K.weight ** 2))).item()
        
    # Détacher et convertir les poids finaux
    W_Q_student = student.W_Q.weight.detach().cpu().numpy()
    W_K_student = student.W_K.weight.detach().cpu().numpy()
    
    return W_Q_student, W_K_student, data_loss_final, reg_loss_final


def compute_S_from_W(W_Q, W_K, R, D):
    """
    Calcule la matrice S = (W_Q^T W_K) / sqrt(R * D).
    W_Q et W_K sont des matrices (R, D) en numpy.
    """
    # W_Q.T @ W_K est une multiplication matricielle (D, R) @ (R, D) -> (D, D)
    return (W_Q.T @ W_K) / np.sqrt(R * D)


def S_MSE(W_Q_student, W_K_student, W_Q_teacher, W_K_teacher, R, R_star, D):
    """
    Calcule l'erreur quadratique moyenne entre les matrices S du Student et du Teacher.
    """
    S_stud = compute_S_from_W(W_Q_student, W_K_student, R, D)
    S_teach = compute_S_from_W(W_Q_teacher, W_K_teacher, R_star, D)
    # MSE = Somme((S_stud - S_teach)^2) / D
    return float(((S_stud - S_teach) ** 2).sum() / D**2)