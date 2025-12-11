import torch
import numpy as np
from src.models import Net

def init_torch(seed=42, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if verbose:
        print(f"[INFO] Random seeds initialized to {seed}")


def train_student_on_data(D, L, R, beta, lam, x_train, y_train,
                          rho=1.0, T=1000, learning_rate=0.02,
                          norm_init=1.0, tol=1e-8):
    student = Net(D, R, L, norm=norm_init, beta=beta)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
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

    with torch.no_grad():
        y_pred_final = student(x_train, delta_in=0.0)
        data_loss_final = torch.sum((y_pred_final - y_train)**2).item()
        reg_loss_final = lam_stud * torch.sum(student.fc1.weight**2).item()

    W_student = student.fc1.weight.detach().cpu().numpy()
    return W_student, data_loss_final, reg_loss_final