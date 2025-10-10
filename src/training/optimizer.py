import torch
import numpy as np
from tqdm import trange


def train(
    model,
    data,
    target,
    lam=0.0,
    rho=1.0,
    n_iter=1000,
    learning_rate=0.01,
    tol=1e-6,
    min_iter=100,
    desc="Training",
):
    """
    Train a model with L2 regularization (like train_student_on_data).
    
    Args:
        model: PyTorch model (e.g., Net)
        data: input tensor (N, L, D)
        target: target tensor
        lam: regularization strength λ
        rho: scaling parameter (default 1.0)
        n_iter: number of training iterations
        learning_rate: learning rate for Adam optimizer
        tol: early stopping tolerance
        min_iter: minimum iterations before checking convergence
        desc: tqdm progress bar label
    Returns:
        (model, data_loss_final, reg_loss_final, total_loss_history)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    prev_total_loss = float("inf")
    losses = []

    for t in trange(n_iter, desc=desc, leave=False):
        optimizer.zero_grad()

        lam_scaled = lam / np.sqrt(rho)
        output = model(data)
        data_loss = torch.sum((output - target) ** 2)
        reg_loss = lam_scaled * torch.sum(model.fc1.weight ** 2)
        total_loss = data_loss + reg_loss

        total_loss.backward()
        optimizer.step()

        cur = total_loss.item()
        losses.append(cur)

        # Early stopping after min_iter
        if t > min_iter and abs(cur - prev_total_loss) < tol:
            break
        prev_total_loss = cur

    # Final evaluation
    with torch.no_grad():
        lam_scaled = lam / np.sqrt(rho)
        output_final = model(data)
        data_loss_final = torch.sum((output_final - target) ** 2).item()
        reg_loss_final = (lam_scaled * torch.sum(model.fc1.weight ** 2)).item()

    return model, data_loss_final, reg_loss_final, losses