import torch
from tqdm import trange

def train(model, data, target, loss_fn, optimizer, n_iter=1000, tol=1e-6):
    """Train the model and return loss history."""
    losses = []
    prev_loss = float("inf")

    for t in trange(n_iter, desc="Training"):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if abs(prev_loss - loss.item()) < tol:
            break
        prev_loss = loss.item()

    return losses