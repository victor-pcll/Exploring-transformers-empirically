import numpy as np
import torch

def evaluate_student(student_model, dataset, device):
    """
    Evaluate the student model on the given dataset.
    Args:
        student_model: the student model to evaluate
        dataset: dataset object with X and y attributes
        device: computation device (CPU or GPU)
    Returns:
        y_pred: predicted outputs as a numpy array
        y_true: true outputs as a numpy array
        attn_matrix: attention matrix from the student model as a numpy array
        X: input sequences as a numpy array
    """
    from torch.utils.data import DataLoader

    full_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    X, y = next(iter(full_loader))
    X = X.long().to(device)
    y_true = y.cpu().numpy()

    with torch.no_grad():
        A_student, y_student = student_model(X, delta_in=0.0)
        y_pred = y_student.detach().cpu().numpy()
        attn_matrix = A_student.cpu().numpy()

    return y_pred, y_true, attn_matrix, X.cpu().numpy().reshape(-1)
