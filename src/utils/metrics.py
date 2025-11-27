import numpy as np
import torch

def accuracy(pred, true):
    """
    Compute the accuracy for count predictions.
    Args:
        pred: predicted counts (numpy array or torch tensor)
        true: true counts (numpy array or torch tensor)
    Returns:
        accuracy: proportion of predictions within 0.5 of the true count
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    pred_rounded = np.round(pred)
    correct = np.abs(pred_rounded - true) < 0.5

    return correct.mean()