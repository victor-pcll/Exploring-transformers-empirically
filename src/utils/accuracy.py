import numpy as np
import torch

def accuracy(pred, true):
    """
    Compare chaque bin individuellement après arrondi.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    pred_rounded = np.round(pred)
    correct = np.abs(pred_rounded - true) < 0.5

    return correct.mean()
