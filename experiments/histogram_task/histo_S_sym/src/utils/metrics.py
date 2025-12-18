import torch

def accuracy(pred, true, return_tensor=False):
    """
    Compute accuracy by comparing argmax of predictions to true labels.
    
    Args:
        pred: logits or predictions (torch.Tensor), shape (N, T) or (N, T, L)
        true: true labels (torch.Tensor), shape (N, T)
        return_tensor: if True, returns a torch.Tensor instead of float
    
    Returns:
        accuracy: fraction of correctly predicted positions
    """
    # Ensure tensors
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(true, torch.Tensor):
        true = torch.tensor(true)
    
    # If predictions are 3D (NxTxL), take argmax along last dimension
    if pred.ndim == 3:
        pred = pred.argmax(dim=-1)
    
    # Compare predictions with true labels
    correct = (pred == true).float()
    acc_value = correct.mean()
    
    if return_tensor:
        return acc_value
    else:
        return acc_value.item()