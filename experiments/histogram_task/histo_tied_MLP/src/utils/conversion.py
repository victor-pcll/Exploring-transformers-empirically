import numpy as np
import torch

def clean_accuracy_list(acc_list):
    """
    Clean a list of accuracy values, ensuring each entry is a list of floats.
    Args:
        acc_list: list of accuracy values (could be floats, lists, or arrays)
    Returns:
        cleaned list of accuracy values as lists of floats
    """
    acc_clean = []
    for sub in acc_list:
        if isinstance(sub, (list, np.ndarray)):
            acc_clean.append([float(x) for x in sub])
        else:
            acc_clean.append([float(sub)])
    return acc_clean

def clean_list(values):
    """
    Clean a list of values by converting tensors and numpy scalars to floats.
    Args:
        values: list of values (tensors, numpy scalars, or floats)
    Returns:
        cleaned list of float values
    """
    return [clean_value(x) for x in values]

def convert_to_numpy(tensor):
    """
    Convert a PyTorch tensor to a NumPy array.
    Args:
        tensor: PyTorch tensor
    Returns:
        NumPy array
    """
    return tensor.detach().cpu().numpy()

def clean_value(x):
    """
    Convert tensor or numpy scalar to float for consistent storage.
    Args:
        x: input value (tensor, numpy scalar, or float)
    Returns:
        float value 
    """
    if torch.is_tensor(x):
        return float(x.detach().cpu().numpy())
    if isinstance(x, np.generic):
        return float(x)
    return x