import torch
import numpy as np

def clean_value(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().numpy())
    if isinstance(x, np.generic):
        return float(x)
    return x