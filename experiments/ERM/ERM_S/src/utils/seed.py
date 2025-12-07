import torch
import numpy as np

def init_torch(seed=42, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if verbose:
        print(f"Random seeds initialized to {seed}")