import torch
import numpy as np
from utils.sym_mat import sym_torch


def init_S(
    d, r=0, type=1, gradient=False, verbosity=0, device="cpu", dtype=torch.float32
):
    """
    Initialize covariance / weight matrix S.

    Args:
        d (int): input dimension
        r (int): hidden dimension (used if type=2)
        type (int): 1 = random symmetric S, 2 = random W -> S = W W^T / sqrt(r*d)
        gradient (bool): if True, S.requires_grad = True
        verbosity (int): debug prints
        device (str): 'cpu' or 'cuda'
        dtype (torch.dtype): tensor dtype

    Returns:
        S (torch.Tensor): d x d symmetric matrix
    """
    if type == 1:
        if verbosity >= 3:
            print("Initialization by symmetric S matrix")
        n = d * (d + 1) // 2
        A = torch.randn(n, device=device, dtype=dtype)
        S = sym_torch(A)

    elif type == 2:
        if verbosity >= 3:
            print("Initialization by W matrix: S = W W^T / sqrt(r*d)")
        if r <= 0:
            raise ValueError("r must be > 0 for type=2")
        W = torch.randn(d, r, device=device, dtype=dtype)
        S = W @ W.T / np.sqrt(r * d)

    else:
        raise ValueError("model type must be 1 or 2")

    if gradient:
        S.requires_grad_(True)

    if verbosity >= 2:
        print(f"S (type={type}) shape: {S.shape}, device={S.device}, dtype={S.dtype}")

    return S
