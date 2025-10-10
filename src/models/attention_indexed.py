import torch
import numpy as np


def AIM(X, S, verbose=False):
    """
    Quadratic pre-activation: X S X^T - Tr(S) * I, normalized by sqrt(d)
    
    Args:
        X (torch.Tensor): T x d tensor
        S (torch.Tensor): d x d symmetric matrix
        verbose (bool): If True, print debug information.
        
    Returns:
        torch.Tensor: T x T tensor
    """
    T, d = X.shape

    if verbose:
        print(f"h input shapes : X {X.shape}, S {S.shape}")
    
    trace_part = torch.trace(S)
    I = torch.eye(T, dtype=X.dtype, device=X.device)
    
    return (X @ S @ X.T - trace_part * I) / np.sqrt(d)


def AIM_batch(X, S, verbose=False):
    """
    Batched quadratic pre-activation: X S X^T - Tr(S) * I, normalized by sqrt(d)
    
    Args:
        X (torch.Tensor): N x T x d tensor
        S (torch.Tensor): d x d symmetric matrix
        verbose (bool): If True, print debug information.
        
    Returns:
        torch.Tensor: N x T x T tensor
    """
    N, T, d = X.shape
    if verbose:
        print(f"h_batch input shapes : X {X.shape}, S {S.shape}")

    # Compute X S X^T for each batch
    XS = torch.matmul(X, S)                # N x T x d
    XSXt = torch.matmul(XS, X.transpose(-2, -1))  # N x T x T

    # Subtract trace part
    trace_part = torch.trace(S)
    I = torch.eye(T, dtype=X.dtype, device=X.device).unsqueeze(0)  # 1 x T x T
    XSXt_centered = XSXt - trace_part * I

    return XSXt_centered / np.sqrt(d)