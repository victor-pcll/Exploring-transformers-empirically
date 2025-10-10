import torch

def sym_torch(A):
    """
    Reconstruct a symmetric matrix from a 1D vector A containing the upper-triangle entries (including diagonal).
    
    Parameters
    ----------
        A (torch.Tensor): 1D tensor of length n = T*(T+1)/2
    
    Returns
    -------
        R (torch.Tensor): symmetric T x T matrix
    """
    n = len(A)
    T = int((torch.sqrt(8*torch.tensor(n, dtype=torch.float)) + 1) // 2)
    assert T*(T+1)//2 == n, "len(A) is not triangular"

    R = torch.zeros((T, T), dtype=A.dtype, device=A.device)
    iu = torch.triu_indices(T, T)
    R[iu[0], iu[1]] = A
    R[iu[1], iu[0]] = A  # enforce symmetry
    return R