import torch
import numpy as np


def init_torch(seed=42, verbose=False):
    """
    Initialise l'environnement PyTorch :
    - Fixe la graine aléatoire (torch et numpy) pour la reproductibilité
    - Détecte automatiquement le device disponible (MPS, CUDA ou CPU)
    - Configure le backend pour des performances stables

    Parameters
    ----------
    seed : int, optional
        Graine utilisée pour la reproductibilité (par défaut 42)

    Returns
    -------
    torch.device
        Le device sélectionné ('cuda', 'mps' ou 'cpu')
    """
    # --- Reproductibilité ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Détection du device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # GPU Apple
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # GPU NVIDIA
    else:
        device = torch.device("cpu")  # Fallback CPU

    # --- Options PyTorch ---
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if verbose:
        print(f"[Init] Using device: {device}")
        print(f"[Init] Random seed set to: {seed}")

    return device
