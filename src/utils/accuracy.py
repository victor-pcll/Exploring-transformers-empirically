def accuracy(pred, true):
    """
    Calcule la précision (accuracy) entre deux séquences de labels.
    
    Args:
        pred: liste, numpy array ou torch tensor des prédictions
        true: liste, numpy array ou torch tensor des labels réels
    
    Returns:
        float: fraction de prédictions correctes (entre 0 et 1)
    """
    # Conversion en listes si nécessaire
    if hasattr(pred, "tolist"):
        pred = pred.tolist()
    if hasattr(true, "tolist"):
        true = true.tolist()
    
    if len(pred) != len(true):
        raise ValueError(f"Les listes doivent avoir la même longueur "
                         f"(pred={len(pred)}, true={len(true)})")
    
    if len(true) == 0:
        return 0.0  
    
    correct = sum(p == t for p, t in zip(pred, true))
    return correct / len(true)