import torch
from src.dataset.histogram_task import HistogramDataset

def prepare_dataset(config):
    """
    Prépare le dataset pour le histogram task, et effectue le split train/valid/test.
    
    Args:
        config: objet de configuration contenant au moins :
            - T: longueur des séquences
            - L: taille de l'alphabet
            - N_train: nombre d'échantillons d'entraînement
            - N_valid: nombre d'échantillons de validation
            - N_test: nombre d'échantillons de test
            - seed: int, pour reproductibilité
    
    Returns:
        train_dataset: torch Dataset pour l'entraînement
        valid_dataset: torch Dataset pour la validation
        test_dataset: torch Dataset pour le test
        full_dataset: torch Dataset complet
    """
    full_dataset = HistogramDataset(config)
    N = len(full_dataset)

    # proportions
    p_fine_tune = config["p_fine_tune"]
    p_test = config["p_test"]
    p_train = 1 - p_fine_tune - p_test

    # conversion en nombres d'échantillons
    N_train = int(N * p_train)
    N_ft = int(N * p_fine_tune)
    N_test  = N - N_train - N_ft  

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [N_train, N_ft, N_test],
        generator=torch.Generator().manual_seed(config["seed"])
    )

    return train_dataset, valid_dataset, test_dataset, full_dataset
