import torch
from src.dataset.histogram_task import HistogramDataset

def prepare_dataset(config):
    """
    Prépare le dataset pour le histogram task, et effectue le split train/test.
    
    Args:
        config: objet de configuration contenant au moins :
            - T: longueur des séquences
            - L: taille de l'alphabet
            - N_train: nombre d'échantillons d'entraînement
            - N_test: nombre d'échantillons de test
        seed: int, pour reproductibilité
    
    Returns:
        train_dataset: torch Dataset pour l'entraînement
        test_dataset: torch Dataset pour le test
        full_dataset: torch Dataset complet
    """
    full_dataset = HistogramDataset(config)
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [config["N_train"], 
        config["N_test"]],
        generator=torch.Generator().manual_seed(config["seed"])
        )
    return train_dataset, test_dataset
