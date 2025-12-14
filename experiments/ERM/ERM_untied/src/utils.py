import torch
import numpy as np
import os
from datetime import datetime

def init_torch(seed=42, verbose=True):
    """Initialise les graines aléatoires pour la reproductibilité."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if verbose:
        print(f"Random seeds initialized to {seed}")


def convert_numeric_config(config, verbose=True):
    """Convertit les valeurs de configuration de type chaîne en int ou float si possible."""
    new_config = config.copy()
    for k, v in new_config.items():
        if isinstance(v, str):
            try:
                new_config[k] = int(v)  # Essayer de convertir en int
            except ValueError:
                try:
                    new_config[k] = float(v)  # Essayer de convertir en float
                except ValueError:
                    pass  # Laisser tel quel si ce n’est ni int ni float
        elif isinstance(v, dict):
            new_config[k] = convert_numeric_config(v, verbose=False) # Traiter récursivement
            
    if verbose:
        print(f"[Config] Converted config: {new_config}")
        
    return new_config

def get_run_dir(base_path="/home/peucelle/tpiv-simulations/results"):
    """Crée un répertoire unique pour la nouvelle exécution."""
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{base_path}/run_{now_str}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir