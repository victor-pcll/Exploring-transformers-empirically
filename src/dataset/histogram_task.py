import torch
import numpy as np
import random
import string

def generate_random_pos(T, L):
    """
    Génère uniformément une composition aléatoire de T
    dont la longueur (nombre de parts) est au plus L.
    """
    if T <= 0:
        raise ValueError("T doit être un entier positif.")
    if L <= 0:
        raise ValueError("L doit être un entier positif.")

    # 1) Choisir uniformément une longueur k dans [1, min(T, L)]
    k = random.randint(1, min(T, L))

    # 2) Choisir k−1 positions de coupures dans les T−1 positions possibles
    cuts = sorted(random.sample(range(1, T), k - 1))

    # 3) Construire les segments entre les coupures
    parts = []
    prev = 0
    for c in cuts:
        parts.append(c - prev)
        prev = c

    # Dernier segment
    parts.append(T - prev)

    return parts


def generate_histogram_task(T, L):
    """
    Génère une séquence d'entiers (taille T) à partir d'une composition aléatoire,
    et un vecteur de comptage correspondant au nombre d'occurrences de chaque entier.

    Renvoie :
        - seq : liste mélangée d'entiers (taille T)
        - count_vector : liste (taille T) avec le nombre d'occurrences
                         de l'entier présent à chaque position
    """
    counts = generate_random_pos(T, L)
    
    if len(counts) > T:
        raise ValueError("Nombre de valeurs uniques trop élevé par rapport à T")
    
    seq = []
    value_count_map = {}
    
    for i, k in enumerate(counts):
        value = i 
        value_count_map[value] = k
        seq.extend([value] * k)
    
    random.shuffle(seq)
    count_vector = [value_count_map[val] for val in seq]
    
    return seq, count_vector

class HistogramDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.T = config["T"]
        self.L = config["L"]
        self.n_samples = config["N_total"]
        rs = np.random.RandomState(config["seed"])
        
        self.X = []
        self.y = []
        for _ in range(self.n_samples):
            x_sample, y_sample = generate_histogram_task(self.T, self.L)
            self.X.append(x_sample)
            self.y.append(y_sample)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)