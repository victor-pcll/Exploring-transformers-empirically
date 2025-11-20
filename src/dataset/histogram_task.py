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
    À partir d'une liste d'entiers positifs (composition),
    renvoie :
        - seq : liste mélangée de lettres (taille T)
        - count_vector : liste (taille T) avec le nombre d'occurrences
                         de la lettre présente à chaque position.
    """
    alphabet = string.ascii_uppercase
    counts = generate_random_pos(T, L)
    
    if len(counts) > len(alphabet):
        raise ValueError("Alphabet insuffisant.")
    
    seq = []
    letter_count_map = {} 
    
    for i, k in enumerate(counts):
        letter = alphabet[i]
        letter_count_map[letter] = k
        seq.extend([letter] * k)
    
    random.shuffle(seq)
    count_vector = [letter_count_map[letter] for letter in seq]
    
    return seq, count_vector

class HistogramDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.T = config["T"]
        self.L = config["L"]
        self.n_samples = config["N_total"]
        rs = np.random.RandomState(config["seed"])
        self.X, self.y = generate_histogram_task(self.T, self.L)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)