import torch
import numpy as np
import random
import string

def generate_random_pos(T):
    """
    Génère une composition aléatoire de T:
    une liste d'entiers positifs, longueur variable,
    dont la somme vaut exactement T.
    """
    if T <= 0:
        raise ValueError("T doit être un entier positif.")

    parts = []
    current = 1 

    for _ in range(T - 1):
        if random.random() < 0.5:
            current += 1
        else:
            parts.append(current)
            current = 1

    parts.append(current)
    return parts

def generate_histogram_task(T):
    """
    À partir d'une liste d'entiers positifs (composition),
    renvoie :
        - seq : liste mélangée de lettres (taille T)
        - count_vector : liste (taille T) avec le nombre d'occurrences
                         de la lettre présente à chaque position.
    """
    alphabet = string.ascii_uppercase
    counts = generate_random_pos(T)
    
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
        self.y, self.X = generate_histogram_task(self.T)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)