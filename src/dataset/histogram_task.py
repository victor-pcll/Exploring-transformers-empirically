import torch
import numpy as np
from collections import Counter

def hist(s):
    c = Counter(s)
    return [c[w] for w in s]

class HistogramDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.seq_len = config["seq_len"]
        self.L = config["L"]
        self.n_samples = config["N_total"]
        rs = np.random.RandomState(config["seed"])
        self.X = rs.randint(0, self.L, (self.n_samples, self.seq_len))
        self.y = np.empty_like(self.X)
        for i in range(self.n_samples):
            self.y[i] = hist(self.X[i])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)