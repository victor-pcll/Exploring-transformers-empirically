import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    """Simple feedforward network used in TP IV."""
    def __init__(self, d: int, r: int, L: int, beta: float):
        super().__init__()
        self.fc1 = nn.Linear(d, r)
        self.activation = nn.Tanh()
        self.beta = beta
        self.L = L

    def forward(self, X):
        out = self.fc1(X)
        for _ in range(self.L):
            out = self.activation(out * self.beta)
        return out