import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    """
    Modèle de réseau de neurones implémentant un mécanisme d'attention simplifié
    pour l'expérience Teacher-Student.
    """
    def __init__(self, input_dim, hidden_dim, number_tokens, norm=1.0, beta=1.0):
        super(Net, self).__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.R = hidden_dim
        self.W_Q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_Q.weight.data.normal_(0, norm)
        self.W_K.weight.data.normal_(0, norm)

    def forward(self, x, delta_in):
        Q = self.W_Q(x) / np.sqrt(self.D)
        K = self.W_K(x) / np.sqrt(self.D)
        attention_matrix = torch.einsum('nap,nbp->nab', Q, K) / np.sqrt(self.R)
        trace_part = (
            torch.norm(self.W_Q.weight)**2 +
            torch.norm(self.W_K.weight)**2
        ) / np.sqrt(2 * self.R * self.D**2)
        
        x = attention_matrix 
        if delta_in > 0.0:
            M = torch.full((self.L, self.L), 1.0 / np.sqrt(2), device=x.device, dtype=x.dtype)
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device, dtype=x.dtype)
            i, j = torch.triu_indices(row=self.L, col=self.L, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]
            x = x + np.sqrt(delta_in) * eps * M
        x = nn.Softmax(dim=-1)(self.beta * x)
        return x