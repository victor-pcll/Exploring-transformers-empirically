import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_tokens, norm=1.0, beta=1.0, device="cpu"):
        super(Net, self).__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.R = hidden_dim
        self.device = device
        
        # Architecture à une seule matrice (symétrique implicite dans l'attention)
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False).to(device)
        self.fc1.weight.data.normal_(0, norm)

    def forward(self, x, delta_in=0.0):
        x = x.to(self.device)
        sqrt_D = torch.sqrt(torch.tensor(self.D, device=x.device, dtype=x.dtype))
        sqrt_R = torch.sqrt(torch.tensor(self.R, device=x.device, dtype=x.dtype))
        
        # Projection
        x = self.fc1(x) / sqrt_D
        
        # Attention: x @ x.T (produit scalaire de x par lui-même)
        attention_matrix = torch.einsum('nap,nbp->nab', x, x) / sqrt_R
        
        # Correction de trace
        trace_part = torch.norm(self.fc1.weight)**2 / (sqrt_R * sqrt_D**2)
        x = attention_matrix - trace_part * torch.eye(self.L, device=x.device)
        
        # Ajout de bruit structurel
        if delta_in > 0.0:
            M = torch.full((self.L, self.L), 1.0/torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype)), device=x.device, dtype=x.dtype)
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device, dtype=x.dtype)
            i, j = torch.triu_indices(row=self.L, col=self.L, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]
            x = x + torch.sqrt(torch.tensor(delta_in, device=x.device, dtype=x.dtype)) * eps * M
            
        x = nn.Softmax(dim=-1)(self.beta * x)
        return x