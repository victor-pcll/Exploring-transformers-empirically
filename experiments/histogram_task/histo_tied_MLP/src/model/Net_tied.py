import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_tokens, seq_len, norm=1.0, beta=1.0, device="cpu"):
        super().__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.seq_len = seq_len
        self.R = hidden_dim
        self.device = device
        self.embed = nn.Embedding(number_tokens, input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc1.weight.data.normal_(0, norm)
        self.to(device)

    def forward(self, x, delta_in=0.0):
        x = x.to(self.device)
        x = self.embed(x)
        x = self.fc1(x) / np.sqrt(self.D)
        attention_matrix = torch.einsum('nap,nbp->nab', x, x) / np.sqrt(self.R)
        trace_part = torch.norm(self.fc1.weight)**2 / np.sqrt(self.R * self.D**2)
        x = attention_matrix - trace_part * torch.eye(self.seq_len, device=x.device)
        if delta_in > 0.0:
            # ajouter bruit symétrique
            M = torch.full((self.seq_len, self.seq_len), 1.0/torch.sqrt(torch.tensor(2.0, device=x.device)))
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device)
            i, j = torch.triu_indices(row=self.L, col=self.L, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]
            x = x + torch.sqrt(torch.tensor(delta_in, device=x.device)) * eps * M
        x = nn.Softmax(dim=-1)(self.beta * x)
        h_pred = x.sum(dim=-1)
        return x, h_pred