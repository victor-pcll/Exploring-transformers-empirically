import torch
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, MLP_dim, L, T, norm=1.0, beta=1.0, device="cpu"):
        super(Net, self).__init__()
        self.beta = beta
        self.D = input_dim
        self.L = L
        self.T = T
        self.n_classes = T + 1
        self.R = hidden_dim
        self.device = device
        self.embed = torch.nn.Embedding(T, input_dim)
        self.relu = torch.nn.ReLU()
        self.W1 = torch.nn.Linear(input_dim, MLP_dim, bias=True)
        self.W2 = torch.nn.Linear(MLP_dim, T + 1, bias=False)
        self.S = torch.nn.Parameter(torch.empty(self.D, self.D))
        torch.nn.init.uniform_(self.S, -norm, norm)
        self.to(device)

    def forward(self, x, delta_in=0.0):
        x = x.to(self.device)  # x.shape = (N, T)
        x = self.embed(x)     # x.shape = (N, T, input_dim)
        A = torch.einsum("nap,pq,nbq->nab", x, self.S, x) / (self.D ** 1.5) # attention_matrix.shape = (N, T, T)
        trace_part = torch.norm(self.S)**2 / (self.D ** 1.5)
        # A = A - trace_part * torch.eye(self.T, device=x.device) # x.shape = (N, T, T)
        if delta_in > 0.0:
            M = torch.full((self.T, self.T), 1.0/torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype)), device=x.device, dtype=x.dtype)
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device, dtype=x.dtype)
            i, j = torch.triu_indices(row=self.T, col=self.T, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]
            A = A + torch.sqrt(torch.tensor(delta_in, device=x.device, dtype=x.dtype)) * eps * M
        x = torch.matmul(torch.nn.Softmax(dim=-1)(self.beta * A), x) # x.shape = (N, T, input_dim)
        x = self.W1(x) # x.shape = (N, T, MLP_dim)
        x = self.relu(x) # x.shape = (N, T, MLP_dim)
        x = self.W2(x) # x.shape = (N, T, n_classes)
        return x