import torch
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, MLP_dim, number_tokens, T, norm=1.0, beta=1.0, device="cpu"):
        super(Net, self).__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.T = T
        self.R = hidden_dim
        self.device = device
        self.embed = torch.nn.Embedding(number_tokens, input_dim)
        self.relu = torch.nn.ReLU()
        self.LayerNorm = torch.nn.LayerNorm(self.T) 
        self.W1 = torch.nn.Linear(T, MLP_dim, bias=True)
        self.W2 = torch.nn.Linear(MLP_dim, T, bias=False)
        self.W0 = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.W0.weight.data.normal_(0, norm)
        W = torch.randn(input_dim, hidden_dim, device=self.S.weight.device)
        S_psd = (W @ W.T) / np.sqrt(self.R * self.D) # Shape: (D, D)
        self.S.weight.data = S_psd.clone() #
        self.to(device)

    def forward(self, x, delta_in=0.0):
        x = x.to(self.device)
        x = self.embed(x)     # x.shape = (N, T, input_dim)
        x = self.W0(x) / np.sqrt(self.D)  # x.shape = (N, T, hidden_dim)
        # x = self.LayerNorm(x) # x.shape = (N, T, hidden_dim)
        attention_matrix = torch.einsum('nap,nbp->nab', x, x) / np.sqrt(self.R)  # attention_matrix.shape = (N, T, T)
        trace_part = torch.norm(self.W0.weight)**2 / np.sqrt(self.R * self.D**2)
        x = attention_matrix - trace_part * torch.eye(self.T, device=x.device) # x.shape = (N, T, T)
        if delta_in > 0.0:
            M = torch.full((self.T, self.T), 1.0/torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype)), device=x.device, dtype=x.dtype)
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device, dtype=x.dtype)
            i, j = torch.triu_indices(row=self.T, col=self.T, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]
            x = x + torch.sqrt(torch.tensor(delta_in, device=x.device, dtype=x.dtype)) * eps * M
        x = torch.nn.Softmax(dim=-1)(self.beta * x) # x.shape = (N, T, T)
        x = self.W1(x) # x.shape = (N, T, MLP_dim)
        x = self.relu(x) # x.shape = (N, T, MLP_dim)
        x = self.W2(x) # x.shape = (N, T, T)
        y = x.sum(dim=-1) # y.shape = (N, T)
        return x, y