import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, MLP_dim, L, T, norm=1.0, beta=1.0, device="cpu"):
        super().__init__()
        self.beta = beta
        self.D = input_dim
        self.L = L
        self.T = T
        self.n_classes = T + 1
        self.R = hidden_dim
        self.device = device

        # --- Embedding ---
        self.embed = nn.Embedding(T, input_dim)

        # --- Untied Q and K projections ---
        self.W_Q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(input_dim, hidden_dim, bias=False)

        # Init
        self.W_Q.weight.data.normal_(0, norm)
        self.W_K.weight.data.normal_(0, norm)

        # --- MLP classifer ---
        self.relu = nn.ReLU()
        self.W1 = nn.Linear(input_dim, MLP_dim, bias=True)
        self.W2 = nn.Linear(MLP_dim, self.n_classes, bias=False)

        self.to(device)

    def forward(self, x, delta_in=0.0):
        x = x.to(self.device)                       # (N, T)
        x = self.embed(x)                           # (N, T, D)

        # --- Untied Q and K ---
        Q = self.W_Q(x) / np.sqrt(self.D)           # (N, T, R)
        K = self.W_K(x) / np.sqrt(self.D)           # (N, T, R)

        # Attention logits
        A = torch.einsum("nap,nbp->nab", Q, K) / np.sqrt(self.R)   # (N, T, T)

        # --- Noise injection ---
        if delta_in > 0.0:
            M = torch.full((self.T, self.T), 1.0 / np.sqrt(2),
                           device=A.device, dtype=A.dtype)
            M.diagonal().fill_(1)

            eps = torch.normal(0.0, 1.0, A.shape, device=A.device)
            i, j = torch.triu_indices(self.T, self.T, offset=1, device=A.device)
            eps[..., j, i] = eps[..., i, j]

            A = A + np.sqrt(delta_in) * eps * M

        # --- Softmax attention ---
        attn = nn.Softmax(dim=-1)(self.beta * A)
        self.attn = attn

        # --- Attention output ---
        x = torch.matmul(attn, x)                   # (N, T, D)

        # --- MLP classifier ---
        x = self.W1(x)
        x = self.relu(x)
        x = self.W2(x)

        return x