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
        torch.nn.init.normal_(self.S, mean=0.0, std=1.0 / np.sqrt(self.D))
        self.to(device)

    def forward(self, x, delta_in=0.0):
        x = x.to(self.device)
        x = self.embed(x)
        upper = torch.triu(self.S) 
        lower = torch.triu(self.S, diagonal=1).transpose(0, 1)
        S_sym = upper + lower
        A = torch.einsum("nap,pq,nbq->nab", x, S_sym, x) / (self.D ** 0.5)
        attn = torch.nn.Softmax(dim=-1)(self.beta * A)
        self.attn = attn
        x = torch.matmul(attn, x)
        x = self.W1(x)
        x = self.relu(x)
        x = self.W2(x)
        return x