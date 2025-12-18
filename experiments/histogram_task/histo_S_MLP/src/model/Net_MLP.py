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
        x = x.to(self.device)  # x.shape = (N, T)
        x = self.embed(x)     # x.shape = (N, T, input_dim)
        A = torch.einsum("nap,pq,nbq->nab", x, self.S, x) / (self.D ** 0.5) # attention_matrix.shape = (N, T, T)
        attn = torch.nn.Softmax(dim=-1)(self.beta * A)
        self.attn = attn
        x = torch.matmul(attn, x) 
        x = self.W1(x) # x.shape = (N, T, MLP_dim)
        x = self.relu(x) # x.shape = (N, T, MLP_dim)
        x = self.W2(x) # x.shape = (N, T, n_classes)
        return x