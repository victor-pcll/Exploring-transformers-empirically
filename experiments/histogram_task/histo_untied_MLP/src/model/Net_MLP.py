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
        self.W_Q = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_K = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_Q.weight.data.normal_(0, norm)
        self.W_K.weight.data.normal_(0, norm)
        self.to(device)

    def forward(self, x, delta_in=0.0):
        x = x.to(self.device) # x.shape = (N, T)
        x = self.embed(x)     # x.shape = (N, T, input_dim)
        Q = self.W_Q(x) / np.sqrt(self.D) # Q.shape = (N, T, R)
        K = self.W_K(x) / np.sqrt(self.D) # K.shape = (N, T, R)
        A = torch.einsum('nap,nbp->nab', Q, K) / np.sqrt(self.R) # A.shape = (N, T, T)
        attn = torch.nn.Softmax(dim=-1)(self.beta * A)
        self.attn = attn
        x = torch.matmul(attn, x) # x.shape = (N, T, input_dim)
        x = self.W1(x) # x.shape = (N, T, MLP_dim)
        x = self.relu(x) # x.shape = (N, T, MLP_dim)
        x = self.W2(x) # x.shape = (N, T, n_classes)
        return x