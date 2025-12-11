import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_tokens, norm=1.0, beta=1.0):
        super().__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.R = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc1.weight.data.normal_(0, norm)

    def forward(self, x, delta_in=0.0):
        x = self.fc1(x) / np.sqrt(self.D)
        attention_matrix = torch.einsum('nap,nbp->nab', x, x) / np.sqrt(self.R)
        trace_part = torch.norm(self.fc1.weight)**2 / np.sqrt(self.R * self.D**2)
        x = attention_matrix - trace_part * torch.eye(self.L, device=x.device)
        if delta_in > 0.0:
            M = torch.full((self.L, self.L), 1/np.sqrt(2), device=x.device, dtype=x.dtype)
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device, dtype=x.dtype)
            i, j = torch.triu_indices(self.L, self.L, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]
            x = x + np.sqrt(delta_in) * eps * M
        x = nn.Softmax(dim=-1)(self.beta * x)
        return x


def compute_S_from_W(W, R, D):
    return (W.T @ W) / np.sqrt(R * D)


def S_MSE(W_student, W_teacher, R, R_star, D):
    S_stud = compute_S_from_W(W_student, R, D)
    S_teach = compute_S_from_W(W_teacher, R_star, D)
    return float(((S_stud - S_teach) ** 2).sum() / D)