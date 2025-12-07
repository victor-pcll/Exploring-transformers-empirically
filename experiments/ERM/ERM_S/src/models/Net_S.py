import torch
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, input_dim, number_tokens, norm=1.0, beta=1.0):
        super(Net, self).__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.S = torch.nn.Parameter(torch.empty(self.D, self.D))
        torch.nn.init.uniform_(self.S, -norm, norm)

    def init_teacher(self, R):
        self.R = R
        W = torch.randn(self.D, self.R, device=self.S.device)
        S_psd = (W @ W.T) / np.sqrt(self.D * self.R)
        device = self.S.device
        self.S.data = S_psd.clone()


    def forward(self, x, delta_in):
        x = torch.einsum("nap,pq,nbq->nab", x, self.S, x) / (self.D ** 1.5)
        trace_part = torch.norm(self.S)**2 / (self.D ** 1.5)
        x = x # - trace_part * torch.eye(self.L, device=x.device)
        if delta_in > 0.0:
            M = torch.full((self.L, self.L), 1.0 / np.sqrt(2), device=x.device, dtype=x.dtype)
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device, dtype=x.dtype)
            i, j = torch.triu_indices(row=self.L, col=self.L, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]
            x = x + np.sqrt(delta_in) * eps * M
        x = torch.nn.Softmax(dim=-1)(self.beta * x)
        return x