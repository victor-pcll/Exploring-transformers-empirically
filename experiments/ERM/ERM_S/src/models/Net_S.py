import torch
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, input_dim, number_tokens, norm=1.0, beta=1.0):
        super(Net, self).__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.S = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.S.weight.data.uniform_(0.0, norm)      

    def init_teacher(self, R):
        """
        Initialize S as a PSD matrix: S = W W^T / sqrt(r * D)
        where W is a random Gaussian matrix.
        """
        self.R = R
        W = torch.randn(self.D, self.R, device=self.S.weight.device)
        S_psd = (W @ W.T) / np.sqrt(self.R * self.D)
        self.S.weight.data = S_psd.clone() 


    def forward(self, x, delta_in):
        x_S = self.S(x)
        attention_matrix = torch.einsum('nap,nbp->nab', x, x_S)
        x = attention_matrix
        if delta_in > 0.0:
            M = torch.full((self.L, self.L), 1.0 / np.sqrt(2), device=x.device, dtype=x.dtype)
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device, dtype=x.dtype)
            i, j = torch.triu_indices(row=self.L, col=self.L, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]
            x = x + np.sqrt(delta_in) * eps * M
        x = torch.nn.Softmax(dim=-1)(self.beta * x)
        return x