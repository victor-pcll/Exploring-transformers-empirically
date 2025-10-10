import torch
import numpy as np

class Net(torch.nn.Module):
    """Network used for teacher/student setup with explicit device/dtype handling."""

    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        number_tokens, 
        device='cpu', 
        dtype=torch.float32, 
        norm=1.0, 
        beta=1.0
    ):
        super().__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.R = hidden_dim
        self.device = torch.device(device)
        self.dtype = dtype

        # Linear layer with explicit device + dtype
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=False, device=self.device, dtype=self.dtype)
        self.fc1.weight.data.normal_(0.0, norm)  # safer explicit mean and std

    def forward(self, x, delta_in=0.0):
        # Ensure x matches the module’s device and dtype
        x = x.to(device=self.device, dtype=self.dtype)

        # Linear projection and normalization
        x = self.fc1(x) / np.sqrt(self.D)

        # Attention matrix
        attention_matrix = torch.einsum('nap,nbp->nab', x, x) / np.sqrt(self.R)

        # Trace correction term
        trace_part = torch.norm(self.fc1.weight)**2 / np.sqrt(self.R * self.D**2)
        x = attention_matrix - trace_part * torch.eye(self.L, device=self.device, dtype=self.dtype)

        # Inject input noise if delta_in > 0
        if delta_in > 0.0:
            M = torch.full((self.L, self.L), 1.0 / np.sqrt(2), device=self.device, dtype=self.dtype)
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=self.device, dtype=self.dtype)
            i, j = torch.triu_indices(row=self.L, col=self.L, offset=1, device=self.device)
            eps[..., j, i] = eps[..., i, j]
            x = x + np.sqrt(delta_in) * eps * M

        # Softmax over last dimension
        return torch.nn.functional.softmax(self.beta * x, dim=-1)