import torch
from src.models.neural_net import NeuralNet
from src.training.optimizer import train

def Adam_opt(config, verbose=False):
    """
    Train a neural network using the Adam optimizer.

    Args:
    ---
    config (dict): Configuration dictionary containing parameters.
    verbose (bool): If True, print additional information.

    Returns:
    ---
    losses (list): List of loss values during training.
    """
    d, r, L, beta = config["d"], config["r"], config["L"], config["beta"]
    model = NeuralNet(d, r, L, beta)

    X = torch.randn(config["T"], d)
    target = torch.randn(config["T"], r)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = torch.nn.MSELoss()

    losses = train(model, X, target, loss_fn, optimizer, n_iter=config["n_iter"])
    return losses