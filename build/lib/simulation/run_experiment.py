import torch
from src.models.neural_net import NeuralNet
from src.training.optimizer import train

def run_experiment(config):
    d, r, L, beta = config["d"], config["r"], config["L"], config["beta"]
    model = NeuralNet(d, r, L, beta)

    X = torch.randn(config["T"], d)
    target = torch.randn(config["T"], r)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = torch.nn.MSELoss()

    losses = train(model, X, target, loss_fn, optimizer, n_iter=config["n_iter"])
    return losses