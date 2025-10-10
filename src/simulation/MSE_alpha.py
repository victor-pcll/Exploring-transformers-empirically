import numpy as np
import torch
from models.neural_net import Net
from models.attention_indexed import AIM_batch
from utils.init_S import init_S
from tqdm import tqdm


def MSE_alpha(
    config,
    device="cpu",
    dtype=torch.float32,
    verbose=False,
):
    """
    Evaluate training and generalization MSE of the student model
    for different values of alpha = n / d^2.
    """

    # Unpack config
    d, L = config["d"], config["L"]
    beta, rho = config["beta"], config["rho"]
    samples = config["samples"]
    Delta_in = config["Delta_in"]
    lr = config["lr"]
    n_iter = config["n_iter"]
    lam = config.get("lam", 0.0)  # optional regularization coefficient
    n_test = config.get("n_test", max(1, int(0.5 * d**2)))  # default test size
    alphas = eval(config["alpha"])
    verbose = config["verbose"]

    r = int(rho * d)  # hidden dimension

    MSE_train, MSE_test, losses, losses_err = [], [], [], []

    # ----------------------
    # Teacher initialization
    # ----------------------
    with torch.no_grad():
        teacher = Net(d, r, L, norm=1.0, beta=beta, device=device, dtype=dtype).to(
            device=device, dtype=dtype
        )
        W_teacher = teacher.fc1.weight.detach().cpu().numpy()
        S_teacher = W_teacher.T @ W_teacher / np.sqrt(r * d)
        S_teacher = torch.tensor(S_teacher, dtype=dtype, device=device)

    # ----------------------
    # Loop over α values
    # ----------------------
    alpha_iter = alphas if verbose else tqdm(alphas)
    for alpha in alpha_iter:
        n = max(1, int(alpha * d**2))

        mean_loss, mse_train, mse_test = [], [], []

        if verbose:
            print(f"\nAlpha = {alpha:.3f}, n = {n}")

        # ----------------------
        # Averaging over samples
        # ----------------------
        for mu in range(samples):
            # Generate synthetic data
            X_teacher = torch.normal(0, 1, (n, L, d), device=device, dtype=dtype)

            with torch.no_grad():
                y_teacher = teacher(X_teacher, delta_in=Delta_in)

            # Initialize trainable matrix S
            S_train = init_S(d, gradient=True, device=device, dtype=dtype)
            S_train.requires_grad_(True)
            opt = torch.optim.Adam([S_train], lr=lr)

            if verbose:
                print(f"  Sample {mu + 1}/{samples}")

            # ----------------------
            # Training loop
            # ----------------------
            for step in range(n_iter):
                y_pred = AIM_batch(X_teacher, S_train)

                # MSE + optional L1 regularization on Tr(S)
                loss = ((y_pred - y_teacher) ** 2).sum()
                if lam > 0:
                    loss += np.sqrt(d * r) * lam * torch.trace(S_train).abs()

                opt.zero_grad()
                loss.backward()
                opt.step()

            # ----------------------
            # Evaluation
            # ----------------------
            with torch.no_grad():
                # Train error (matrix distance)
                mse_S_train = ((S_train - S_teacher) ** 2).mean().item()

                # Test generalization
                X_test = torch.normal(0, 1, (n_test, L, d), device=device, dtype=dtype)
                y_test_teacher = AIM_batch(X_test, S_teacher)
                y_test_student = AIM_batch(X_test, S_train)
                mse_S_test = ((y_test_student - y_test_teacher) ** 2).mean().item()

            mean_loss.append(loss.item())
            mse_train.append(mse_S_train)
            mse_test.append(mse_S_test)

        # ----------------------
        # Aggregate statistics
        # ----------------------
        mean = np.mean(mean_loss)
        err = np.std(mean_loss) / np.sqrt(len(mean_loss))

        if verbose:
            print(
                f"Loss: {mean:.6f} ± {err:.6f} | "
                f"MSE_train: {np.mean(mse_train):.6f} | "
                f"MSE_test: {np.mean(mse_test):.6f}"
            )

        losses.append(mean)
        losses_err.append(err)
        MSE_train.append(np.mean(mse_train))
        MSE_test.append(np.mean(mse_test))

    return alphas, losses, losses_err, MSE_train, MSE_test
