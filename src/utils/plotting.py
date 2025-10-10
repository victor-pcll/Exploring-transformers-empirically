import matplotlib.pyplot as plt


def plot_loss(
    losses,
    title="Loss over Iterations",
    xlabel="Iteration",
    ylabel="Loss",
    log_scale=False,
    save_path=None,
    show=True,
):
    """
    Plot a loss curve with customization options.

    Parameters
    ----------
    losses : list or array-like
        Sequence of loss values over training iterations.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    log_scale : bool, optional
        If True, use a logarithmic scale for the y-axis.
    save_path : str, optional
        If provided, saves the plot to the given path.
    show : bool, optional
        Whether to display the plot.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(losses, color="tab:blue", linewidth=2, label="Training Loss")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=10)
    if log_scale:
        plt.yscale("log")
    plt.grid(alpha=0.3, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_mse_results(
    alphas,
    est_gen,
    est_err,
    MSE_train,
    MSE_test,
    xlabel=r"$\alpha$",
    log_x=False,
    log_y=False,
    save_path=None,
    show=True,
):
    """
    Plot generalization error, training MSE, and test MSE in three subplots sharing the x-axis.

    Parameters
    ----------
    alphas : list or array-like
        Values for the alpha parameter on the x-axis.
    est_gen : list or array-like
        Estimated generalization error values.
    est_err : list or array-like
        Error bars for the generalization error.
    MSE_train : list or array-like
        Training mean squared error values.
    MSE_test : list or array-like
        Test mean squared error values.
    xlabel : str, optional
        Label for the x-axis.
    log_x : bool, optional
        If True, use logarithmic scale for the x-axis.
    log_y : bool, optional
        If True, use logarithmic scale for the y-axis on all subplots.
    save_path : str, optional
        If provided, saves the plot to the given path.
    show : bool, optional
        Whether to display the plot.
    """
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

    # Generalization error with error bars
    axs[0].errorbar(
        alphas,
        est_gen,
        yerr=est_err,
        fmt="o-",
        color="tab:blue",
        label="Generalization Error",
    )
    axs[0].set_ylabel("Generalization Error", fontsize=12)
    axs[0].grid(alpha=0.3, which="both", linestyle="--")
    axs[0].legend()
    if log_y:
        axs[0].set_yscale("log")
    if log_x:
        axs[0].set_xscale("log")

    # Training MSE
    axs[1].plot(alphas, MSE_train, "s-", color="tab:green", label="Training MSE")
    axs[1].set_ylabel("Training MSE", fontsize=12)
    axs[1].grid(alpha=0.3, which="both", linestyle="--")
    axs[1].legend()
    if log_y:
        axs[1].set_yscale("log")
    if log_x:
        axs[1].set_xscale("log")

    # Test MSE
    axs[2].plot(alphas, MSE_test, "d-", color="tab:red", label="Test MSE")
    axs[2].set_xlabel(xlabel, fontsize=12)
    axs[2].set_ylabel("Test MSE", fontsize=12)
    axs[2].grid(alpha=0.3, which="both", linestyle="--")
    axs[2].legend()
    if log_y:
        axs[2].set_yscale("log")
    if log_x:
        axs[2].set_xscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
