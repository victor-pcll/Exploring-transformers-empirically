def plot_loss(losses):
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()