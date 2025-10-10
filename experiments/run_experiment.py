from simulation.MSE_alpha import MSE_alpha
from utils.plotting import plot_mse_results
from utils.io import save_results
from utils.seeds import init_torch
from utils.convert import convert_numeric_config
import yaml
import os
from datetime import datetime

# Désactiver MPS et CUDA
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Forcer CPU
os.environ["PYTORCH_DEVICE"] = "cpu"
import torch

# Forcer CPU partout
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_default_tensor_type(torch.FloatTensor)

# Charger la configuration
with open("src/config/default.yaml") as f:
    config = yaml.safe_load(f)

# Initialiser les graines
device = init_torch(42, verbose=config["verbose"])
dtype = torch.float32

# Convertir les paramètres numériques
config = convert_numeric_config(config, verbose=config["verbose"])

# Lancer l'expérience (CPU)
alphas, losses, losses_err, MSE_train, MSE_test = MSE_alpha(config, device, dtype)

# Create a run-specific folder with timestamp
run_folder = datetime.now().strftime("results/run_%Y%m%d_%H%M%S")
os.makedirs(run_folder, exist_ok=True)

# Visualiser les pertes
fig_path = os.path.join(run_folder, "MSE_alpha_plot.png")
plot_mse_results(
    alphas,
    losses,
    losses_err,
    MSE_train,
    MSE_test,
    log_x=False,
    log_y=False,
    save_path=fig_path,
    show=True,
)
print(f"Figure saved to: {fig_path}")

# Prepare results dictionary
results = {
    "alphas": alphas,
    "losses": losses,
    "losses_err": losses_err,
    "MSE_train": MSE_train,
    "MSE_test": MSE_test,
    "config": config,  # optional: save parameters too
}

# Save the results
log_path = os.path.join(run_folder, "MSE_alpha_results.pkl")
save_results(results, log_path, verbose=True, timestamp=False, format="pickle")
print(f"Results saved to: {log_path}")
