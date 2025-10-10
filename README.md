# Exploring Transformers Empirically  
*Architecture, regularization, and their impact on expressivity, spectral properties, and learning across synthetic and real data*

This repository contains the code, experiments, and supplementary materials for the paper:

> **Exploring Transformers Empirically**  
> *Author(s): Peucelle Victor*  
> [Link to the paper (arXiv / conference / journal if available)]

---

## 📖 Overview  
This repository provides modular Python code and simulations for the TP IV project, focusing on transformer architectures. It includes:

- Modular network definitions, training routines, and experiment scripts.  
- Configuration-driven reproducible experiments.  
- Utilities for plotting, saving results, and managing random seeds.

---

## 📂 Repository Structure

```
tpiv-simulations/
│
├── pyproject.toml              
├── README.md                   # Project overview and instructions
├── .gitignore
├── data/                       # Data and data generators
│   ├── synthetic/              # Simulated datasets
│   └── raw/                    # Raw data if applicable
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── models/                 # Network class definitions
│   │   ├── __init__.py
│   │   └── neural_net.py
│   ├── training/               # Optimization, loss functions, metrics
│   │   ├── __init__.py
│   │   └── optimizer.py
│   ├── simulation/             # Experiment scripts and loops
│   │   ├── __init__.py
│   │   └── run_experiment.py
│   ├── utils/                  # Utilities (plotting, IO, seeds)
│   │   ├── __init__.py
│   │   ├── plotting.py
│   │   ├── io.py
│   │   └── seeds.py
│   └── config/                 # Configuration files
│       ├── __init__.py
│       └── default.yaml
│
├── experiments/                # Notebooks and reproducible scripts
│   ├── exp_symmetric_init.ipynb
│   ├── exp_rho_variation.ipynb
│   └── exp_generalization.ipynb
│
└── results/                    # Outputs: figures, logs, checkpoints
    ├── figures/
    └── logs/
```

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/victor-pcll/Exploring-transformers-empirically.git
cd tpiv-simulations
pip install .
```

Run experiments using:

```bash
python -m src.simulation.run_experiment
```

---

## 🛠️ Development

For editable installs during development, run:

```bash
pip install -e .
```

---

If you encounter any issues or have questions, please open an issue on the repository.