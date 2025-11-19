# Exploring Transformers Empirically  
*Architecture, regularization, and their impact on expressivity, spectral properties, and learning across synthetic and real data*

---

This repository contains the code, experiments, and supplementary materials for the paper:

> **Exploring Transformers Empirically**  
> *Author(s): Peucelle Victor*  
> [Link to the paper (arXiv / conference / journal if available)]

<a href="https://victor-pcll.github.io/Exploring-transformers-empirically/">
  <img src="https://img.shields.io/badge/Docs-00aced?style=flat&logo=read-the-docs&logoColor=white" alt="Documentation"/>
</a>

---

## 🛠️ Technologies & Libraries

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white"/>
</p>

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
├── experiments/               
│   ├── BO/                  
│   │   ├── BO_exp_cluster.py
│   │   └── BO_exp_cluster.sbatch
│   ├── ERM_untied/                  
│   │   ├── ERM_untied_exp_cluster.py
│   │   └── ERM_untied_exp_cluster.sbatch
│   ├── ERM_S/                  
│   │   ├── ERM_S_exp_cluster.py
│   │   └── ERM_S_exp_cluster.sbatch
│   └── ERM_tied/                  
│       ├── ERM_tied_exp_cluster.py
│       └── ERM_tied_exp_cluster.sbatch      
│
├── results/                   
│   └── run_<JOBID>/     
│       ├── config.csv
│       ├── err.txt
│       ├── log.txt
│       ├── experiment.txt
│       ├── logs_<RUN>.txt
│       ├── summary.csv 
│       ├── W_runs_<RUN>.pkl 
│       └── config_used.pkl
│
├── notebooks/
│   ├── analysis.ipynb
│   └── sanity_check.ipynb
│
├── test/
│   ├── test_cluster.py
│   └── test_cluster.sbatch
│
├── adr/
│   ├── 001-adr-cluster.md
│   └── 002-adr-torch.md
│
├── requirements.txt
└── pyproject.toml 
```

---

## ADR (Architecture Decision Records)
This project maintains important architectural decisions in the `adr/` folder:

- [001-ADR Cluster](./adr/001-adr-cluster.md)
- [002-ADR Torch](./adr/002-adr-torch.md)
- [003-ADR Sphinx Documentation](./adr/003-adr-sphinx.md)

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
chmod +x .venv/bin/activate                          
. .venv/bin/activate  
python -m experiments.run_experiment
```

---

## 🛠️ Development

For editable installs during development, run:

```bash
pip install -e .
```

---

If you encounter any issues or have questions, please open an issue on the repository.