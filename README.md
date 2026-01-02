# Exploring Transformers Empirically  
*Architecture, regularization, and their impact on expressivity, spectral properties, and learning across synthetic and real data*

---

This repository contains the code, experiments, and supplementary materials for the paper:

> **Exploring Transformers Empirically**  
> *Author(s): Peucelle Victor*  

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
│   │   ├── main.py
│   │   ├── src/
│   │   └── job.sbatch
│   ├── ERM/  
│   │   ├── ERM_untied/                  
│   │   │   ├── main.py
│   │   │   ├── src/
│   │   │   └── job.sbatch
│   │   ├── ERM_S/                  
│   │   │   ├── main.py
│   │   │   ├── src/
│   │   │   └── job.sbatch
│   │   └── ERM_tied/                  
│   │       ├── main.py
│   │       ├── src/
│   │       └── job.sbatch 
│   └── histogram_task/
│       ├── histo_S_MLP/                  
│       │   ├── main.py
│       │   ├── src/
│       │   └── job.sbatch
│       ├── histo_untied_MLP/                  
│       │   ├── main.py
│       │   ├── src/
│       │   └── job.sbatch
│       ├── histo_S_sym/                  
│       │   ├── main.py
│       │   ├── src/
│       │   └── job.sbatch
│       └── histo_tied_MLP/                  
│           ├── main.py
│           ├── src/
│           └── job.sbatch    
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
│   ├── BO_visua.ipynb
│   ├── Synthetic_S_visua.ipynb
│   ├── Synthetic_tied_visua.ipynb
│   ├── Synthetic_untied_visua.ipynb
│   ├── Real_data_S_visua.ipynb
│   ├── Real_data_untied_visua.ipynb
│   └── Real_data_tied_visua.ipynb
│
├── test/
│   ├── test_cluster.py
│   ├── notebook.ipynb
│   └── test_cluster.sbatch
│
├── adr/
│   ├── 001-adr-cluster.md
│   ├── 002-adr-torch.md
│   ├── 003-adr-sphinx.md
│   └── 004-adr-slurm.md
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
- [004-ADR Slurm](./adr/004-adr-slurm.md)

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

## 🧪 Cluster Tests

To verify that the cluster environment is correctly set up for running experiments, use the provided test scripts located in the `test/` directory.

- `test_cluster.py`: A simple Python script to check environment variables, dependencies, and basic functionality on the cluster.
- `test_cluster.sbatch`: A Slurm batch script to submit `test_cluster.py` as a job on the cluster scheduler.

To run the test script directly on a cluster node, execute:

```bash
python test/test_cluster.py
```

To submit the test job to the cluster queue, use:

```bash
sbatch test/test_cluster.sbatch
```

These tests help ensure that the cluster environment is properly configured before launching large-scale experiments.

---

## 🛠️ Development

For editable installs during development, run:

```bash
pip install -e .
```

---

If you encounter any issues or have questions, please open an issue on the repository.