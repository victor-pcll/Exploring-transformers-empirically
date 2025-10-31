# Architectural Decision Record: Use of PyTorch for ADR Simulations

## Status
Accepted

## Context
For the ADR (Advection-Diffusion-Reaction) simulations in this project, a robust, flexible, and efficient computational framework is required. The framework should support tensor operations, automatic differentiation, GPU acceleration, and integration with scientific computing workflows.

## Decision
We have decided to use **PyTorch** as the primary computational library for implementing ADR simulations. PyTorch offers dynamic computation graphs, extensive GPU support, and a rich ecosystem for scientific computing and machine learning.

## Consequences
- Code development will leverage PyTorch’s tensor operations and automatic differentiation.
- Simulations can be accelerated using GPUs, improving performance.
- The project will depend on PyTorch and its compatible versions.
- Users must have PyTorch installed in their environment to run simulations.

## Installation

To install PyTorch, use the following commands depending on your environment and hardware:

### CPU-only installation
```bash
pip install torch torchvision torchaudio
```

### GPU installation (NVIDIA CUDA 11.8 example)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Refer to the official PyTorch installation guide for other versions and platforms: https://pytorch.org/get-started/locally/

## Key PyTorch Functions Used in ADR Simulations

- `torch.tensor`: Create tensors for simulation data.
- `torch.autograd`: Automatic differentiation for gradient calculations.
- `torch.nn`: Neural network modules if needed for parameter estimation or surrogate modeling.
- `torch.optim`: Optimization algorithms.
- `torch.cuda`: GPU device management and acceleration.
- `torch.save` and `torch.load`: Save and load model states and simulation results.

## Running Simulations on a Cluster

When running simulations on a cluster, ensure the following:

1. **Environment Setup**  
   Load the appropriate Python environment or module that includes PyTorch:
   ```bash
   module load python/3.x
   pip install --user torch torchvision torchaudio
   ```

2. **GPU Access**  
   Request GPU resources in your job script (example for SLURM):
   ```bash
   #SBATCH --gres=gpu:1
   ```

3. **Job Submission Script**  
   Example SLURM script snippet:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=adr_sim
   #SBATCH --output=adr_sim.out
   #SBATCH --error=adr_sim.err
   #SBATCH --time=02:00:00
   #SBATCH --partition=gpu
   #SBATCH --gres=gpu:1
   #SBATCH --mem=8G

   module load python/3.x
   source ~/envs/adr/bin/activate
   python run_adr_simulation.py
   ```

4. **Data Management**  
   Store input and output data in cluster shared storage or transfer results after job completion.

## References
- PyTorch Official Website: https://pytorch.org/
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- PyTorch Tutorials: https://pytorch.org/tutorials/

---

This ADR will be updated as the project evolves or if alternative computational frameworks are considered.
