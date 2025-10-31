#!/bin/bash
#SBATCH --job-name=test_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/home/peucelle/tpiv-simulations/results/test_%j/log.txt
#SBATCH --error=/home/peucelle/tpiv-simulations/results/test_%j/err.txt
#SBATCH --chdir=/home/peucelle/tpiv-simulations/test

module load python
source $HOME/venv/bin/activate
export PYTHONPATH=/home/peucelle/tpiv-simulations/src:$PYTHONPATH

echo "=== JOB START ($(date)) ==="
echo "Node: $(hostname)"
echo "Python env: $(which python)"
echo "Directory: $(pwd)"
echo "------------------------------------"

# Run main cluster diagnostic
python -u test_cluster.py dummy_argument

# Simple CUDA check
python -u -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "------------------------------------"
echo "=== JOB END ($(date)) ==="