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


# Ensure output directories exist
mkdir -p /home/peucelle/tpiv-simulations/results/test_%j

# Load Python module
module load python

# Activate virtual environment
source $HOME/venv/bin/activate

#
# Set PYTHONPATH for project (root of tpiv-simulations)
export PYTHONPATH=/home/peucelle/tpiv-simulations:$PYTHONPATH

echo "🚀 === CLUSTER TEST START ($(date)) ==="
echo "Node: $(hostname)"
echo "Python env: $(which python)"
echo "Directory: $(pwd)"
echo "PYTHONPATH set to: $PYTHONPATH"
echo "------------------------------------"

# Run main cluster diagnostic
python -u test_cluster.py

# Simple CUDA check
python -u -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "------------------------------------"
echo "🏁 === CLUSTER TEST END ($(date)) ==="