#!/bin/bash
#SBATCH --job-name=ERM_untied_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/peucelle/tpiv-simulations/results/run_%x/log.txt
#SBATCH --error=/home/peucelle/tpiv-simulations/results/run_%x/err.txt
#SBATCH --chdir /home/peucelle/tpiv-simulations/experiments
#SBATCH --array=0-9
#SBATCH --mem-per-cpu=7000

# Create a timestamp for the run directory
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Load all modules
module load gcc
module load python

# Activate the environment
source $HOME/venv/bin/activate

# Add src folder to Python path
export PYTHONPATH=/home/peucelle/tpiv-simulations/src:$PYTHONPATH

# Create a single directory for this run timestamp
export RUN_DIR=/home/peucelle/tpiv-simulations/results/run_${RUN_TIMESTAMP}
mkdir -p $RUN_DIR

# Pass both the task ID and run directory to the Python script
$HOME/venv/bin/python /home/peucelle/tpiv-simulations/experiments/ERM_S/ERM_S_exp_cluster.py $SLURM_ARRAY_TASK_ID $RUN_DIR >> ${RUN_DIR}/log.txt 2>> ${RUN_DIR}/err.txt