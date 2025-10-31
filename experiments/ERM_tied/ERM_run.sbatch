#!/bin/bash
#SBATCH --chdir /home/boncorag/ERM_attention
#SBATCH --job-name=ERM_attention
#SBATCH --array=0-31
#SBATCH --output=ERM_attention.out
#SBATCH --error=ERM_attention.err
#SBATCH --ntasks=1
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=7000
#SBATCH --time=1:00:00

# Load all modules
module load gcc
module load python

# Activate the environment
source .venv/bin/activate




# python ERM_linear.py ${SLURM_ARRAY_TASK_ID} $lreg $kappa
python ERM_softmax.py ${SLURM_ARRAY_TASK_ID} 