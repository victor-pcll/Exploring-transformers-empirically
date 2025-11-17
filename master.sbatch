#!/bin/bash
#SBATCH --job-name=master_job
#SBATCH --time=01:00:00

echo "Lancement des sous-jobs…"

for i in {0..9}
do
    sbatch worker.sbatch $i
done