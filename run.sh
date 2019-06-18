#!/bin/sh
#SBATCH --account=rrg-rgmelko-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --job-name=scaling-rnn
#SBATCH --output=scaling-rnn.log
#SBATCH --array=0-39

python Run.py $SLURM_ARRAY_TASK_ID
