#!/bin/bash
#SBATCH --account=def-rgmelko
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=2500M
#SBATCH --array=0-7499%10

module load python/3.6
cd /home/ejaazm/projects/def-rgmelko/ejaazm
source ./scaling/bin/activate
python3 ./NNQuST/scaling/tfim_single_run.py $SLURM_ARRAY_TASK_ID
