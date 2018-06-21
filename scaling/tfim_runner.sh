#!/bin/bash
#SBATCH --account=def-rgmelko
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=2500M
#SBATCH --array=0-499%25
#SBATCH --output=/home/ejaazm/projects/def-rgmelko/ejaazm/tfim1d_scaling/logs/%x-%J.out

module load python/3.6
cd /home/ejaazm/projects/def-rgmelko/ejaazm
source ./scaling/bin/activate
python3 ./NNQuST/tfim_single_run.py -n $1 -h $2
