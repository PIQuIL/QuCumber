#!/bin/bash
#SBATCH --account=def-rgmelko
#SBATCH --time=0-00:05
#SBATCH --gres=gpu:1
#SBATCH --mem=2500M
#SBATCH --output=/home/ejaazm/projects/def-rgmelko/ejaazm/tfim1d_dataset_scaling/logs/%x-%J-%t.out

module load python/3.6
cd /home/ejaazm/projects/def-rgmelko/ejaazm
source ./scaling/bin/activate
python3 ./NNQuST/dataset_scaling.py -n $1 -h $2
