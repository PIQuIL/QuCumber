#!/bin/bash
#SBATCH --account=def-rgmelko
#SBATCH --time=0-03:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-999
#SBATCH --mem=8000M
#SBATCH --output=/home/ejaazm/scratch/tfim1d_dataset_scaling/logs/%x-%J-%t.out

module load python/3.6
cd /home/ejaazm/projects/def-rgmelko/ejaazm
source ./scaling/bin/activate
python3 ./NNQuST/dataset_scaling.py --rerun

# later use array from 1000 to 1799

