#!/bin/bash

ARG_VALUE="$1"

SLURM_PATH="./slurm/logs/$x.$j"
mkdir -p $SLURM_PATH

#SBATCH --job-name="Pre-process $ARG_VALUE Annotations Data"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=$SLURM_PATH/output.out
#SBATCH --error=$SLURM_PATH/error.err
#SBATCH --partition=courses

# Load necessary modules (Anaconda)
module anaconda3/2024.06

# activate conda enironment
source activate hpc_project

# Execute model training script
python scripts/data_preprocess.py --mode $ARG_VALUE

# Deactivate the environment
conda deactivate
