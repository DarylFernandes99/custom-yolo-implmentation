#!/bin/bash

ARG_VALUE="$1"

SLURM_PATH="./slurm/logs/$x.$j"
mkdir -p $SLURM_PATH

#SBATCH --job-name="FSDP Training using $ARG_VALUE GPUs"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=$ARG_VALUE
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=$SLURM_PATH/output.out
#SBATCH --error=$SLURM_PATH/error.err
#SBATCH --partition=courses-gpu

# Load necessary modules (CUDA, Anaconda)
module load cuda/12.3.0 anaconda3/2024.06

# activate conda enironment
source activate hpc_project

# Execute model training script
torchrun --nproc_per_node=$ARG_VALUE scripts/fsdp_training.py

# Deactivate the environment
conda deactivate
