#!/bin/bash

ARG_VALUE="$1"

#SBATCH --job-name="FSDP Training using $ARG_VALUE GPUs"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=$ARG_VALUE
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=courses-gpu

# Load necessary modules (CUDA, Anaconda)
module load cuda/12.3.0 anaconda3/2024.06

# activate conda enironment
source activate hpc_lecture

# Execute model training script
torchrun --nproc_per_node=$ARG_VALUE scripts/fsdp_training.py

# Deactivate the environment
conda deactivate
