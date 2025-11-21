#!/bin/bash
#
#SBATCH --job-name="FSDP Training using GPUs"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=./slurm/logs/%x.%j/output.out
#SBATCH --error=./slurm/logs/%x.%j/error.err
#SBATCH --partition=courses-gpu

# Create directory for logs (Slurm provides these variables)
mkdir -p "./slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

# Load necessary modules (CUDA, Anaconda)
module load cuda/12.3.0 anaconda3/2024.06

# activate conda enironment
source activate hpc_project

# Execute model training script
torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE scripts/fsdp_training.py

# Deactivate the environment
conda deactivate
