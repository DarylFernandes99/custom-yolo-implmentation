#!/bin/bash
#
#SBATCH --job-name="Pre-process Annotations Data"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=./slurm/logs/%x.%j/output.out
#SBATCH --error=./slurm/logs/%x.%j/error.err
#SBATCH --partition=courses

ARG_VALUE="$1"

# Create directory for logs (Slurm provides these variables)
mkdir -p "./slurm/logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}"

# Load necessary modules (Anaconda)
module load anaconda3/2024.06

# activate conda enironment
source activate hpc_project

# Execute model training script
python scripts/data_preprocess.py --mode $ARG_VALUE

# Deactivate the environment
conda deactivate
