#!/bin/bash
#SBATCH --partition=courses
#SBATCH --job-name=train_data_download
#SBATCH --output=./slurm/logs/%x.%j/output.out
#SBATCH --error=./slurm/logs/%x.%j/error.err
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=8

# Create directory for logs (Slurm provides these variables)
mkdir -p "./slurm/logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}"

# Check for argument
if [ -z "$1" ]; then
    echo "Error: No data type specified."
    echo "Usage: sbatch slurm/download_dataset.sbatch <data_type>"
    echo "Available data types: train_images, val_images, test_images, annotations, stuff_annotations, all"
    exit 1
fi

# Run the download script with the provided argument
bash scripts/prepare_dataset.sh "$1"
