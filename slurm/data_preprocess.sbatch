#!/bin/bash

ARG_VALUE="$1"

#SBATCH --job-name="Pre-process $ARG_VALUE Annotations Data"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=courses

# Load necessary modules (Anaconda)
module anaconda3/2024.06

# activate conda enironment
source activate hpc_project

# Execute model training script
python scripts/data_preprocess.py --mdoe $ARG_VALUE

# Deactivate the environment
conda deactivate
