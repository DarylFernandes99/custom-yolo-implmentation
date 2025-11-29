#!/bin/bash
#
#SBATCH --job-name="Training using CPUs"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --output=./slurm/logs/%x.%j/output.out
#SBATCH --error=./slurm/logs/%x.%j/error.err
#SBATCH --partition=courses

WORLD_SIZE=${1:-"1"}
MODE=${2:-"None"}
PRECISION=${3:-"None"}
BATCH_SIZE=${4:-"None"}
PREFETCH_FACTOR=${5:-"None"}
DATASET_PERCENT=${6:-"None"}

# Create directory for logs (Slurm provides these variables)
mkdir -p "./slurm/logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}"

# Load necessary modules (CUDA, Anaconda)
module load cuda/12.3.0 anaconda3/2024.06

# activate conda enironment
source activate hpc_project

#############################################################
# Generate Random open port to start torchrun
is_port_in_use() {
    local port=$1
    # Port is in use (connection succeeded)
    if nc -z localhost "$port" &>/dev/null; then
        return 0
    # Port is likely free (connection refused)
    else
        return 1
    fi
}

# Find a free random port
find_free_port() {
    local start_port=49152
    local end_port=65535
    local port

    while true; do
        # Generate a random number between start_port and end_port using 'shuf'
        port=$(shuf -i "$start_port"-"$end_port" -n 1)

        # Check if the generated port is in use
        if is_port_in_use "$port"; then
            echo "Port $port is in use. Trying another one..." >&2
        else
            echo "$port"
            return 0
        fi
    done
}

AVAILABLE_PORT=$(find_free_port)
#############################################################

# Execute model training script with optional arguments
CMD_ARGS=""
if [ "$MODE" != "None" ]; then
    CMD_ARGS="$CMD_ARGS --mode $MODE"
fi
if [ "$PRECISION" != "None" ]; then
    CMD_ARGS="$CMD_ARGS --precision $PRECISION"
fi
if [ "$BATCH_SIZE" != "None" ]; then
    CMD_ARGS="$CMD_ARGS --batch_size $BATCH_SIZE"
fi
if [ "$PREFETCH_FACTOR" != "None" ]; then
    CMD_ARGS="$CMD_ARGS --prefetch_factor $PREFETCH_FACTOR"
fi
if [ "$DATASET_PERCENT" != "None" ]; then
    CMD_ARGS="$CMD_ARGS --dataset_percent $DATASET_PERCENT"
fi
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$AVAILABLE_PORT --nproc_per_node=$WORLD_SIZE scripts/distributed_training.py $CMD_ARGS --device cpu

# Deactivate the environment
conda deactivate
