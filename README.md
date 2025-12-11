# Multi-Class Object Detection with Distributed Training

This project implements a scalable multi-class object detection system using a custom YOLO-like architecture. It is designed for high-performance computing (HPC) environments, supporting distributed training strategies like FSDP (Fully Sharded Data Parallel) and DDP (Distributed Data Parallel).

## Features

-   **Scalable Distributed Training**: Supports FSDP and DDP for training on multiple GPUs/nodes.
-   **Efficient Data Pipeline**: Utilizes Dask and Parquet for optimized loading and processing of large datasets (e.g., COCO).
-   **Mixed Precision**: Accelerated training with `bfloat16` and `float16` support.
-   **Experiment Tracking**: Integrated with Weights & Biases (WandB) for real-time monitoring.
-   **Centralized Configuration**: All parameters managed via `config.yaml`.

## Installation

1.  **Prerequisites**:
    -   Anaconda or Miniconda
    -   Python 3.12+
    -   CUDA-enabled GPU ecosystem

2.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

3.  **Create Conda Environment**:
    ```bash
    conda env create -f environment.yml
    conda activate hpc_project
    ```
    
    *Alternatively, create from history:*
    ```bash
    conda env export --from-history > environment.yml
    ```

## Data Preparation

The project expects the COCO 2017 dataset.

1.  **Download Dataset**:
    [COCO Dataset Download](https://cocodataset.org/#download) (Train/Val images and annotations).
    Place them in `dataset/raw/`.

2.  **Preprocess Annotations**:
    Convert COCO JSON annotations to efficient Parquet format using the provided script.
    ```bash
    python scripts/data_preprocess.py --mode train
    python scripts/data_preprocess.py --mode val
    ```
    *Note: Ensure `config.yaml` points to the correct directories.*

## Configuration

The `config.yaml` file controls all aspects of the project.

-   **`data`**: Paths to input/output directories, worker counts, and batch sizes.
-   **`model`**: Architecture settings (input size, depth, width).
-   **`training`**: Hyperparameters (epochs, lr, optimizer), distributed strategy (`fsdp`, `ddp`), and loss weights.
-   **`wandb`**: Weights & Biases logging settings.
-   **`checkpoint`**: Model saving frequency and directory.

## Usage

### Local Training

For debugging or single-node training:

```bash
torchrun --nproc_per_node=1 scripts/distributed_training.py
```

*Note: The script reads arguments primarily from `config.yaml`.*

### Distributed Training (HPC / SLURM)

This project is optimized for SLURM-managed clusters.

**1. CPU Training:**
Edit `slurm/distributed_training_cpu.sbatch` to set `MODE`, `BATCH_SIZE`, `PRECISION`, etc.
```bash
# Edit parameters inside the script first
vi slurm/distributed_training_cpu.sbatch

# Submit the job
sbatch slurm/distributed_training_cpu.sbatch
```

**2. GPU Training:**
Edit `slurm/distributed_training_gpu.sbatch` to set parameters.
```bash
# Edit parameters inside the script first
vi slurm/distributed_training_gpu.sbatch

# Submit the job
sbatch slurm/distributed_training_gpu.sbatch
```

**Configurable Variables in Script:**
- `MODE`: `train` or `val`
- `PRECISION`: `bfloat16`, `float16`, or `float32`
- `BATCH_SIZE`: Integer
- `PREFETCH_FACTOR`: Integer
- `DATASET_PERCENT`: Percentage of dataset to use (e.g. `0.1`), optional.
- `WORLD_SIZE`: (CPU only) Number of processes.

*If using Weights & Biases:*
```bash
export WANDB_API_KEY="your_key_here"
sbatch --export=WANDB_API_KEY=$WANDB_API_KEY slurm/distributed_training_gpu.sbatch
```

## References

-   [YOLOv11 Model Building](https://www.analyticsvidhya.com/blog/2025/01/yolov11-model-building/)
-   [Ultralytics YOLO](https://github.com/ultralytics/ultralytics/)
-   [YOLO Loss Function (GFL/VFL)](https://learnopencv.com/yolo-loss-function-gfl-vfl-loss/)
