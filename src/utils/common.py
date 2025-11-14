import os
import torch

def get_num_workers():
    """
    Dynamically determine the number of DataLoader workers under SLURM.
    Uses SLURM_CPUS_PER_TASK if available, otherwise defaults to a safe heuristic.
    """
    cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
    num_gpus = torch.cuda.device_count() or 1

    if cpus_per_task is not None:
        cpus = int(cpus_per_task)
        num_workers = max(1, cpus // num_gpus)
    else:
        # fallback heuristic if not running under SLURM
        import multiprocessing
        num_workers = max(2, multiprocessing.cpu_count() // num_gpus)

    # Safety cap (avoid excessive forking)
    return min(num_workers, 16)
