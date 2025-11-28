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

def get_num_threads(world_size: int):
    """
    Dynamically determine the number of threads per process for CPU training.
    If running distributed CPU training (torchrun), divides total CPUs by world size.
    """

    if world_size > 1:
        try:
            import multiprocessing
            total_cpus = multiprocessing.cpu_count()
            num_threads = total_cpus // world_size
            print(f"Number of threads: {num_threads}")
            return max(1, num_threads)
        except (ValueError, ImportError):
            print("Failed to determine number of threads")
            return 4

    print(f"Number of threads (default): {torch.get_num_threads()}")
    return torch.get_num_threads()