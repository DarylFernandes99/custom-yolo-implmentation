import os
import glob
import json
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

def get_checkpoint_config(checkpoint_path):
    """
    Extracts the configuration from a checkpoint file.
    """
    try:
        with open(os.path.join(checkpoint_path, "model_config.json"), "r") as f:
            model_cfg_checkpoint = json.load(f)
    except FileNotFoundError:
        print("[WARNING] Model config file not found in checkpoint directory")
        raise FileNotFoundError("Model config file not found in checkpoint directory")
    
    return model_cfg_checkpoint

def find_latest_checkpoint(checkpoint_dir, extension='*.pth'):
    """
    Finds the file with the latest modification time in the specified directory 
    that matches the given extension pattern.
    """
    # Create the search pattern
    search_pattern = os.path.join(checkpoint_dir, extension)
    
    # Get a list of all files matching the pattern
    list_of_files = glob.glob(search_pattern)
    
    # Filter out directories (only keep actual files)
    files = list(filter(os.path.isfile, list_of_files))

    if not files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")

    # Find the latest file based on modification time (getmtime)
    latest_file = max(files, key=os.path.getmtime)
    
    return latest_file