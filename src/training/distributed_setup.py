import os
import torch
import torch.distributed as dist
from typing import Optional, Union

def init_distributed_mode(device: str):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("[WARNING] Not using distributed mode")
        rank, world_size, gpu = 0, 1, 0

    if device == "cuda":
        torch.cuda.set_device(gpu)

    dist.init_process_group(
        backend="nccl" if device == "cuda" else "gloo", init_method="env://", world_size=world_size, rank=rank
    )
    if device == "cuda":
        dist.barrier(device_ids=[int(rank)])
    else:
        dist.barrier()
    print("[INFO] Distributed process group initialized")
    return rank, world_size, gpu

def reduce_value(value: Union[float, torch.Tensor], average: bool = True) -> Union[float, torch.Tensor]:
    """
    Reduces the value across all processes.
    
    Args:
        value (float or torch.Tensor): Value to reduce
        average (bool): If True, returns the average. If False, returns the sum.
    
    Returns:
        float or torch.Tensor: Reduced value
    """
    if not dist.is_initialized():
        return value
        
    world_size = dist.get_world_size()
    if world_size < 2:
        return value

    with torch.no_grad():
        # Convert float to tensor if needed
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
            # Move to current device if available, else stay on CPU
            if torch.cuda.is_available() and dist.get_backend() == "nccl":
                value = value.cuda()
        
        # Let's check if we need to move it.
        if dist.get_backend() == "nccl" and not value.is_cuda:
            value = value.cuda()
            
        dist.all_reduce(value)
        
        if average:
            value /= world_size
            
    return value.item()

def cleanup_distribute_mode():
    if dist.is_initialized():
        dist.destroy_process_group()
        print("[INFO] Distirbuted process group destroyed")
