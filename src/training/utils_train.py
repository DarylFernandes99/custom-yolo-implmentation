import os
import torch
import functools
from torch import nn
import torch.optim as optim
from typing import Dict, Union, Tuple
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    ShardingStrategy,
    FullyShardedDataParallel as FSDP,
    fully_shard,
    MixedPrecisionPolicy,
)
from src.model.model_blocks import Conv
from src.utils.common import get_num_threads
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision

def get_optimizer(model: nn.Module, lr: float, weight_decay: float, patience: int, factor: float) -> Tuple[optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau]:
    """
    Initializes the AdamW optimizer and ReduceLROnPlateau scheduler.

    Args:
        model (nn.Module): The model for which to create the optimizer.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.

    Returns:
        Tuple[optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau]: A tuple containing the optimizer and scheduler.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    return optimizer, scheduler

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, val_loss: float, checkpoint_dir: str = "experiments/checkpoints") -> None:
    """
    Saves a training checkpoint, including model state, optimizer state, epoch, and validation loss.

    Args:
        model (nn.Module): The model to save.
        optimizer (optim.Optimizer): The optimizer state to save.
        epoch (int): The current training epoch.
        val_loss (float): The validation loss at the current epoch.
        checkpoint_dir (str, optional): Directory where checkpoints will be saved. Defaults to "experiments/checkpoints".
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = f"{checkpoint_dir}/model_epoch_{epoch}.pth"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }, checkpoint_file)
    print(f"[INFO] Saved checkpoint at {checkpoint_file}")

def prepare_fsdp_model(model: nn.Module, device_id: int, config: Dict[str, Union[str, int]], world_size: int, device: str) -> nn.Module:
    """
    Wraps the model with PyTorch's FullyShardedDataParallel (FSDP1) for distributed training.
    
    This function configures mixed precision, auto-wrap policies, and sharding strategies based on the provided config.

    Args:
        model (nn.Module): The PyTorch model to wrap.
        device_id (int): The GPU device ID to use for this process.
        config (Dict[str, Union[str, int]]): Configuration dictionary containing parameters for 'precision', 
                                             'auto_wrap_policy_min_params', and 'sharding_strategy'.
        world_size (int): The total number of processes involved in the distributed training.
        device (str): The device to use for training.

    Returns:
        nn.Module: The FSDP-wrapped model.
    """

    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "cpu":
        torch.set_num_threads(get_num_threads(world_size))
    
    mixed_precision = None
    if config['precision'] in ("bfloat16", "float16"):
        print("[INFO] Setting up precision - {}".format(config['precision']))
        mixed_precision = MixedPrecision(
            param_dtype=getattr(torch, config['precision']),
            reduce_dtype=getattr(torch, config['precision']),
            buffer_dtype=getattr(torch, config['precision']),
            cast_forward_inputs=True,
        )
    
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, 
        min_num_params=int(config['auto_wrap_policy_min_params'])
    )
    
    sharding_strategy = ShardingStrategy.NO_SHARD
    if world_size != 1 and config['sharding_strategy'] in ("FULLY_SHARD", "SHARD_GROUP_OP", "HYBRID_SHARD", "_HYBRID_SHARD_ZERO2"):
        sharding_strategy = ShardingStrategy[config['sharding_strategy']]

    # Create DeviceMesh
    mesh_device_type = "cuda" if device == "cuda" else "cpu"
    mesh = init_device_mesh(mesh_device_type, (world_size,))

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        use_orig_params=True,
        device_id=device_id if device == "cuda" else torch.device("cpu"),
        device_mesh=mesh
    ).to(device)

    return model

def prepare_fsdp2_model(model: nn.Module, device_id: int, config: Dict[str, Union[str, int]], world_size: int, device: str) -> nn.Module:
    """
    Wraps the model using PyTorch's FSDP2 (per-layer fully_shard) for optimized distributed training.

    This implementation applies `fully_shard` in a bottom-up manner:
    1. Iterates through the model modules in reverse order.
    2. Wraps leaf layers (`Conv`, `nn.MaxPool2d`) individually.
    3. Finally, wraps the entire root model.

    This strategy ensures efficient parameter sharding and handles mixed precision settings if configured.
    
    Args:
        model (nn.Module): The PyTorch model to wrap.
        device_id (int): The GPU device ID to use for this process.
        config (Dict[str, Union[str, int]]): Configuration dictionary containing 'precision' settings 
                                             (e.g., "bfloat16", "float16", "float32).
        world_size (int): The total number of processes involved (unused in this specific implementation 
                          but kept for interface consistency).
        device (str): The device to use for training.

    Returns:
        nn.Module: The FSDP2-wrapped model, with layers individually sharded.
    """
    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "cpu":
        torch.set_num_threads(get_num_threads(world_size))

    model = model.to(device_id if device == "cuda" else device)

    mp_policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None, cast_forward_inputs=True)
    if config.get('precision') in ("bfloat16", "float16"):
        dtype = getattr(torch, config['precision'])
        mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype, cast_forward_inputs=True)
        
        # Align buffer precision with parameters to prevent type mismatch errors (e.g. in BatchNorm)
        for buffer in model.buffers():
            buffer.data = buffer.data.to(dtype=dtype)

    # Create DeviceMesh
    mesh_device_type = "cuda" if device == "cuda" else "cpu"
    mesh = init_device_mesh(mesh_device_type, (world_size,))

    for module in reversed(list(model.modules())):
        if isinstance(module, (Conv, nn.MaxPool2d)):
            fully_shard(module, mp_policy=mp_policy, reshard_after_forward=True, mesh=mesh)

    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=True, mesh=mesh)
    
    return model

def prepare_ddp_model(model: nn.Module, device_id: int, config: Dict[str, Union[str, int]], world_size: int, device: str) -> nn.Module:
    """
    Wraps the model with PyTorch's DistributedDataParallel (DDP) for distributed training.

    Args:
        model (nn.Module): The PyTorch model to wrap.
        device_id (int): The GPU device ID to use for this process.
        config (Dict[str, Union[str, int]]): Configuration dictionary.
        world_size (int): The total number of processes involved in the distributed training.
        device (str): The device to use for training.

    Returns:
        nn.Module: The DDP-wrapped model.
    """
    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "cpu":
        torch.set_num_threads(get_num_threads(world_size))
    
    model = model.to(device_id if device == "cuda" else device)

    find_unused_parameters = config.get('find_unused_parameters', False) if config else False

    model = DDP(model, device_ids=[device_id] if device == "cuda" else None, find_unused_parameters=find_unused_parameters)
    
    return model
