import os
import time
import torch
import wandb
# import itertools
import functools
from torch import nn
# from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision

from src.model.model_builder import Model
from src.model.losses import YoloDFLQFLoss
from src.training.train_model import train
from src.utils.config_loader import load_config
from src.training.wandb_setup import setup_wandb
from src.data.data_loader import get_data_loaders
from src.training.utils_train import get_optimizer
from src.training.distributed_setup import init_distributed_mode, cleanup_distribute_mode

# def custom_full_shard(module: nn.Module):
#     return fully_shard(module, reshard_after_forward=True)

def prepare_fsdp_model(model, device_id, config, world_size):
    """
    Wrap model with FSDP for distributed training.
    
    Args:
        model: The model to wrap
        device_id: GPU device ID
        
    Returns:
        FSDP-wrapped model
    """
    
    model = model.to(device_id)
    
    mixed_precision = None
    if config['precision'] in ("bfloat16", "float16", "float32"):
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

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        use_orig_params=True,
        ignored_modules=[m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    )

    return model


def main():
    """
    Main training entry point with distributed setup.
    """
    # Load configuration
    cfg = load_config()
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    
    # Initialize distributed training
    rank, world_size, gpu = init_distributed_mode()
    
    # Initialize Weights & Biases (only on rank 0)
    use_wandb = cfg.get("wandb", {}).get("enable", False)
    wandb_run = None
    try:
        if rank == 0 and use_wandb:
            wandb_config = cfg["wandb"]
            config = {
                "gpu": world_size,
                **training_cfg
            }
            wandb_run = setup_wandb(config, wandb_config)
        
        # Create model
        model = Model(**model_cfg['config'], num_classes=model_cfg.get('num_classes', 172))
        print("[INFO] Model initialized")

        if training_cfg['fsdp']['use_fsdp']:
            model = prepare_fsdp_model(model, gpu, training_cfg['fsdp'], world_size)
        
        # Get data loaders
        train_loader, val_loader = get_data_loaders(
            os.path.join(cfg['data']['processed_dir'], cfg['data']['train_parquet']),
            os.path.join(cfg['data']['processed_dir'], cfg['data']['val_parquet']),
            cfg['data']['train_images'],
            cfg['data']['val_images'],
            training_cfg['batch_size'],
            is_test=training_cfg['is_test']
        )
        
        # Setup optimizer and scheduler
        optimizer, scheduler = get_optimizer(
            model,
            training_cfg['learning_rate'],
            training_cfg['weight_decay'],
            training_cfg['learning_rate_patience'],
            training_cfg['learning_rate_factor']
        )
        
        # Setup loss criterion
        criterion = YoloDFLQFLoss(
            num_classes=model_cfg['num_classes'],
            lambda_box=training_cfg.get('lambda_box', 1.5),
            lambda_cls=training_cfg.get('lambda_cls', 1.0)
        )
        
        # Start training
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=training_cfg["epochs"],
            device=gpu,
            num_classes=model_cfg['num_classes'],
            rank=rank,
            use_wandb=use_wandb,
            wandb_instance=wandb_run,
            log_interval=training_cfg.get('log_interval', 10),
            checkpoint_dir=training_cfg.get('checkpoint_dir', 'experiments/checkpoints'),
            iou_threshold=training_cfg.get('iou_threshold', 0.5),
            conf_threshold=training_cfg.get('conf_threshold', 0.25)
        )
    except Exception as e:
        print("[ERROR] {}".format(str(e)))
    finally:
        # Cleanup
        if rank == 0 and use_wandb and wandb_run:
            wandb.finish()
            print("[INFO] WanDB destroyed")
        
        cleanup_distribute_mode()

if __name__ == "__main__":
    main()
