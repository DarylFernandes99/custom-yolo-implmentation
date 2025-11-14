import time
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from src.model.model_builder import Model
from src.model.losses import YoloDFLQFLoss
from src.training.train_model import train
from src.utils.config_loader import load_config
from src.data.data_loader import get_data_loaders
from src.training.utils_train import get_optimizer
from src.training.distributed_setup import init_distributed_mode


def prepare_fsdp_model(model, device_id):
    """
    Wrap model with FSDP for distributed training.
    
    Args:
        model: The model to wrap
        device_id: GPU device ID
        
    Returns:
        FSDP-wrapped model
    """
    auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=1e7)
    model = model.to(device_id)
    fsdp_model = FSDP(model, auto_wrap_policy=auto_wrap_policy, mixed_precision=True)
    return fsdp_model


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
    use_wandb = cfg.get("wandb", {}).get("enabled", True)
    wandb_run = None
    
    if rank == 0 and use_wandb:
        wandb_config = cfg["wandb"]
        config = {
            "gpu": world_size,
            **training_cfg
        }
        wandb_run = wandb.init(
            project=wandb_config['project_name'],
            name=f"{wandb_config['run_name']}_{int(time.time())}",
            config=config
        )
    
    # Create model
    model = Model(num_classes=model_cfg.get('num_classes', 172))
    fsdp_model = prepare_fsdp_model(model, gpu)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        cfg['data']['train_parquet'],
        cfg['data']['val_parquet'],
        cfg['data']['train_images'],
        cfg['data']['val_images'],
        training_cfg['batch_size'],
    )
    
    # Setup optimizer and scheduler
    optimizer, scheduler = get_optimizer(
        fsdp_model,
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
        model=fsdp_model,
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
    
    # Cleanup
    if rank == 0 and use_wandb and wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
