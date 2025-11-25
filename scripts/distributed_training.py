import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import time
import torch
import wandb
import argparse
import functools
from datetime import datetime

from torchinfo import summary
from src.model.model_builder import Model
from src.model.losses import YoloDFLQFLoss
from src.training.train_model import train
from src.utils.config_loader import load_config
from src.training.wandb_setup import setup_wandb
from src.data.data_loader import get_data_loaders
from src.training.utils_train import get_optimizer
from src.training.distributed_setup import init_distributed_mode, cleanup_distribute_mode
from src.training.utils_train import prepare_fsdp_model, prepare_ddp_model, prepare_fsdp2_model

def main():
    """
    Main training entry point with distributed setup.
    """
    parser = argparse.ArgumentParser(description="Disrtibuted training with FSDP or DDP.")
    parser.add_argument("--mode", type=str, required=True, metavar="M",
                        choices=["fsdp", "ddp", "fsdp2"],
                        help="mode to set which distributed training to use (fsdp - for FSDP, ddp - for DDP, fsdp2 - for FSDP2)")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config()
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    checkpoint_cfg = cfg["checkpoint"]
    
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
                "mode": args.mode,
                **training_cfg
            }
            wandb_run = setup_wandb(config, wandb_config, args.mode)
        
        # Create model
        model = Model(**model_cfg['config'], num_classes=model_cfg.get('num_classes', 172))

        if args.mode == "fsdp":
            model = prepare_fsdp_model(model, gpu, training_cfg['fsdp'], world_size)
            print("[INFO] FSDP model initialzed")
        elif args.mode == "fsdp2":
            model = prepare_fsdp2_model(model, gpu, training_cfg['fsdp2'], world_size)
            print("[INFO] FSDP2 model initialzed")
        elif args.mode == "ddp":
            model = prepare_ddp_model(model, gpu, training_cfg['ddp'])
            print("[INFO] DDP model initialzed")
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
        
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
            lambda_box=training_cfg['weights'].get('bbox_loss', 1.5),
            lambda_cls=training_cfg['weights'].get('cls_loss', 1.0)
        )
        
        postfix = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')
        checkpoint_dir = os.path.join(checkpoint_cfg.get('checkpoint_dir', 'experiments/checkpoints'), postfix)
        
        model_summary_string = str(summary(model, input_size=(1, 3, 640, 640), verbose=0))
        if wandb_run is not None:
            summary_table = wandb.Table(columns=["PyTorch Model Summary"], data=[[model_summary_string]])
            artifact = wandb.Artifact(f"model-architecture-{postfix}", type="model_summary",
                          description="Summary of the PyTorch model architecture using torchinfo.")
            artifact.add(summary_table, "model_summary_table")
            wandb_run.log_artifact(artifact)
        
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
            checkpoint_dir=checkpoint_dir,
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
