import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import time
import json
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
from src.utils.common import get_checkpoint_config, find_latest_checkpoint
from src.training.distributed_setup import init_distributed_mode, cleanup_distribute_mode
from src.training.utils_train import prepare_fsdp_model, prepare_ddp_model, prepare_fsdp2_model

def main(args):
    """
    Main training entry point with distributed setup.

    Args:
        args: Command-line arguments.
    """

    # Load configuration
    cfg = load_config()
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    checkpoint_cfg = cfg["checkpoint"]
    initial_epoch = 0
    
    postfix = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')

    if args.load_from_checkpoint:
        checkpoint_dir = os.path.join(checkpoint_cfg.get('checkpoint_dir', 'experiments/checkpoints'), args.load_from_checkpoint)
        model_cfg_checkpoint = get_checkpoint_config(checkpoint_dir)
        model_cfg['config'] = model_cfg_checkpoint['config']
        model_cfg['num_classes'] = model_cfg_checkpoint['num_classes']
        args.precision = model_cfg_checkpoint['precision']
        args.mode = model_cfg_checkpoint['mode']
        print("[INFO] Loaded model config from checkpoint directory: precision = {}, mode = {}".format(args.precision, args.mode))
    else:
        checkpoint_dir = os.path.join(checkpoint_cfg.get('checkpoint_dir', 'experiments/checkpoints'), postfix)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # save the model config in a json file in the checkpoint directory along with the num_classes
        with open(os.path.join(checkpoint_dir, "model_config.json"), "w") as f:
            json.dump({
                "config": model_cfg['config'],
                "num_classes": model_cfg.get('num_classes', 172),
                "mode": args.mode,
                "precision": args.precision
            }, f)
            print("[INFO] Model config saved to checkpoint directory")
    
    # Initialize distributed training
    rank, world_size, gpu = init_distributed_mode(device=args.device)
    
    # Initialize Weights & Biases (only on rank 0)
    use_wandb = cfg.get("wandb", {}).get("enable", False)
    wandb_run = None
    try:
        # Override config with CLI argument BEFORE model initialization
        if args.mode == "fsdp":
            training_cfg['fsdp']['precision'] = args.precision
        elif args.mode == "fsdp2":
            training_cfg['fsdp2']['precision'] = args.precision
        elif args.mode == "ddp":
            training_cfg['ddp']['precision'] = args.precision
        
        if args.batch_size is not None:
            training_cfg['batch_size'] = args.batch_size
        
        if args.prefetch_factor is not None:
            data_cfg['prefetch_factor'] = args.prefetch_factor

        if rank == 0 and use_wandb:
            wandb_config = cfg["wandb"]
            config = {
                "device": args.device,
                "world_size": world_size,
                "mode": args.mode,
                "checkpoint_path": checkpoint_dir,
                **training_cfg
            }
            wandb_run = setup_wandb(config=config, wandb_config=wandb_config, args=args)
        
        # Create model
        model = Model(**model_cfg['config'], num_classes=model_cfg['num_classes'])

        if args.mode == "fsdp":
            model = prepare_fsdp_model(model=model, device_id=gpu, config=training_cfg['fsdp'], world_size=world_size, device=args.device)
            print("[INFO] FSDP model initialzed")
        elif args.mode == "fsdp2":
            model = prepare_fsdp2_model(model=model, device_id=gpu, config=training_cfg['fsdp2'], world_size=world_size, device=args.device)
            print("[INFO] FSDP2 model initialzed")
        elif args.mode == "ddp":
            model = prepare_ddp_model(model=model, device_id=gpu, config=training_cfg['ddp'], world_size=world_size, device=args.device)
            print("[INFO] DDP model initialzed")
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
        
        model_summary_string = str(summary(model, input_size=(1, 3, 640, 640), verbose=0, device=args.device))
        if rank == 0 and wandb_run is not None:
            summary_table = wandb.Table(columns=["PyTorch Model Summary"], data=[[model_summary_string]])
            artifact = wandb.Artifact(
                f"model-architecture-{postfix}", type="model_summary",
                description="Summary of the PyTorch model architecture using torchinfo."
            )
            artifact.add(summary_table, "model_summary_table")
            wandb_run.log_artifact(artifact)
        
        model = model.to(args.device)
        
        # Get data loaders
        train_loader, val_loader = get_data_loaders(
            train_parquet=os.path.join(data_cfg['processed_dir'], data_cfg['train_parquet']),
            val_parquet=os.path.join(data_cfg['processed_dir'], data_cfg['val_parquet']),
            train_images=data_cfg['train_images'],
            val_images=data_cfg['val_images'],
            batch_size=training_cfg['batch_size'],
            is_test=training_cfg['is_test'],
            prefetch_factor=data_cfg.get('prefetch_factor', 2),
            percent=args.dataset_percent,
            device=args.device
        )
        
        # Setup optimizer and scheduler
        optimizer, scheduler = get_optimizer(
            model=model,
            lr=training_cfg['learning_rate'],
            weight_decay=training_cfg['weight_decay'],
            patience=training_cfg['learning_rate_patience'],
            factor=training_cfg['learning_rate_factor']
        )
        
        # Load checkpoint if specified
        if args.load_from_checkpoint:
            # Get the latest checkpoint
            checkpoint_path = find_latest_checkpoint(checkpoint_dir)
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            
            # Update initial epoch
            initial_epoch = checkpoint['epoch']
            
            # Load model state
            model.load_state_dict(checkpoint["model_state"])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            print(f"[INFO] Loaded model and optimizer from checkpoint at epoch {checkpoint['epoch']} from {checkpoint_path}")
        
        # Setup loss criterion
        criterion = YoloDFLQFLoss(
            num_classes=model_cfg['num_classes'],
            lambda_box=training_cfg['weights'].get('bbox_loss', 1.5),
            lambda_cls=training_cfg['weights'].get('cls_loss', 1.0)
        )

        # Start training
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            initial_epoch=initial_epoch,
            num_epochs=initial_epoch + training_cfg["epochs"],
            device=gpu if args.device == "cuda" else "cpu",
            num_classes=model_cfg['num_classes'],
            rank=rank,
            use_wandb=use_wandb,
            wandb_instance=wandb_run,
            log_interval=training_cfg.get('log_interval', 10),
            checkpoint_dir=checkpoint_dir,
            iou_threshold=training_cfg.get('iou_threshold', 0.5),
            conf_threshold=training_cfg.get('conf_threshold', 0.25),
            distributed_mode=args.mode,
            precision=args.precision
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("[ERROR] {}".format(str(e)))
    finally:
        # Cleanup
        if rank == 0 and use_wandb and wandb_run:
            wandb.finish()
            print("[INFO] WanDB destroyed")
        
        cleanup_distribute_mode()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disrtibuted training with FSDP or DDP.")
    parser.add_argument("--device", type=str, default="cuda", metavar="D",
                        choices=["cpu", "cuda"],
                        help="device to use for training (default: cuda)")
    parser.add_argument("--mode", type=str, required=True, metavar="M",
                        choices=["fsdp", "ddp", "fsdp2"],
                        help="mode to set which distributed training to use (fsdp - for FSDP, ddp - for DDP, fsdp2 - for FSDP2)")
    parser.add_argument("--precision", type=str, default="float32", metavar="P",
                        choices=["bfloat16", "float16", "float32"],
                        help="precision to use for training (default: float32)")
    parser.add_argument("--batch_size", type=int, default=None, metavar="B",
                        help="batch size to use for training (default: use config.yaml batch_size)")
    parser.add_argument("--prefetch_factor", type=int, default=None, metavar="F",
                        help="prefetch factor to use for training (default: use config.yaml prefetch_factor)")
    parser.add_argument("--dataset_percent", type=float, default=1.0, metavar="DP",
                        help="percent of the dataset to use for training (default: 1.0)")
    parser.add_argument("--load_from_checkpoint", type=str, default=None, metavar="LC",
                        help="checkpoint folder name to load from (default: None)")
    args = parser.parse_args()
    
    main(args)
