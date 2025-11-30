import time
import wandb
import argparse
from datetime import datetime
from typing import Dict, Union

def setup_wandb(config: Dict[str, Union[str, int, float]], wandb_config: Dict[str, Union[str, int, bool]], args: argparse.Namespace):
    if args.mode == "ddp":
        del config["fsdp"], config["fsdp2"]
    elif args.mode == "fsdp":
        del config["ddp"], config["fsdp2"]
    elif args.mode == "fsdp2":
        del config["ddp"], config["fsdp"]
    
    wandb_run = wandb.init(
        entity=wandb_config['entity'],
        project=wandb_config['project_name'],
        name=f"{args.device}_{args.mode}_{config[args.mode]['precision']}_{wandb_config['run_name']}_{datetime.now().strftime('%d-%m-%Y--%H:%M:%S')}",
        config=config
    )
    print("[INFO] WanDB Initialzed")

    return wandb_run
