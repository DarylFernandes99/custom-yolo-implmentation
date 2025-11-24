import time
import wandb
from datetime import datetime
from typing import Dict, Union

def setup_wandb(config: Dict[str, Union[str, int, float]], wandb_config: Dict[str, Union[str, int, bool]], mode: str):
    if mode == "ddp":
        del config["fsdp"], config["fsdp2"]
    elif mode == "fsdp":
        del config["ddp"], config["fsdp2"]
    elif mode == "fsdp2":
        del config["ddp"], config["fsdp"]
    
    wandb_run = wandb.init(
        entity=wandb_config['entity'],
        project=wandb_config['project_name'],
        name=f"{mode}_{wandb_config['run_name']}_{datetime.now().strftime('%d-%m-%Y--%H:%M:%S')}",
        config=config
    )
    print("[INFO] WanDB Initialzed")

    return wandb_run
