import time
import wandb
from datetime import datetime
from typing import Dict, Union

def setup_wandb(config: Dict[str, Union[str, int, float]], wandb_config: Dict[str, Union[str, int, bool]]):
    wandb_run = wandb.init(
        entity=wandb_config['entity'],
        project=wandb_config['project_name'],
        name=f"{wandb_config['run_name']}_{datetime.now().strftime('%d-%m-%Y--%H:%M:%S')}",
        config=config
    )
    print("[INFO] WanDB Initialzed")

    return wandb_run
