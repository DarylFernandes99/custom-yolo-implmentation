import time
import wandb
from typing import Dict, Union

def setup_wandb(config: Dict[str, Union[str, int, float]], wandb_config: Dict[str, Union[str, int, bool]]):
    wandb_run = wandb.init(
        entity=wandb_config['entity'],
        project=wandb_config['project_name'],
        name=f"{wandb_config['run_name']}_{int(time.time())}",
        config=config
    )

    return wandb_run
