# custom-yolo-implmentation

[Dataset link](https://cocodataset.org/#download) 


## Create environment.yml file
```bash
conda env export --from-history > environment.yml
```

## Create conda environment
`Note: Make sure you have conda installed on your system before running the following commands.`
```bash
conda env create -f environment.yml
conda activate hpc_project
```

# Start distributed training
```
torchrun \ 
    --nproc_per_node=<world_size> \
    scripts/distributed_training.py \
    $CMD_ARGS \
    --device <device [cpu or cuda]> \
    --precision <precision [bfloat16 or float16]> \
    --batch_size <batch_size> \
    --prefetch_factor <prefetch_factor>
```

## Run SLURM script
### For distributed training using CPUs
```bash
sbatch \ 
    --export=WANDB_API_KEY="<enter_wandb_api_key>" \ 
    slurm/distributed_training_cpu.sbatch \
    <world_size default=1> \ 
    <mode [train or val]> \ 
    <precision [bfloat16 or float16]> \ 
    <batch_size> \ 
    <prefetch_factor>
```

### For distributed training using GPUs
```bash
sbatch \ 
    --export=WANDB_API_KEY="<enter_wandb_api_key>" \ 
    slurm/distributed_training_gpu.sbatch \ 
    <mode [train or val]> \ 
    <precision [bfloat16 or float16]> \ 
    <batch_size> \ 
    <prefetch_factor>
```

References:
https://www.analyticsvidhya.com/blog/2025/01/yolov11-model-building/
https://github.com/ultralytics/ultralytics/
https://learnopencv.com/yolo-loss-function-gfl-vfl-loss/
