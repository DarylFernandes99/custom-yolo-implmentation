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

References:
https://www.analyticsvidhya.com/blog/2025/01/yolov11-model-building/
https://github.com/ultralytics/ultralytics/
https://learnopencv.com/yolo-loss-function-gfl-vfl-loss/
