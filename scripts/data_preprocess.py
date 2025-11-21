import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import argparse

from src.utils.config_loader import load_config
from src.data.data_preprocessor import DataPreprocess

def main():
    parser = argparse.ArgumentParser(description="Data preprocesing of Annotation files for COCO dataset.")
    parser.add_argument("--mode", type=str, required=True, metavar="M",
                        choices=["train", "val"],
                        help="mode to set which data to process (train - to process training data, val - to process validation data)")
    
    args = parser.parse_args()

    try:
        # Load configuration
        cfg = load_config()
        data_cfg = cfg["data"]

        data_tag = args.mode
        
        DataPreprocess.create_parquet_data(
            annotations_dir=data_cfg['annotations_dir'],
            output_dir=data_cfg['processed_dir'],
            file_names=[f"instances_{data_tag}2017.json", f"stuff_{data_tag}2017.json"],
            keys=["images", "annotations", "categories"],
            columns = [
                ["file_name", "height", "width", "id"],
                ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'],
                ['supercategory', 'id', 'name']
            ],
            chunk_sizes=[1_000, 10_000, 1_000],
            output_folder=data_cfg[f"{data_tag}_parquet"],
            is_test=data_cfg['is_test']
        )

    except Exception as e:
        print("[ERROR] {}".format(str(e)))

if __name__ == "__main__":
    main()
