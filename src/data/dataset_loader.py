import os
import ast
import torch
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial

from PIL import Image
from torchvision import tv_tensors
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F

class DetectionDataset(Dataset):
    """
    Custom Dataset for multi-class object detection.
    Each row in the Parquet file corresponds to one image.
    """

    def __init__(self, parquet_path, image_dir, transform=None, is_test=False):
        """
        Args:
            parquet_path (str): Path to the parquet file (train/val).
            image_dir (str): Directory where the image files (.jpg) are stored.
            transform (callable, optional): Torchvision transforms to apply.
        """
        # Load parquet file into memory (each row = one image)
        self.df = pd.read_parquet(parquet_path)
        print("[INFO] Loaded parquet file - {}".format(parquet_path))
        if is_test:
            self.df = self.df.head(20)
            print("[INFO] Reducing data for test")
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["file_name"])

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Extract annotations
        boxes = torch.from_numpy(np.array(row["bbox"].tolist(), dtype=np.float32))
        labels = torch.from_numpy(np.array(row["category_id"].tolist(), dtype=np.float32))
        labels = labels.unsqueeze(-1).reshape(labels.shape[0], 1)
        name = row["name"]

        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=F.get_size(image)
        )

        # Optional: segmentation masks can be handled later if needed
        # segm = row["segmentation"]

        # Build target dict
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]), 
        }

        # Apply transforms
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        target['boxes'] = torch.cat([target['boxes'], target['labels']], dim=1)
        del target['labels']
        target['name'] = name

        return image, target
