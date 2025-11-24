import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.utils.common import get_num_workers
from src.data.collate import collate_fn
from src.data.dataset_loader import DetectionDataset
from src.data.transforms import get_train_transforms, get_val_transforms

def get_data_loaders(train_parquet, val_parquet, train_images, val_images, batch_size, is_test: bool = False):
    num_workers = get_num_workers()

    train_dataset = DetectionDataset(train_parquet, train_images, get_train_transforms(), is_test)
    val_dataset = DetectionDataset(val_parquet, val_images, get_val_transforms(), is_test)

    is_distributed = torch.distributed.is_initialized()

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    print("[INFO] Creating Train Loader...", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    print("[INFO] Creating Val Loader...", flush=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader
