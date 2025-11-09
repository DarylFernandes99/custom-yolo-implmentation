import torch

def collate_fn(batch):
    """
    Custom collate function to handle variable-sized targets.
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets
