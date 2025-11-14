import torch
from torchvision.transforms import v2 as T

def get_train_transforms():
    """Torch transforms for training"""
    return T.Compose([
        T.ToImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.Resize((640, 640)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms():
    """Simpler transforms for validation/testing"""
    return T.Compose([
        T.ToImage(),
        T.Resize((640, 640)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
