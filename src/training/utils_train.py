import torch
import torch.optim as optim

def get_optimizer(model, lr, weight_decay, patience, factor):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    return optimizer, scheduler

def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir="experiments/checkpoints"):
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }, f"{checkpoint_dir}/model_epoch_{epoch}.pth")
