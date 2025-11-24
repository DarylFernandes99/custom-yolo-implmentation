import os
import torch
from tqdm import tqdm

from src.training.metrics import DetectionMetrics
from src.training.utils_train import save_checkpoint
from src.training.distributed_setup import reduce_value

def decode_predictions(preds, conf_threshold=0.25, top_k=100):
    """
    Convert raw model predictions to format suitable for metrics computation.
    
    Args:
        preds: (N, 175, 8400) - model outputs
        conf_threshold: minimum confidence threshold
        top_k: maximum number of predictions per image
    
    Returns:
        List of tensors, each (M, 5) containing [x, y, w, h, class_id]
    """
    N = preds.shape[0]
    preds = preds.transpose(1, 2)  # (N, 8400, 175)
    
    batch_predictions = []
    
    for b in range(N):
        pred = preds[b]  # (8400, 175)
        pred_box = pred[:, :4]  # (8400, 4) - xywh
        pred_scores = pred[:, 4:]  # (8400, C) - class scores
        
        # Get max class score and class index for each anchor
        max_scores, class_ids = pred_scores.sigmoid().max(dim=1)  # (8400,), (8400,)
        
        # Filter by confidence threshold
        mask = max_scores >= conf_threshold
        filtered_boxes = pred_box[mask]  # (M, 4)
        filtered_scores = max_scores[mask]  # (M,)
        filtered_classes = class_ids[mask]  # (M,)
        
        if filtered_boxes.numel() == 0:
            batch_predictions.append(torch.zeros(0, 5, device=preds.device))
            continue
        
        # Sort by score and keep top_k
        if filtered_scores.numel() > top_k:
            top_indices = torch.topk(filtered_scores, top_k)[1]
            filtered_boxes = filtered_boxes[top_indices]
            filtered_classes = filtered_classes[top_indices]
        
        # Combine to (M, 5) format
        predictions = torch.cat([
            filtered_boxes,
            filtered_classes.unsqueeze(1).float()
        ], dim=1)
        
        batch_predictions.append(predictions)
    
    return batch_predictions


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    num_epochs,
    device,
    num_classes=171,
    rank=0,
    use_wandb=False,
    wandb_instance=None,
    log_interval=10,
    checkpoint_dir="experiments/checkpoints",
    iou_threshold=0.5,
    conf_threshold=0.25
):
    """
    Training loop for object detection model.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for model parameters
        scheduler: Learning rate scheduler
        criterion: Loss function
        num_epochs: Number of training epochs
        device: Device to train on (cuda:0, cpu, etc.)
        num_classes: Number of object classes
        rank: Process rank for distributed training (0 for single GPU)
        use_wandb: Whether to use Weights & Biases logging
        wandb_instance: Active wandb run instance (if use_wandb=True)
        log_interval: Interval for logging training metrics
        checkpoint_dir: Directory to save model checkpoints
        iou_threshold: IoU threshold for detection metrics
        conf_threshold: Confidence threshold for predictions
    """
    
    # Initialize metrics tracker
    detection_metrics = DetectionMetrics(
        num_classes=num_classes,
        iou_threshold=iou_threshold
    )
    
    for epoch in range(num_epochs):
        # ============ TRAINING ============
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        total_train_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0
        
        train_pbar = tqdm(
            train_loader, 
            desc=f"[Epoch {epoch+1}/{num_epochs}] Training", 
            disable=(rank != 0)
        )
        
        for batch_idx, (images, targets) in enumerate(train_pbar):
            images = images.to(device)
            gt_box = [target["boxes"].to(device) for target in targets]
            
            preds = model(images)
            loss, loss_dict = criterion(preds, gt_box)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            total_train_loss += loss_dict['total_loss']
            total_box_loss += loss_dict['box_loss']
            total_cls_loss += loss_dict['cls_loss']
            
            avg_total_loss = total_train_loss / (batch_idx + 1)
            avg_box_loss = total_box_loss / (batch_idx + 1)
            avg_cls_loss = total_cls_loss / (batch_idx + 1)
            
            train_pbar.set_postfix({
                "Loss": f"{avg_total_loss:.4f}",
                "Box": f"{avg_box_loss:.4f}",
                "Cls": f"{avg_cls_loss:.4f}"
            })
            
            if use_wandb and rank == 0 and batch_idx % log_interval == 0 and wandb_instance is not None:
                step = epoch * len(train_loader) + batch_idx
                wandb_instance.log({
                    "train/total_loss": loss_dict['total_loss'],
                    "train/box_loss": loss_dict['box_loss'],
                    "train/cls_loss": loss_dict['cls_loss'],
                    "step": step
                })
        
        # Average training losses for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_box = total_box_loss / len(train_loader)
        avg_train_cls = total_cls_loss / len(train_loader)

        # Synchronize training losses across all processes
        if rank != -1:
             avg_train_loss = reduce_value(avg_train_loss, average=True)
             avg_train_box = reduce_value(avg_train_box, average=True)
             avg_train_cls = reduce_value(avg_train_cls, average=True)
        
        # ============ VALIDATION ============
        model.eval()
        total_val_loss = 0.0
        total_val_box = 0.0
        total_val_cls = 0.0
        
        # Reset metrics for this epoch
        detection_metrics.reset()
        
        val_pbar = tqdm(
            val_loader, 
            desc=f"[Epoch {epoch+1}/{num_epochs}] Validation", 
            disable=(rank != 0)
        )
        
        with torch.no_grad():
            for val_idx, (images, targets) in enumerate(val_pbar):
                images = images.to(device)
                gt_box = [target["boxes"].to(device) for target in targets]
                
                # Use the same criterion for validation
                preds = model(images)
                loss, loss_dict = criterion(preds, gt_box)
                
                # Accumulate validation losses
                total_val_loss += loss_dict['total_loss']
                total_val_box += loss_dict['box_loss']
                total_val_cls += loss_dict['cls_loss']
                
                # Compute detection metrics
                decoded_preds = decode_predictions(preds, conf_threshold=conf_threshold)
                
                # Update metrics for each image in batch
                for i in range(len(decoded_preds)):
                    if gt_box[i].numel() > 0:
                        detection_metrics.update(decoded_preds[i], gt_box[i])
                
                avg_val_loss = total_val_loss / (val_idx + 1)
                avg_val_box = total_val_box / (val_idx + 1)
                avg_val_cls = total_val_cls / (val_idx + 1)
                
                val_pbar.set_postfix({
                    "Loss": f"{avg_val_loss:.4f}",
                    "Box": f"{avg_val_box:.4f}",
                    "Cls": f"{avg_val_cls:.4f}"
                })
        
        # Average validation losses
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_box = total_val_box / len(val_loader)
        avg_val_cls = total_val_cls / len(val_loader)

        # Synchronize validation losses across all processes
        avg_val_loss = reduce_value(avg_val_loss, average=True)
        avg_val_box = reduce_value(avg_val_box, average=True)
        avg_val_cls = reduce_value(avg_val_cls, average=True)
        
        # Compute detection metrics
        metrics_dict = detection_metrics.compute()
        
        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # ============ LOGGING & CHECKPOINTING ============
        if rank == 0:
            if use_wandb and wandb_instance:
                wandb_instance.log({
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_train_loss,
                    "train/epoch_box_loss": avg_train_box,
                    "train/epoch_cls_loss": avg_train_cls,
                    "val/epoch_loss": avg_val_loss,
                    "val/epoch_box_loss": avg_val_box,
                    "val/epoch_cls_loss": avg_val_cls,
                    "val/precision": metrics_dict['precision'],
                    "val/recall": metrics_dict['recall'],
                    "val/f1_score": metrics_dict['f1_score'],
                    "val/mAP": metrics_dict['mAP'],
                    "lr": optimizer.param_groups[0]["lr"]
                })
            
            save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_dir=checkpoint_dir)
            
            # Display epoch summary using tqdm.write (doesn't interrupt progress bars)
            tqdm.write(f"{'='*80}")
            tqdm.write(f"Epoch {epoch+1}/{num_epochs} Summary:")
            tqdm.write(f"  Train - Total: {avg_train_loss:.4f} | Box: {avg_train_box:.4f} | Cls: {avg_train_cls:.4f}")
            tqdm.write(f"  Val   - Total: {avg_val_loss:.4f} | Box: {avg_val_box:.4f} | Cls: {avg_val_cls:.4f}")
            tqdm.write(f"  Metrics - Precision: {metrics_dict['precision']:.4f} | Recall: {metrics_dict['recall']:.4f} | F1: {metrics_dict['f1_score']:.4f} | mAP: {metrics_dict['mAP']:.4f}")
            tqdm.write(f"  Detection - TP: {metrics_dict['true_positives']} | FP: {metrics_dict['false_positives']} | FN: {metrics_dict['false_negatives']}")
            tqdm.write(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            tqdm.write(f"{'='*80}\n")
