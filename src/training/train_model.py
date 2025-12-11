import os
import torch
from tqdm import tqdm
from torch.amp import GradScaler
try:
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
except ImportError:
    ShardedGradScaler = None

from src.training.metrics import DetectionMetrics
from src.training.utils_train import save_checkpoint
from src.training.distributed_setup import reduce_value

def decode_predictions(preds, anchors, strides, conf_threshold=0.25, top_k=100, num_classes=171):
    """
    Convert raw model predictions to format suitable for metrics computation.
    
    Args:
        preds: (N, 4*reg_max + nc, 8400) - raw model outputs
        anchors: (2, 8400) - anchor points
        strides: (1, 8400) - strides
        conf_threshold: minimum confidence threshold
        top_k: maximum number of predictions per image
    
    Returns:
        List of tensors, each (M, 5) containing [x, y, w, h, class_id]
    """
    # preds is (N, C, M)
    # Split box and cls
    # We need to know reg_max. Assuming 16 based on previous context or we can deduce.
    # The Head has self.ch=16. 4*16 = 64.
    # We can pass this or hardcode for now.
    reg_max = 16
    box_channels = 4 * reg_max
    
    pred_dist = preds[:, :box_channels, :] # (N, 64, M)
    pred_scores = preds[:, box_channels:, :] # (N, nc, M)
    
    N = preds.shape[0]
    M_anchors = preds.shape[2]
    
    # Decode boxes using DFL logic (Expectation)
    # Reshape dist to (N, 4, 16, M) -> permute to (N, M, 4, 16)
    pred_dist = pred_dist.view(N, 4, reg_max, M_anchors).permute(0, 3, 1, 2)
    pred_dist = pred_dist.softmax(3)
    
    # Expectation
    values = torch.arange(reg_max, device=preds.device, dtype=preds.dtype)
    pred_box_raw = torch.sum(pred_dist * values, dim=3) # (N, M, 4) -> l,t,r,b
    
    # Convert l,t,r,b to xywh using anchors and strides
    # anchors: (2, M) -> (1, M, 2)
    # strides: (1, M) -> (1, M, 1)
    anchors_t = anchors.transpose(0, 1).unsqueeze(0) # (1, M, 2)
    strides_t = strides.transpose(0, 1).unsqueeze(0) # (1, M, 1)
    
    # pred_box_raw is (l, t, r, b) relative to anchor
    # x1 = anchor_x - l * stride
    # y1 = anchor_y - t * stride
    # x2 = anchor_x + r * stride
    # y2 = anchor_y + b * stride
    
    # Wait, the Head implementation of decoding was:
    # a, b = self.dfl(box).chunk(2, 1)  (where box was 64 channels)
    # a = anchors - a
    # b = anchors + b
    # box = ((a+b)/2, b-a) * stride
    
    # Here pred_box_raw corresponds to the output of self.dfl(box) (which is 4 channels)
    # split into lt (2) and rb (2)?
    # The DFL block in model_blocks.py:
    # return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
    # It returns (N, 4, M).
    # So pred_box_raw (N, M, 4) is correct (just transposed).
    
    # Let's follow Head logic:
    # pred_box_raw is (N, M, 4). Let's transpose to (N, 4, M) to match Head logic if needed, 
    # but we can work with (N, M, 4).
    
    lt = pred_box_raw[:, :, :2] # (N, M, 2)
    rb = pred_box_raw[:, :, 2:] # (N, M, 2)
    
    # a = anchors - lt
    # b = anchors + rb
    # But wait, Head logic:
    # a, b = dfl_out.chunk(2, 1) -> dfl_out is (N, 4, M)
    # So a is (N, 2, M) (lt?), b is (N, 2, M) (rb?)
    # a = anchors.unsqueeze(0) - a
    # b = anchors.unsqueeze(0) + b
    # box = torch.cat(((a+b)/2, b-a), dim=1)
    # This means 'a' and 'b' became x1y1 and x2y2?
    # No.
    # If dfl_out is [l, t, r, b], then:
    # a = [l, t], b = [r, b]
    # new_a = anchor - [l, t] = [ax-l, ay-t] = [x1, y1]
    # new_b = anchor + [r, b] = [ax+r, ay+b] = [x2, y2]
    # center = (new_a + new_b) / 2 = [(x1+x2)/2, (y1+y2)/2] = [cx, cy]
    # size = new_b - new_a = [x2-x1, y2-y1] = [w, h]
    
    # So yes, this logic converts ltrb to xywh (center format).
    # And finally * strides.
    
    x1y1 = anchors_t - lt
    x2y2 = anchors_t + rb
    
    xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    
    pred_box = torch.cat([xy, wh], dim=2) * strides_t # (N, M, 4)
    
    batch_predictions = []
    
    for b in range(N):
        # Per image
        boxes = pred_box[b] # (M, 4)
        scores = pred_scores[b].transpose(0, 1).sigmoid() # (M, nc)
        
        # Filter by confidence
        max_scores, class_ids = scores.max(dim=1)
        mask = max_scores >= conf_threshold
        
        filtered_boxes = boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = class_ids[mask]
        
        if filtered_boxes.numel() == 0:
            batch_predictions.append(torch.zeros(0, 5, device=preds.device))
            continue
            
        if filtered_scores.numel() > top_k:
            top_indices = torch.topk(filtered_scores, top_k)[1]
            filtered_boxes = filtered_boxes[top_indices]
            filtered_classes = filtered_classes[top_indices]
            
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
    initial_epoch,
    num_epochs,
    device,
    num_classes=171,
    rank=0,
    use_wandb=False,
    wandb_instance=None,
    log_interval=10,
    checkpoint_dir="experiments/checkpoints",
    iou_threshold=0.5,
    conf_threshold=0.25,
    distributed_mode="ddp",
    precision="float32"
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
        initial_epoch: Initial epoch for training
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
        distributed_mode: Mode of distributed training ("ddp", "fsdp", "fsdp2")
        precision: Precision mode ("float32", "float16", "bfloat16")
    """
    
    # Initialize mixed precision components
    use_amp = precision in ["float16", "bfloat16"]
    scaler = None
    
    if use_amp and precision == "float16":
        if distributed_mode.startswith("fsdp") and device == "cuda":
            if ShardedGradScaler is not None:
                scaler = ShardedGradScaler()
                if rank == 0:
                    print("[INFO] Initialized ShardedGradScaler for FSDP float16 training")
            else:
                scaler = GradScaler(device)
                if rank == 0:
                    print("[INFO] ShardedGradScaler not found, falling back to GradScaler for float16")
        else:
            scaler = GradScaler(device)
            if rank == 0:
                print(f"[INFO] Initialized GradScaler for {distributed_mode} float16 training on {device}")
    elif use_amp and precision == "bfloat16" and rank == 0:
        print("[INFO] Using bfloat16 precision (no scaler needed)")

    # Initialize metrics tracker
    detection_metrics = DetectionMetrics(
        num_classes=num_classes,
        iou_threshold=iou_threshold
    )
    
    for epoch in range(initial_epoch, num_epochs):
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
            
            optimizer.zero_grad()
            
            enable_autocast = (distributed_mode == "ddp" and use_amp)
            amp_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
            
            with torch.autocast(device_type="cpu" if device == "cpu" else "cuda", dtype=amp_dtype, enabled=enable_autocast):
                preds, anchors, strides = model(images)
                loss, loss_dict = criterion(preds, gt_box, anchors, strides)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
                
                enable_autocast = (distributed_mode == "ddp" and use_amp)
                amp_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16

                with torch.autocast(device_type="cpu" if device == "cpu" else "cuda", dtype=amp_dtype, enabled=enable_autocast):
                    preds, anchors, strides = model(images)
                    loss, loss_dict = criterion(preds, gt_box, anchors, strides)
                
                # Accumulate validation losses
                total_val_loss += loss_dict['total_loss']
                total_val_box += loss_dict['box_loss']
                total_val_cls += loss_dict['cls_loss']
                
                # Compute detection metrics
                decoded_preds = decode_predictions(preds, anchors, strides, conf_threshold=conf_threshold, num_classes=num_classes)
                
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
            
            save_checkpoint(model, optimizer, epoch+1, avg_val_loss, checkpoint_dir=checkpoint_dir)
            
            # Display epoch summary using tqdm.write (doesn't interrupt progress bars)
            tqdm.write(f"{'='*80}")
            tqdm.write(f"Epoch {epoch+1}/{num_epochs} Summary:")
            tqdm.write(f"  Train - Total: {avg_train_loss:.4f} | Box: {avg_train_box:.4f} | Cls: {avg_train_cls:.4f}")
            tqdm.write(f"  Val   - Total: {avg_val_loss:.4f} | Box: {avg_val_box:.4f} | Cls: {avg_val_cls:.4f}")
            tqdm.write(f"  Metrics - Precision: {metrics_dict['precision']:.4f} | Recall: {metrics_dict['recall']:.4f} | F1: {metrics_dict['f1_score']:.4f} | mAP: {metrics_dict['mAP']:.4f}")
            tqdm.write(f"  Detection - TP: {metrics_dict['true_positives']} | FP: {metrics_dict['false_positives']} | FN: {metrics_dict['false_negatives']}")
            tqdm.write(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            tqdm.write(f"{'='*80}\n")
