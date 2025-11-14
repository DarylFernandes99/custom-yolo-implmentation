import torch
import numpy as np
from typing import Dict, List, Tuple


def box_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes (vectorized).
    
    Args:
        boxes1: (N, 4) tensor in xywh format
        boxes2: (M, 4) tensor in xywh format
        
    Returns:
        iou: (N, M) tensor of IoU values
    """
    # Convert xywh to xyxy
    def xywh_to_xyxy(boxes):
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    boxes1_xyxy = xywh_to_xyxy(boxes1)
    boxes2_xyxy = xywh_to_xyxy(boxes2)
    
    # Calculate intersection
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Calculate union
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])  # (N,)
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])  # (M,)
    union = area1[:, None] + area2[None, :] - inter  # (N, M)
    
    iou = inter / (union + 1e-6)
    return iou


class DetectionMetrics:
    """
    Compute detection metrics including precision, recall, and mAP.
    """
    
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all accumulated statistics."""
        self.total_predictions = 0
        self.total_ground_truths = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        # Per-class statistics
        self.class_tp = torch.zeros(self.num_classes)
        self.class_fp = torch.zeros(self.num_classes)
        self.class_fn = torch.zeros(self.num_classes)
        self.class_gt_count = torch.zeros(self.num_classes)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               pred_scores: torch.Tensor = None, score_threshold: float = 0.5):
        """
        Update metrics with a batch of predictions and targets.
        
        Args:
            predictions: (N, 5) tensor [x, y, w, h, class_id]
            targets: (M, 5) tensor [x, y, w, h, class_id]
            pred_scores: (N,) tensor of confidence scores (optional)
            score_threshold: minimum confidence to consider a prediction
        """
        if predictions.numel() == 0 and targets.numel() == 0:
            return
        
        # Filter predictions by score threshold if provided
        if pred_scores is not None and predictions.numel() > 0:
            mask = pred_scores >= score_threshold
            predictions = predictions[mask]
            if pred_scores is not None:
                pred_scores = pred_scores[mask]
        
        # Handle empty predictions or targets
        if predictions.numel() == 0:
            self.false_negatives += targets.size(0)
            if targets.numel() > 0:
                for cls_id in targets[:, 4].long():
                    if 0 <= cls_id < self.num_classes:
                        self.class_fn[cls_id] += 1
                        self.class_gt_count[cls_id] += 1
            return
        
        if targets.numel() == 0:
            self.false_positives += predictions.size(0)
            if predictions.numel() > 0:
                for cls_id in predictions[:, 4].long():
                    if 0 <= cls_id < self.num_classes:
                        self.class_fp[cls_id] += 1
            return
        
        # Compute IoU between all predictions and targets
        pred_boxes = predictions[:, :4]  # (N, 4)
        target_boxes = targets[:, :4]     # (M, 4)
        
        ious = box_iou_batch(pred_boxes, target_boxes)  # (N, M)
        
        # Match predictions to targets
        matched_targets = set()
        
        for i in range(predictions.size(0)):
            pred_class = predictions[i, 4].long().item()
            
            # Find best matching target
            best_iou = 0
            best_target_idx = -1
            
            for j in range(targets.size(0)):
                if j in matched_targets:
                    continue
                
                target_class = targets[j, 4].long().item()
                
                # Check class match and IoU
                if pred_class == target_class and ious[i, j] > best_iou:
                    best_iou = ious[i, j].item()
                    best_target_idx = j
            
            # Determine TP or FP
            if best_iou >= self.iou_threshold and best_target_idx >= 0:
                self.true_positives += 1
                matched_targets.add(best_target_idx)
                if 0 <= pred_class < self.num_classes:
                    self.class_tp[pred_class] += 1
            else:
                self.false_positives += 1
                if 0 <= pred_class < self.num_classes:
                    self.class_fp[pred_class] += 1
        
        # Unmatched targets are false negatives
        num_unmatched = targets.size(0) - len(matched_targets)
        self.false_negatives += num_unmatched
        
        for j in range(targets.size(0)):
            target_class = targets[j, 4].long().item()
            if 0 <= target_class < self.num_classes:
                self.class_gt_count[target_class] += 1
                if j not in matched_targets:
                    self.class_fn[target_class] += 1
        
        self.total_predictions += predictions.size(0)
        self.total_ground_truths += targets.size(0)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary containing precision, recall, f1-score, etc.
        """
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-6)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        # Per-class precision and recall
        class_precision = self.class_tp / (self.class_tp + self.class_fp + 1e-6)
        class_recall = self.class_tp / (self.class_tp + self.class_fn + 1e-6)
        
        # Mean Average Precision (simplified)
        valid_classes = self.class_gt_count > 0
        if valid_classes.sum() > 0:
            mAP = class_precision[valid_classes].mean().item()
        else:
            mAP = 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'mAP': float(mAP),
            'true_positives': int(self.true_positives),
            'false_positives': int(self.false_positives),
            'false_negatives': int(self.false_negatives),
            'total_predictions': int(self.total_predictions),
            'total_ground_truths': int(self.total_ground_truths)
        }
    
    def get_class_metrics(self, class_id: int) -> Dict[str, float]:
        """Get metrics for a specific class."""
        precision = self.class_tp[class_id] / (self.class_tp[class_id] + self.class_fp[class_id] + 1e-6)
        recall = self.class_tp[class_id] / (self.class_tp[class_id] + self.class_fn[class_id] + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positives': int(self.class_tp[class_id]),
            'false_positives': int(self.class_fp[class_id]),
            'false_negatives': int(self.class_fn[class_id]),
            'ground_truths': int(self.class_gt_count[class_id])
        }


def compute_average_iou(predictions: List[torch.Tensor], 
                        targets: List[torch.Tensor]) -> float:
    """
    Compute average IoU across a batch.
    
    Args:
        predictions: List of (N_i, 4) tensors
        targets: List of (M_i, 4) tensors
        
    Returns:
        Average IoU value
    """
    total_iou = 0.0
    total_pairs = 0
    
    for pred, target in zip(predictions, targets):
        if pred.numel() == 0 or target.numel() == 0:
            continue
        
        ious = box_iou_batch(pred, target)
        max_ious = ious.max(dim=1)[0]  # Best IoU for each prediction
        
        total_iou += max_ious.sum().item()
        total_pairs += pred.size(0)
    
    return total_iou / (total_pairs + 1e-6)

