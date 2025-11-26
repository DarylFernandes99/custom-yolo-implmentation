import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# HELPER: IoU (vectorized)
# -----------------------------------------------------------
def bbox_iou(box1, box2):
    """
    box1: (M, 4)
    box2: (M, 4)
    All in xywh format
    Returns IoU for each pair (M,)
    """
    # xywh → xyxy
    b1_x1 = box1[:, 0] - box1[:, 2] / 2
    b1_y1 = box1[:, 1] - box1[:, 3] / 2
    b1_x2 = box1[:, 0] + box1[:, 2] / 2
    b1_y2 = box1[:, 3] + box1[:, 1] / 2

    b2_x1 = box2[:, 0] - box2[:, 2] / 2
    b2_y1 = box2[:, 1] - box2[:, 3] / 2
    b2_x2 = box2[:, 0] + box2[:, 2] / 2
    b2_y2 = box2[:, 1] + box2[:, 3] / 2

    # intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # union
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


# -----------------------------------------------------------
# QFL (Quality Focal Loss) - Fully Vectorized
# -----------------------------------------------------------
def quality_focal_loss(pred_scores, target_scores, beta=2.0):
    """
    pred_scores: (M, C)
    target_scores: (M, C)
    """
    pred_sigmoid = pred_scores.sigmoid()

    pos_term = target_scores * (1 - pred_sigmoid).pow(beta) * torch.log(pred_sigmoid + 1e-12)
    neg_term = (1 - target_scores) * (pred_sigmoid.pow(beta)) * torch.log(1 - pred_sigmoid + 1e-12)

    loss = -(pos_term + neg_term).sum() / pred_scores.size(0)
    return loss


# -----------------------------------------------------------
# Smooth L1 Regression Loss (DFL approximation)
# -----------------------------------------------------------
def distribution_focal_loss(pred_box, target_box):
    """
    pred_box: (M, 4)
    target_box: (M, 4)
    """
    return F.smooth_l1_loss(pred_box, target_box, reduction="mean")


# -----------------------------------------------------------
# MAIN LOSS (Batch Supported, Vectorized per image)
# -----------------------------------------------------------
class YoloDFLQFLoss(nn.Module):
    def __init__(self, num_classes=171, lambda_box=1.5, lambda_cls=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls

    def forward(self, preds, gt_boxes_list):
        """
        preds: (N, 175, 8400)
        gt_boxes_list: list of length N, each → (Mi, 5)
        """
        N = preds.shape[0]
        # Cast to float32 for numerical stability (prevents inf/nan in log/exp with float16)
        preds = preds.float().transpose(1, 2)  # (N, 8400, 175)

        total_box = 0.0
        total_cls = 0.0
        used_images = 0

        for b in range(N):

            pred = preds[b]                                 # (8400, 175)
            pred_box = pred[:, :4]                          # (8400, 4)
            pred_scores = pred[:, 4:]                       # (8400, C)

            gt_boxes = gt_boxes_list[b]                     # (M, 5)
            if gt_boxes.numel() == 0:
                continue

            M = gt_boxes.size(0)
            gt_xywh = gt_boxes[:, 0:4].to(preds.dtype)

            # ---------------------------------------------------
            # VECTORIZED MATCHING (NO LOOPS)
            # Compute distances between all GT centers and 8400 predictions
            # ---------------------------------------------------
            gt_centers = gt_xywh[:, 0:2]                    # (M, 2)
            pred_centers = pred_box[:, 0:2]                 # (8400, 2)

            dist = torch.cdist(gt_centers, pred_centers)    # (M, 8400)
            idx = dist.argmin(dim=1)                        # (M,)

            # Gather matched predictions
            matched_pred_box = pred_box[idx]                # (M, 4)
            matched_pred_scores = pred_scores[idx]          # (M, C)

            # ---------------------------------------------------
            # DFL (smooth L1)
            # ---------------------------------------------------
            box_loss = distribution_focal_loss(matched_pred_box, gt_xywh)

            # ---------------------------------------------------
            # QFL: Compute IoU once for every matched GT
            # ---------------------------------------------------
            iou = bbox_iou(matched_pred_box, gt_xywh).unsqueeze(1)  # (M, 1)

            # Build target classification matrix (vectorized)
            target_scores = torch.zeros_like(matched_pred_scores)    # (M, C)
            class_ids = gt_boxes[:, 4].long().unsqueeze(1)           # (M, 1)

            target_scores.scatter_(1, class_ids, iou.to(target_scores.dtype))                # fill IoU at GT class

            cls_loss = quality_focal_loss(matched_pred_scores, target_scores)

            # ---------------------------------------------------
            # Accumulate loss
            # ---------------------------------------------------
            total_box += box_loss
            total_cls += cls_loss
            used_images += 1

        if used_images == 0:
            return torch.tensor(0.0, device=preds.device), {}

        mean_box = total_box / used_images
        mean_cls = total_cls / used_images

        total_loss = self.lambda_box * mean_box + self.lambda_cls * mean_cls

        return total_loss, {
            "total_loss": total_loss.detach().item(),
            "box_loss": mean_box.detach().item(),
            "cls_loss": mean_cls.detach().item()
        }
