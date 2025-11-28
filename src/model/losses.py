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
# Distribution Focal Loss (DFL)
# -----------------------------------------------------------
def distribution_focal_loss(pred_dist, target_val):
    """
    pred_dist: (M, reg_max + 1) - predicted distribution logits
    target_val: (M,) - continuous target value
    """
    # Discretize target
    dis_left = target_val.long()
    dis_right = dis_left + 1
    
    weight_left = dis_right.float() - target_val
    weight_right = target_val - dis_left.float()
    
    loss = F.cross_entropy(pred_dist, dis_left, reduction='none') * weight_left + \
           F.cross_entropy(pred_dist, dis_right, reduction='none') * weight_right
           
    return loss.mean()


# -----------------------------------------------------------
# MAIN LOSS (Batch Supported, Vectorized per image)
# -----------------------------------------------------------
class YoloDFLQFLoss(nn.Module):
    def __init__(self, num_classes=171, lambda_box=1.5, lambda_cls=1.0, lambda_dfl=1.5, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_dfl = lambda_dfl
        self.reg_max = reg_max

    def forward(self, preds, gt_boxes_list, anchors, strides):
        """
        preds: (N, 4*reg_max + num_classes, 8400)
        gt_boxes_list: list of length N, each → (Mi, 5)
        anchors: (2, 8400) - anchor points in feature map coordinates (or image coords?)
                 Wait, Head.make_anchors returns anchors in feature map coords + offset.
                 But then multiplies by stride for final output.
                 In Head.forward, we return:
                 anchors, strides = make_anchors(x, self.stride, 0.5)
                 anchors = anchors.transpose(0, 1)
                 
                 make_anchors returns anchors in GRID coordinates (feature map).
                 strides contains the stride for each anchor.
                 
                 So `anchors` passed here are in GRID coordinates.
                 `strides` are the scaling factors.
                 
                 GT boxes are in PIXEL coordinates (xywh).
                 
                 To compute DFL targets, we need to convert GT to distance from anchor in GRID units.
                 
                 1. Convert GT to GRID coordinates: gt_grid = gt_pixel / stride
                 2. Convert GT grid center to distance from anchor: 
                    l = anchor_x - gt_x1_grid
                    t = anchor_y - gt_y1_grid
                    r = gt_x2_grid - anchor_x
                    b = gt_y2_grid - anchor_y
                    
                    Wait, DFL target is usually l, t, r, b.
                    
                    Let's check standard DFL target definition.
                    Target is distance from anchor center to box sides.
                    
                    gt_cx, gt_cy, gt_w, gt_h = gt_grid
                    gt_x1 = gt_cx - gt_w/2
                    gt_y1 = gt_cy - gt_h/2
                    gt_x2 = gt_cx + gt_w/2
                    gt_y2 = gt_cy + gt_h/2
                    
                    target_l = anchor_x - gt_x1
                    target_t = anchor_y - gt_y1
                    target_r = gt_x2 - anchor_x
                    target_b = gt_y2 - anchor_y
                    
                    These targets should be positive and within [0, reg_max].
                    
        """
        N = preds.shape[0]
        # Cast to float32 for numerical stability
        preds = preds.float().transpose(1, 2)  # (N, 8400, C_total)
        
        # Anchors and strides are (2, 8400) and (1, 8400)
        # Transpose to (8400, 2) and (8400, 1) for easier broadcasting with flattened preds
        anchors = anchors.transpose(0, 1).to(preds.device)
        strides = strides.transpose(0, 1).to(preds.device)
        
        # Split predictions
        box_channels = 4 * self.reg_max
        pred_dist_flat = preds[:, :, :box_channels]  # (N, 8400, 4*reg_max)
        pred_scores = preds[:, :, box_channels:]     # (N, 8400, num_classes)
        
        # Decode box distribution to get predicted boxes (l, t, r, b) in GRID units
        B, A, _ = pred_dist_flat.shape
        pred_dist = pred_dist_flat.view(B, A, 4, self.reg_max)
        pred_dist_softmax = pred_dist.softmax(3)
        values = torch.arange(self.reg_max, device=preds.device, dtype=preds.dtype)
        pred_ltrb = torch.sum(pred_dist_softmax * values, dim=3) # (N, 8400, 4) - l,t,r,b
        
        # Convert pred_ltrb to pred_xywh (in PIXEL coords) for IoU calculation
        # x1 = (anchor_x - l) * stride
        # y1 = (anchor_y - t) * stride
        # x2 = (anchor_x + r) * stride
        # y2 = (anchor_y + b) * stride
        
        # anchors is (8400, 2) -> (1, 8400, 2)
        # strides is (8400, 1) -> (1, 8400, 1)
        
        anchors_exp = anchors.unsqueeze(0)
        strides_exp = strides.unsqueeze(0)
        
        pred_l = pred_ltrb[:, :, 0]
        pred_t = pred_ltrb[:, :, 1]
        pred_r = pred_ltrb[:, :, 2]
        pred_b = pred_ltrb[:, :, 3]
        
        pred_x1 = (anchors_exp[:, :, 0] - pred_l) * strides_exp[:, :, 0]
        pred_y1 = (anchors_exp[:, :, 1] - pred_t) * strides_exp[:, :, 0]
        pred_x2 = (anchors_exp[:, :, 0] + pred_r) * strides_exp[:, :, 0]
        pred_y2 = (anchors_exp[:, :, 1] + pred_b) * strides_exp[:, :, 0]
        
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1
        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        
        pred_xywh = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=2) # (N, 8400, 4)
        
        total_box = 0.0
        total_cls = 0.0
        total_dfl = 0.0
        used_images = 0

        for b in range(N):
            # Per image
            p_dist = pred_dist[b]       # (8400, 4, 16)
            p_xywh = pred_xywh[b]       # (8400, 4) - pixel coords
            p_scores = pred_scores[b]   # (8400, C)
            
            gt_boxes = gt_boxes_list[b] # (M, 5)
            
            # Initialize target scores (background = 0)
            target_scores = torch.zeros_like(p_scores)
            
            if gt_boxes.numel() > 0:
                M = gt_boxes.size(0)
                gt_xywh = gt_boxes[:, 0:4].to(preds.dtype) # Pixel coords
                
                # Matching using PIXEL coords
                gt_centers = gt_xywh[:, 0:2]
                pred_centers = p_xywh[:, 0:2]
                
                dist = torch.cdist(gt_centers, pred_centers)
                idx = dist.argmin(dim=1)
                
                # Matched predictions
                matched_p_xywh = p_xywh[idx]        # (M, 4)
                matched_p_dist = p_dist[idx]        # (M, 4, 16)
                matched_anchors = anchors[idx]      # (M, 2)
                matched_strides = strides[idx]      # (M, 1)
                
                # 1. DFL Loss
                # Convert GT to l,t,r,b in GRID units
                # gt_xywh is pixel.
                gt_x1 = gt_xywh[:, 0] - gt_xywh[:, 2] / 2
                gt_y1 = gt_xywh[:, 1] - gt_xywh[:, 3] / 2
                gt_x2 = gt_xywh[:, 0] + gt_xywh[:, 2] / 2
                gt_y2 = gt_xywh[:, 1] + gt_xywh[:, 3] / 2
                
                # Normalize to grid
                gt_x1_grid = gt_x1 / matched_strides[:, 0]
                gt_y1_grid = gt_y1 / matched_strides[:, 0]
                gt_x2_grid = gt_x2 / matched_strides[:, 0]
                gt_y2_grid = gt_y2 / matched_strides[:, 0]
                
                # Calculate target l,t,r,b
                target_l = matched_anchors[:, 0] - gt_x1_grid
                target_t = matched_anchors[:, 1] - gt_y1_grid
                target_r = gt_x2_grid - matched_anchors[:, 0]
                target_b = gt_y2_grid - matched_anchors[:, 1]
                
                target_ltrb = torch.stack([target_l, target_t, target_r, target_b], dim=1)
                
                # Clamp targets to [0, reg_max - 1 - 0.01]
                target_ltrb = target_ltrb.clamp(0, self.reg_max - 1 - 0.01)
                
                # Compute DFL loss
                dfl_loss_val = 0.0
                for i in range(4):
                    dfl_loss_val += distribution_focal_loss(matched_p_dist[:, i, :], target_ltrb[:, i])
                dfl_loss_val = dfl_loss_val / 4.0
                total_dfl += dfl_loss_val
                
                # 2. QFL Targets
                iou = bbox_iou(matched_p_xywh, gt_xywh).unsqueeze(1)
                class_ids = gt_boxes[:, 4].long()
                
                matched_targets = torch.zeros(M, self.num_classes, device=preds.device, dtype=preds.dtype)
                matched_targets.scatter_(1, class_ids.unsqueeze(1), iou.to(preds.dtype))
                target_scores[idx] = matched_targets
                
            # QFL Loss (All anchors)
            cls_loss = quality_focal_loss(p_scores, target_scores)
            total_cls += cls_loss
            used_images += 1

        if used_images == 0:
            return torch.tensor(0.0, device=preds.device), {}

        mean_dfl = total_dfl / used_images
        mean_cls = total_cls / used_images
        
        # Total loss
        total_loss = self.lambda_dfl * mean_dfl + self.lambda_cls * mean_cls
        
        return total_loss, {
            "total_loss": total_loss.detach().item(),
            "box_loss": mean_dfl.detach().item(), # Reporting DFL as box loss
            "cls_loss": mean_cls.detach().item()
        }

