import math
import torch
import time
import torchvision
import numpy as np
from torch import nn
from typing import List

def autopad(k, p=None, d=1):
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def make_anchors(x: List[torch.tensor], strides: List[torch.tensor], offset: float = 0.5):
    """
    Create a grid of anchor reference points and a corresponding per-anchor stride tensor
    from a list of feature-map tensors.
    The function computes 2D anchor centers (x, y) for each spatial location of each
    feature map in `x`. Each anchor is represented in feature-map (cell) coordinates
    (i.e., integer grid coordinates plus `offset`) and is paired with its feature-map
    stride value. The returned dtype and device match those of the first tensor in `x`.
    
    Parameters
    ----------
    x : List[torch.Tensor]
        List of feature-map tensors. Each tensor is expected
        to have shape (N, C, H, W) or otherwise expose a (H, W) spatial shape via
        its .shape attribute at indices [-2:] (the code uses x[i].shape => _, _, H, W).
        All tensors should be on the same device and have the same dtype; the returned
        tensors will use x[0].device and x[0].dtype.
    strides : List[int]
        List of stride values (one per feature-map) that indicate the effective
        downsampling (number of input pixels per feature-map cell) for each feature
        map. The length of `strides` must match the length of `x`.
    offset : float, optional
        Sub-cell offset added to the integer grid coordinates when computing anchor
        centers. Default is 0.5 (centers of cells). Can be any float to shift the
        anchor reference point within each cell.
    
    Returns
    -------
    anchors : torch.Tensor
        Float tensor of shape (M, 2) containing anchor center coordinates in
        feature-map (cell) units as (x, y) pairs. M is the total number of spatial
        locations across all feature maps, i.e. sum_i (H_i * W_i). The ordering
        is such that x (column) index varies fastest within each feature map.
    strides_out : torch.Tensor
        Float tensor of shape (M, 1) containing the stride value repeated for each
        corresponding anchor. This can be used to scale anchors to image/pixel
        coordinates by multiplying anchors * strides_out.
    """
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device

    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        # shift x
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset
        # shift y
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)

def fuse_conv(conv: nn.Conv2d, norm: nn.Module):
    """
    Fuse a 2D convolution and a subsequent batch normalization layer into a single Conv2d.
    This function creates a new torch.nn.Conv2d whose weights and bias incorporate the
    effect of the provided normalization layer so that applying the fused Conv2d alone
    is equivalent (for inference) to applying conv followed by norm. The returned
    module has bias=True, requires_grad=False and is placed on the same device as the
    original conv weights.
    
    Parameters
    ----------
    conv : torch.nn.Conv2d
        The convolution layer to fuse. Its in_channels, out_channels, kernel_size,
        stride, padding and groups attributes are preserved. If conv.bias is None
        the convolutional bias term is treated as zero during fusion.
    norm : torch.nn.modules.batchnorm._BatchNorm
        The normalization layer to fuse (e.g., torch.nn.BatchNorm2d). Must expose
        weight (gamma), bias (beta), running_mean, running_var and eps. The fusion
        uses these statistics and affine parameters to fold normalization into the
        convolution weights and bias.
    
    Returns
    -------
    torch.nn.Conv2d
        A new Conv2d instance equivalent to conv followed by norm for inference.
        The fused convolution has bias enabled, gradients disabled (requires_grad=False),
        and resides on the same device as the original conv weights.
    """
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[2:4] < min_wh) | (x[2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(0, -1)[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls = x.split((4, nc), 1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), x[i, 4 + nc:]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), x[:, 4 + nc:]), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output
