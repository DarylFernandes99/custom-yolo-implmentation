import math
import torch
from torch import nn

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
