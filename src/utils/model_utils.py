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
