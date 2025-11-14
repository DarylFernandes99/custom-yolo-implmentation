import torch
from torch import nn
from typing import List

from src.model.model_blocks import Conv, DFL

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

class Head(nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc: int = 1, filters: List[int] = []):
        """
        The Head is the final prediction layer that interprets
        the refined features from the neck to produce the final
        output. This component generates the bounding box
        coordinates, confidence scores, and class probabilities
        for all detected objects within the image.

        Parameters
        ----------
        nc : int, optional
            Number of classes for detection (default: 1). This
            determines the final classification output channels
            of the head modules.
        filters : List[int], optional
            Number of input channels for each detection layer
            (one entry per detection layer). Each element
            configures the channel size expected by the
            corresponding box and class submodules.
            Default is an empty tuple.
        """
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        
        self.box = nn.ModuleList(
            nn.Sequential(
                Conv(x, box, nn.SiLU(), k=3, p=1),
                Conv(box, box, nn.SiLU(), k=3, p=1),
                nn.Conv2d(box, out_channels=4 * self.ch, kernel_size=1)
            ) for x in filters
        )
        
        self.cls = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, nn.SiLU(), k=3, p=1, g=x),
                Conv(x, cls, nn.SiLU()),
                Conv(cls, cls, nn.SiLU(), k=3, p=1, g=cls),
                Conv(cls, cls, nn.SiLU()),
                nn.Conv2d(cls, out_channels=self.nc,kernel_size=1)
            )) for x in filters
        )
    
    def forward(self, x):
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)
