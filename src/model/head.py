import math
import torch
from torch import nn
from typing import List

from src.model.model_blocks import Conv, DFL
from src.utils.model_utils import make_anchors

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
                Conv(x, x, nn.SiLU(), k=3, p=1, g=x),
                Conv(x, cls, nn.SiLU()),
                Conv(cls, cls, nn.SiLU(), k=3, p=1, g=cls),
                Conv(cls, cls, nn.SiLU()),
                nn.Conv2d(cls, out_channels=self.nc, kernel_size=1)
            ) for x in filters
        )

        self.initialize_biases()

    def initialize_biases(self):
        # Initialize biases for classification to prevent instability at start
        prior_prob = 1e-2
        bias_value = math.log(prior_prob / (1 - prior_prob))
        
        for module in self.cls:
            last_layer = module[-1]
            if isinstance(last_layer, nn.Conv2d):
                nn.init.constant_(last_layer.bias, bias_value)

    
    def forward(self, x):
        """
        Forward pass of the Head module.
        
        Returns:
            x (torch.Tensor): Raw predictions (N, 4*reg_max + nc, 8400)
            anchors (torch.Tensor): Anchor points (8400, 2)
            strides (torch.Tensor): Strides for each anchor (8400, 1)
        """
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
            
        # Compute anchors and strides (needed for both training loss and inference decoding)
        # Note: make_anchors returns (anchors, strides)
        # We use a dummy forward pass or just compute it based on input shapes
        # The `make_anchors` function uses the feature map shapes.
        
        anchors, strides = make_anchors(x, self.stride, 0.5)
        anchors, strides = anchors.transpose(0, 1), strides.transpose(0, 1) # (2, 8400), (1, 8400) -> Wait, make_anchors returns (M, 2) and (M, 1) usually?
        
        # Let's check make_anchors implementation in this file:
        # return torch.cat(anchor_tensor), torch.cat(stride_tensor)
        # anchor_tensor list of (H*W, 2) -> cat -> (M, 2)
        # stride_tensor list of (H*W, 1) -> cat -> (M, 1)
        
        # The original code did:
        # self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        # So it transposed them to (2, M) and (1, M)? 
        # Let's check usage: 
        # a = self.anchors.unsqueeze(0) - a  -> (1, 2, M) - (N, 4, M)? No.
        # a, b = self.dfl(box).chunk(2, 1) -> box is (N, 4*ch, M) -> dfl -> (N, 4, M) -> chunk -> (N, 2, M)
        # So anchors should be (2, M) to broadcast with (N, 2, M).
        
        # So yes, transpose is needed if we want (2, M).
        
        anchors, strides = make_anchors(x, self.stride, 0.5)
        anchors = anchors.transpose(0, 1) # (2, M)
        strides = strides.transpose(0, 1) # (1, M)
        
        # Flatten predictions
        # x is list of (N, C, H, W)
        # view to (N, C, -1) and cat dim 2 -> (N, C, M)
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        
        return x, anchors, strides
