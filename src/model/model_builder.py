import torch
from torch import nn
from typing import List

from src.model.neck import Neck
from src.model.head import Head
from src.model.model_blocks import Conv
from src.model.backbone import Backbone
from src.utils.model_utils import fuse_conv

class Model(nn.Module):
    def __init__(self, width: List[int], depth: List[int], csp: List[bool], num_classes: int):
        """
        ModelBuilder class that composes a Backbone, Neck (FPN), and Head to form a full model.

        Parameters
        ----------
        width : List[int]
            Channel widths for each stage of the network. width[0] is interpreted as the input channel
            count (used to construct a dummy input), and width[3], width[4], width[5] are used to
            construct the head's input channels.
        depth : List[int]
            Depth (number of blocks) for each stage of the backbone/neck. The list length must match
            the expected architecture configuration.
        csp : List[bool]
            Boolean flags indicating whether to use CSP-style blocks for corresponding stages.
        num_classes : int
            Number of target classes that the head will predict.
        """
        super().__init__()
        self.net = Backbone(width, depth, csp)
        self.fpn = Neck(width, depth, csp)
        self.num_classes = num_classes

        img_dummy = torch.zeros(1, width[0], 640, 640)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        
        with torch.no_grad():
            backbone_out = self.net(img_dummy)
            neck_out = self.fpn(backbone_out)
            self.head.stride = torch.tensor([img_dummy.shape[-1] / x.shape[-2] for x in neck_out])
        
        self.stride = self.head.stride

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
