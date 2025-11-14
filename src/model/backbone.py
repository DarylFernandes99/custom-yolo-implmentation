import torch
from torch import nn
from typing import List

from src.model.model_blocks import Conv, C3K2, SPPF, PSA

class Backbone(nn.Module):
    def __init__(self, width: List[int], depth: List[int], csp: List[bool]):
        """
        The Backbone is a deep Convolutional Neural Network (CNN)
        that processes the input image to extract fundamental
        visual features across various scales. It acts as the
        primary feature extractor for the entire model,
        transforming raw pixels into a set of informative
        feature maps.

        Parameters:
        -----------
        width : List[int]
            A list of integers representing the number of channels
            for each layer.
        depth : List[int]
            A list of integers representing the depth of each CSP
            block.
        csp : List[bool]
            A list of Boolean values representing whether to use
            CSP (Cross Stage Partial) module for each layer.
        """

        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], nn.SiLU(), k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], nn.SiLU(), k=3, s=2, p=1))
        self.p2.append(C3K2(width[2], width[3], depth[0], csp[0], r=4))
        # p3/8
        self.p3.append(Conv(width[3], width[3], nn.SiLU(), k=3, s=2, p=1))
        self.p3.append(C3K2(width[3], width[4], depth[1], csp[0], r=4))
        # p4/16
        self.p4.append(Conv(width[4], width[4], nn.SiLU(), k=3, s=2, p=1))
        self.p4.append(C3K2(width[4], width[4], depth[2], csp[1], r=2))
        # p5/32
        self.p5.append(Conv(width[4], width[5], nn.SiLU(), k=3, s=2, p=1))
        self.p5.append(C3K2(width[5], width[5], depth[3], csp[1], r=2))
        self.p5.append(SPPF(width[5], width[5]))
        self.p5.append(PSA(width[5], depth[4]))

        self.p1 = nn.Sequential(*self.p1)
        self.p2 = nn.Sequential(*self.p2)
        self.p3 = nn.Sequential(*self.p3)
        self.p4 = nn.Sequential(*self.p4)
        self.p5 = nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5
