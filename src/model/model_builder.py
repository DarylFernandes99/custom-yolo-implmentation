import torch
from torch import nn
from PIL import Image
from typing import List

from src.model.neck import Neck
from src.model.head import Head
from src.model.model_blocks import Conv
from src.model.backbone import Backbone
from src.data.transforms import get_val_transforms
from src.utils.model_utils import fuse_conv, dist2bbox, non_max_suppression

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

    def load_weights(self, weights_path):
        """
        Load model weights from a checkpoint file.
        
        Parameters
        ----------
        weights_path : str
            Path to the checkpoint file.
        """
        checkpoint = torch.load(weights_path, map_location=next(self.parameters()).device)
        
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
            
        self.load_state_dict(state_dict)
        print(f"Weights loaded successfully from {weights_path}")

    def inference(self, image, conf_thres=0.25, iou_thres=0.45):
        """
        Run inference on a single image.
        
        Parameters
        ----------
        image : str, PIL.Image.Image, or torch.Tensor
            Input image. Can be a file path, a PIL Image, or a tensor.
        conf_thres : float
            Confidence threshold for NMS.
        iou_thres : float
            IoU threshold for NMS.
            
        Returns
        -------
        list
            List of detections [xyxy, conf, cls]
        """
        self.eval()
        
        # Handle input
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        if isinstance(image, Image.Image):
            transform = get_val_transforms()
            image_tensor = transform(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            image_tensor = image
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
        else:
            raise ValueError("Unsupported image type. Must be path, PIL Image, or Tensor.")
            
        image_tensor = image_tensor.to(next(self.parameters()).device)
        
        with torch.no_grad():
            # Forward pass
            preds = self.forward(image_tensor) # preds is (x, anchors, strides)
            
            x, anchors, strides = preds
            
            # Split box and cls
            # x is (N, no, M) where no = nc + ch*4
            box, cls = x.split((self.head.ch * 4, self.head.nc), 1)
            
            # Decode box with DFL
            if self.head.dfl is not None:
                 box = self.head.dfl(box) # Returns (N, 4, M)
            
            # Convert to bbox
            box = dist2bbox(box, anchors.unsqueeze(0), xywh=True, dim=1) # (N, 4, M)
            
            # Scale by stride
            box = box * strides
            
            # Concatenate
            y = torch.cat((box, cls), 1)
            
            # NMS
            return non_max_suppression(y, conf_thres=conf_thres, iou_thres=iou_thres, nc=self.num_classes)
