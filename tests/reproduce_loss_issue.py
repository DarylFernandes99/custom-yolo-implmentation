import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.losses import YoloDFLQFLoss

def test_loss_ignores_background():
    # Setup
    num_classes = 2
    reg_max = 16
    loss_fn = YoloDFLQFLoss(num_classes=num_classes, reg_max=reg_max)
    
    N = 1
    num_anchors = 10
    # Channels: 4 * reg_max (box dist) + num_classes (scores)
    channels = 4 * reg_max + num_classes
    
    # Create dummy anchors and strides
    # Anchors: (2, num_anchors)
    # Strides: (1, num_anchors)
    anchors = torch.zeros(2, num_anchors)
    anchors[0, :] = 5.0 # x centers
    anchors[1, :] = 5.0 # y centers
    
    strides = torch.ones(1, num_anchors) * 32.0 # Stride 32
    
    # Create a prediction tensor
    preds = torch.zeros(N, channels, num_anchors, requires_grad=True)
    
    # Create a dummy GT
    # (M, 5) -> [x, y, w, h, class_id]
    # GT in PIXEL coords
    # Anchor at 5,5 (grid). Stride 32. Pixel center = 160, 160.
    # Let's put GT at 160, 160 with size 64x64.
    gt_boxes = torch.tensor([[160.0, 160.0, 64.0, 64.0, 0.0]]) 
    gt_boxes_list = [gt_boxes]
    
    # Calculate initial loss
    loss1, _ = loss_fn(preds, gt_boxes_list, anchors, strides)
    
    print(f"Initial Loss: {loss1.item()}")
    
    # Modify prediction for an anchor (Anchor 0) to match GT
    # We want predicted l,t,r,b to match GT relative to anchor.
    # GT grid coords: 160/32 = 5.0. Size 64/32 = 2.0.
    # x1=4, y1=4, x2=6, y2=6.
    # Anchor at 5,5.
    # l = 5-4 = 1
    # t = 5-4 = 1
    # r = 6-5 = 1
    # b = 6-5 = 1
    # So we want expectation of distribution to be 1.0 for all 4 coords.
    
    with torch.no_grad():
        # Set box distribution for Anchor 0
        # x (l): index 1
        preds[0, 1, 0] = 10.0
        # y (t): index 1 (+reg_max)
        preds[0, reg_max + 1, 0] = 10.0
        # w (r): index 1 (+2*reg_max)
        preds[0, 2*reg_max + 1, 0] = 10.0
        # h (b): index 1 (+3*reg_max)
        preds[0, 3*reg_max + 1, 0] = 10.0
        
        # Set class score for Anchor 0 (Class 0)
        preds[0, 4*reg_max + 0, 0] = 5.0
    
    # Calculate loss again
    loss2, _ = loss_fn(preds, gt_boxes_list, anchors, strides)
    print(f"Loss with matched anchor: {loss2.item()}")
    
    # Now, drastically change the CLASS prediction of the UNMATCHED anchor (Anchor 1).
    with torch.no_grad():
        # Increase score for class 0 on anchor 1
        preds[0, 4*reg_max + 0, 1] = 10.0 
    
    loss3, _ = loss_fn(preds, gt_boxes_list, anchors, strides)
    print(f"Loss after spiking background anchor score: {loss3.item()}")
    
    if abs(loss3.item() - loss2.item()) < 1e-6:
        print("FAIL: Loss did not change! Background anchors are ignored.")
    else:
        print("PASS: Loss changed. Background anchors are included.")

if __name__ == "__main__":
    test_loss_ignores_background()
