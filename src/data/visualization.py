import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_image(image_tensor, title="Image"):
    """
    Display a single image tensor using matplotlib.
    image_tensor: torch.Tensor [C, H, W] or numpy array [H, W, C]
    """
    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        image = image_tensor

    # Convert to uint8 for display if normalized
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()

def draw_bboxes(image_tensor, boxes, labels=None, scores=None, title="Image + Boxes", color=(0, 255, 0)):
    """
    Draw bounding boxes on an image tensor or numpy array.
    
    Args:
        image_tensor (torch.Tensor or np.array): Image tensor [C,H,W] or numpy [H,W,C]
        boxes (Tensor or List): Bounding boxes [[x, y, w, h], ...]
        labels (List[int], optional): Class labels for each box.
        scores (List[float], optional): Confidence scores for predictions.
        title (str): Plot title.
        color (tuple): BGR or RGB color for boxes.
    """
    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        image = image_tensor.copy()

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    _, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    for i, box in enumerate(boxes):
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=np.array(color) / 255.0,
            facecolor='none'
        )
        ax.add_patch(rect)

        # Label text (class or score)
        label_text = ""
        if labels is not None:
            label_text += f"{labels[i]}"
        if scores is not None:
            label_text += f" ({scores[i]:.2f})"

        if label_text:
            ax.text(
                x, y - 5, label_text,
                color='white', fontsize=8,
                bbox={'facecolor': np.array(color) / 255.0, 'alpha': 0.5, 'pad': 1}
            )

    ax.set_title(title)
    ax.axis("off")
    plt.show()

def visualize_comparison(image, target, prediction=None, figsize=(16, 8)):
    """
    Visualize original image and ground truth boxes side-by-side using subplots.
    Optionally includes predicted boxes in a third panel.
    
    Args:
        image (torch.Tensor or np.array): Image tensor [C, H, W] or numpy [H, W, C]
        target (dict): Target data with 'boxes' tensor of shape [N, 5] where columns are [x, y, w, h, class_id]
        prediction (dict, optional): Model output with keys:
            - 'boxes': tensor of shape [N, 5] where columns are [x, y, w, h, class_id]
            - 'scores': tensor of shape [N] (optional)
        figsize (tuple): Figure size for the plot.
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = image.copy()
    
    # Normalize to uint8 for display
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Split boxes and classes from target
    boxes_with_classes = target["boxes"].cpu().numpy()
    boxes_gt = boxes_with_classes[:, :4]  # [x, y, w, h]
    labels_gt = boxes_with_classes[:, 4].astype(int)  # [class_id]

    # Determine number of subplots
    n_plots = 2 if prediction is None else 3
    
    _, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Panel 1: Original Image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis("off")
    
    # Panel 2: Ground Truth Boxes
    axes[1].imshow(img_np)
    for i, box in enumerate(boxes_gt):
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor='lime',
            facecolor='none'
        )
        axes[1].add_patch(rect)
        
        # Add label
        label_text = f"{labels_gt[i]}"
        axes[1].text(
            x, y - 5, label_text,
            color='white', fontsize=8,
            bbox={'facecolor': 'lime', 'alpha': 0.7, 'pad': 1, 'edgecolor': 'none'}
        )
    
    axes[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axes[1].axis("off")
    
    # Panel 3: Predicted Boxes (if provided)
    if prediction is not None:
        boxes_with_classes_pred = prediction["boxes"].detach().cpu().numpy()
        boxes_pred = boxes_with_classes_pred[:, :4]  # [x, y, w, h]
        labels_pred = boxes_with_classes_pred[:, 4].astype(int)  # [class_id]
        
        scores_pred = None
        if "scores" in prediction:
            scores_pred = prediction["scores"].detach().cpu().numpy()
        
        axes[2].imshow(img_np)
        for i, box in enumerate(boxes_pred):
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            axes[2].add_patch(rect)
            
            # Add label with score
            label_text = f"{labels_pred[i]}"
            if scores_pred is not None:
                label_text += f" ({scores_pred[i]:.2f})"
            axes[2].text(
                x, y - 5, label_text,
                color='white', fontsize=8,
                bbox={'facecolor': 'red', 'alpha': 0.7, 'pad': 1, 'edgecolor': 'none'}
            )
        
        axes[2].set_title("Predictions", fontsize=14, fontweight='bold')
        axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()
