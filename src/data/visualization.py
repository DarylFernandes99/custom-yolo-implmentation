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

    fig, ax = plt.subplots(1, figsize=(10, 10))
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
                bbox=dict(facecolor=np.array(color) / 255.0, alpha=0.5, pad=1)
            )

    ax.set_title(title)
    ax.axis("off")
    plt.show()

def visualize_comparison(image, target, prediction=None):
    """
    Visualize image, ground truth boxes, and predicted boxes side-by-side.
    
    Args:
        image: Image tensor data.
        target: Target data.
        prediction (dict, optional): Model output with keys:
            - 'boxes': tensor of shape [N, 4]
            - 'labels': tensor of shape [N]
            - 'scores': tensor of shape [N]
    """
    print(target["boxes"])
    boxes_gt = target["boxes"].cpu().numpy()
    labels_gt = target["labels"].cpu().numpy()

    # Convert prediction tensors to numpy
    boxes_pred, labels_pred, scores_pred = None, None, None
    if prediction is not None:
        boxes_pred = prediction["boxes"].detach().cpu().numpy()
        labels_pred = prediction["labels"].detach().cpu().numpy()
        scores_pred = prediction["scores"].detach().cpu().numpy()

    # Plot raw image
    show_image(image, title="Raw Image")

    # Plot image with GT boxes
    draw_bboxes(image, boxes_gt, labels=labels_gt, title="Ground Truth Boxes", color=(0, 255, 0))

    # Plot image with predicted boxes (if provided)
    if prediction is not None:
        draw_bboxes(image, boxes_pred, labels=labels_pred, scores=scores_pred,
                    title="Predicted Boxes", color=(255, 0, 0))
