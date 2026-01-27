"""
Inference module for YOLO models.

This module provides functions for running inference on images using trained YOLO models.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor

from yolo.config.config import NMSConfig
from yolo.data.transforms import LetterBox
from yolo.model.yolo import YOLO
from yolo.tools.drawer import draw_bboxes
from yolo.training.module import YOLOModule
from yolo.utils.bounding_box_utils import Vec2Box, bbox_nms


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Tuple[YOLO, dict, str]:
    """
    Load a YOLO model from a Lightning checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file
        device: Device to load model on (auto-detected if None)

    Returns:
        Tuple of (model, model_config, device)
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load the Lightning module from checkpoint
    # Skip pretrained weight loading since checkpoint already has trained weights
    # Use strict=False to ignore EMA keys (ema.*) and loss function keys (_loss_fn.*)
    module = YOLOModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        weight_path=None,  # Don't load pretrained weights, use checkpoint state_dict
        strict=False,  # Ignore EMA and loss function keys
    )
    module.eval()
    module.to(device)

    return module.model, module._model_cfg, device


def preprocess_image(
    image_path: str,
    image_size: Tuple[int, int] = (640, 640),
) -> Tuple[Tensor, Image.Image]:
    """
    Load and preprocess an image for inference.

    Args:
        image_path: Path to the image file
        image_size: Target size (width, height)

    Returns:
        Tuple of (preprocessed tensor, original PIL image)
    """
    image = Image.open(image_path).convert("RGB")
    letterbox = LetterBox(target_size=image_size)
    target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,))}
    image_tensor, _ = letterbox(image, target)
    return image_tensor.unsqueeze(0), image


def run_inference(
    model: YOLO,
    model_cfg: dict,
    image_tensor: Tensor,
    device: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.65,
    max_detections: int = 300,
) -> List[Tensor]:
    """
    Run inference on a preprocessed image tensor.

    Args:
        model: YOLO model
        model_cfg: Model configuration (with anchor info)
        image_tensor: Preprocessed image tensor [1, 3, H, W]
        device: Device string
        conf_threshold: Confidence threshold for NMS
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image

    Returns:
        List of prediction tensors, one per image in batch.
        Each tensor has shape [N, 6] with columns [class_id, x1, y1, x2, y2, confidence]
    """
    image_tensor = image_tensor.to(device)
    image_size = list(image_tensor.shape[-2:])

    # Create converter
    vec2box = Vec2Box(model, model_cfg.anchor, image_size, device)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_cls, pred_anc, pred_box = vec2box(outputs["Main"])

    # NMS
    nms_cfg = NMSConfig(
        min_confidence=conf_threshold,
        min_iou=iou_threshold,
        max_bbox=max_detections,
    )
    predictions = bbox_nms(pred_cls, pred_box, nms_cfg)

    return predictions


def scale_boxes_to_original(
    boxes: Tensor,
    original_size: Tuple[int, int],
    letterbox_size: Tuple[int, int],
) -> Tensor:
    """
    Scale bounding boxes from letterboxed image space to original image space.

    Args:
        boxes: Tensor of shape [N, 6] with [class_id, x1, y1, x2, y2, conf]
        original_size: Original image size (width, height)
        letterbox_size: Letterboxed image size (width, height)

    Returns:
        Scaled boxes tensor
    """
    if len(boxes) == 0:
        return boxes

    orig_w, orig_h = original_size
    lb_w, lb_h = letterbox_size

    # Calculate scale and padding used in letterbox
    scale = min(lb_w / orig_w, lb_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (lb_w - new_w) // 2
    pad_y = (lb_h - new_h) // 2

    # Clone boxes to avoid modifying original
    scaled_boxes = boxes.clone()

    # Remove padding offset and scale back to original size
    scaled_boxes[:, 1] = (boxes[:, 1] - pad_x) / scale  # x1
    scaled_boxes[:, 2] = (boxes[:, 2] - pad_y) / scale  # y1
    scaled_boxes[:, 3] = (boxes[:, 3] - pad_x) / scale  # x2
    scaled_boxes[:, 4] = (boxes[:, 4] - pad_y) / scale  # y2

    # Clamp to image boundaries
    scaled_boxes[:, 1] = scaled_boxes[:, 1].clamp(0, orig_w)
    scaled_boxes[:, 2] = scaled_boxes[:, 2].clamp(0, orig_h)
    scaled_boxes[:, 3] = scaled_boxes[:, 3].clamp(0, orig_w)
    scaled_boxes[:, 4] = scaled_boxes[:, 4].clamp(0, orig_h)

    return scaled_boxes


def predict_image(
    checkpoint_path: str,
    image_path: str,
    output_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.65,
    max_detections: int = 300,
    draw_boxes: bool = True,
    class_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640),
) -> dict:
    """
    Run inference on a single image and optionally save results.

    Args:
        checkpoint_path: Path to model checkpoint (.ckpt)
        image_path: Path to input image
        output_path: Path to save output image (None to skip saving)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections
        draw_boxes: Whether to draw bounding boxes on the image
        class_names: List of class names for labels
        device: Device to use (auto-detected if None)
        image_size: Input size for the model

    Returns:
        Dictionary with predictions and metadata
    """
    # Load model
    model, model_cfg, device = load_model_from_checkpoint(checkpoint_path, device)

    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path, image_size)

    # Run inference
    predictions = run_inference(
        model=model,
        model_cfg=model_cfg,
        image_tensor=image_tensor,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )

    # Format results
    pred = predictions[0]
    results = {
        "image_path": str(image_path),
        "num_detections": len(pred),
        "detections": [],
    }

    if len(pred) > 0:
        # Scale boxes from letterbox space to original image space
        scaled_pred = scale_boxes_to_original(
            pred,
            original_size=original_image.size,  # (width, height)
            letterbox_size=image_size,  # (width, height)
        )

        for det in scaled_pred:
            class_id = int(det[0].item())
            results["detections"].append({
                "class_id": class_id,
                "class_name": class_names[class_id] if class_names else str(class_id),
                "bbox": [det[1].item(), det[2].item(), det[3].item(), det[4].item()],
                "confidence": det[5].item(),
            })

        # Draw and save if requested
        if draw_boxes and output_path:
            result_image = draw_bboxes(original_image, scaled_pred, idx2label=class_names)
            result_image.save(output_path)
            results["output_path"] = str(output_path)

    return results


def predict_directory(
    checkpoint_path: str,
    input_dir: str,
    output_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.65,
    max_detections: int = 300,
    draw_boxes: bool = True,
    class_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640),
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> List[dict]:
    """
    Run inference on all images in a directory.

    Args:
        checkpoint_path: Path to model checkpoint (.ckpt)
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections
        draw_boxes: Whether to draw bounding boxes
        class_names: List of class names for labels
        device: Device to use
        image_size: Input size for the model
        extensions: Valid image file extensions

    Returns:
        List of result dictionaries
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model once
    model, model_cfg, device = load_model_from_checkpoint(checkpoint_path, device)

    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    results = []
    for img_file in sorted(image_files):
        # Preprocess
        image_tensor, original_image = preprocess_image(str(img_file), image_size)

        # Inference
        predictions = run_inference(
            model=model,
            model_cfg=model_cfg,
            image_tensor=image_tensor,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

        # Format results
        pred = predictions[0]
        result = {
            "image_path": str(img_file),
            "num_detections": len(pred),
            "detections": [],
        }

        if len(pred) > 0:
            # Scale boxes from letterbox space to original image space
            scaled_pred = scale_boxes_to_original(
                pred,
                original_size=original_image.size,  # (width, height)
                letterbox_size=image_size,  # (width, height)
            )

            for det in scaled_pred:
                class_id = int(det[0].item())
                result["detections"].append({
                    "class_id": class_id,
                    "class_name": class_names[class_id] if class_names else str(class_id),
                    "bbox": [det[1].item(), det[2].item(), det[3].item(), det[4].item()],
                    "confidence": det[5].item(),
                })

            # Draw and save
            if draw_boxes:
                out_file = output_path / img_file.name
                result_image = draw_bboxes(original_image, scaled_pred, idx2label=class_names)
                result_image.save(out_file)
                result["output_path"] = str(out_file)

        results.append(result)
        print(f"Processed {img_file.name}: {len(pred)} detections")

    return results
