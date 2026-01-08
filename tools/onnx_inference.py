#!/usr/bin/env python3
"""
ONNX Inference Script for YOLOv9 models.

This script performs inference on ONNX models exported in the YOLOv9 format:
- Input: [B, 3, H, W] normalized to [0, 1]
- Output: [B, 4+num_classes, num_anchors] with XYWH boxes and sigmoid scores

Compatible with:
- Original WongKinYiu/yolov9 exports
- YOLOV9MIT exports with YOLOv9ExportWrapper
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int],
    fill_value: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterboxing to maintain aspect ratio.

    Args:
        image: Input image (H, W, C) in BGR format
        target_size: Target size (height, width)
        fill_value: Padding fill value (default: 114, gray)

    Returns:
        resized_image: Letterboxed image
        scale: Scale factor applied
        padding: (pad_x, pad_y) padding applied
    """
    target_h, target_w = target_size
    orig_h, orig_w = image.shape[:2]

    # Calculate scale to fit within target while maintaining aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    padded = np.full((target_h, target_w, 3), fill_value, dtype=np.uint8)

    # Center the resized image
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return padded, scale, (pad_x, pad_y)


def preprocess(
    image_path: str,
    input_size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Load and preprocess image for inference.

    Args:
        image_path: Path to input image
        input_size: Model input size (height, width)

    Returns:
        input_tensor: Preprocessed tensor [1, 3, H, W]
        original_image: Original image for visualization
        scale: Scale factor for box rescaling
        padding: Padding applied for box rescaling
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    original_image = image.copy()

    # Letterbox resize
    image, scale, padding = letterbox(image, input_size)

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # HWC to CHW
    image = image.transpose(2, 0, 1)

    # Add batch dimension
    input_tensor = np.expand_dims(image, axis=0)

    return input_tensor, original_image, scale, padding


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from XYWH (center) to XYXY (corners) format.

    Args:
        boxes: [N, 4] in XYWH format (center_x, center_y, width, height)

    Returns:
        boxes: [N, 4] in XYXY format (x1, y1, x2, y2)
    """
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return xyxy


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> List[int]:
    """
    Non-Maximum Suppression.

    Args:
        boxes: [N, 4] in XYXY format
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        keep: List of indices to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess(
    output: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    num_classes: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Post-process YOLOv9 output.

    Args:
        output: Model output [B, 4+num_classes, num_anchors]
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
        num_classes: Number of classes (auto-detected if None)

    Returns:
        detections: List of [N, 6] arrays per batch (x1, y1, x2, y2, score, class_id)
    """
    batch_size = output.shape[0]
    if num_classes is None:
        num_classes = output.shape[1] - 4

    all_detections = []

    for b in range(batch_size):
        # Extract boxes and scores
        # output[b] shape: [4+num_classes, num_anchors]
        boxes_xywh = output[b, :4, :].T  # [num_anchors, 4]
        class_scores = output[b, 4:, :].T  # [num_anchors, num_classes]

        # Get max class score and class id for each anchor
        max_scores = np.max(class_scores, axis=1)  # [num_anchors]
        class_ids = np.argmax(class_scores, axis=1)  # [num_anchors]

        # Filter by confidence
        mask = max_scores > conf_threshold
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            all_detections.append(np.zeros((0, 6)))
            continue

        # Convert to XYXY
        boxes_xyxy = xywh_to_xyxy(boxes_xywh)

        # Apply NMS per class
        final_boxes = []
        final_scores = []
        final_classes = []

        for cls_id in range(num_classes):
            cls_mask = class_ids == cls_id
            if not np.any(cls_mask):
                continue

            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = max_scores[cls_mask]

            keep = nms(cls_boxes, cls_scores, iou_threshold)

            final_boxes.append(cls_boxes[keep])
            final_scores.append(cls_scores[keep])
            final_classes.append(np.full(len(keep), cls_id))

        if len(final_boxes) == 0:
            all_detections.append(np.zeros((0, 6)))
            continue

        final_boxes = np.concatenate(final_boxes)
        final_scores = np.concatenate(final_scores)
        final_classes = np.concatenate(final_classes)

        # Combine into [N, 6] array
        detections = np.column_stack(
            [final_boxes, final_scores, final_classes]
        )

        # Sort by score descending
        detections = detections[detections[:, 4].argsort()[::-1]]

        all_detections.append(detections)

    return all_detections


def scale_boxes(
    boxes: np.ndarray,
    scale: float,
    padding: Tuple[int, int],
) -> np.ndarray:
    """
    Scale boxes back to original image coordinates.

    Args:
        boxes: [N, 4] boxes in XYXY format (letterboxed coordinates)
        scale: Scale factor used in letterboxing
        padding: (pad_x, pad_y) padding applied

    Returns:
        boxes: [N, 4] boxes in original image coordinates
    """
    if len(boxes) == 0:
        return boxes

    boxes = boxes.copy()
    pad_x, pad_y = padding

    # Remove padding offset
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y

    # Scale back
    boxes /= scale

    return boxes


def draw_detections(
    image: np.ndarray,
    detections: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Draw bounding boxes on image.

    Args:
        image: Input image (BGR)
        detections: [N, 6] array (x1, y1, x2, y2, score, class_id)
        class_names: Optional list of class names

    Returns:
        image: Image with drawn boxes
    """
    image = image.copy()

    # Generate colors for classes
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)

        color = tuple(map(int, colors[class_id % len(colors)]))

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {score:.2f}"
        else:
            label = f"Class {class_id}: {score:.2f}"

        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return image


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Inference for YOLOv9 models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (auto-detected if not specified)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Input size (height width), default: 640 640",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: display only)",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Path to class names file (one name per line)",
    )

    args = parser.parse_args()

    # Load ONNX model
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed. Install with: pip install onnxruntime")
        return 1

    print(f"Loading model: {args.model}")
    session = ort.InferenceSession(args.model)

    # Get model info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"Input: {input_info.name} {input_info.shape}")
    print(f"Output: {output_info.name} {output_info.shape}")

    # Determine input size from model if possible
    input_size = tuple(args.input_size)
    if input_info.shape[2] is not None and input_info.shape[3] is not None:
        model_h, model_w = input_info.shape[2], input_info.shape[3]
        if model_h != 0 and model_w != 0:
            input_size = (model_h, model_w)
            print(f"Using model input size: {input_size}")

    # Load class names if provided
    class_names = None
    if args.class_names:
        with open(args.class_names) as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(class_names)} class names")

    # Preprocess image
    print(f"Processing image: {args.image}")
    input_tensor, original_image, scale, padding = preprocess(
        args.image, input_size
    )

    # Run inference
    print("Running inference...")
    outputs = session.run(None, {input_info.name: input_tensor})
    output = outputs[0]

    print(f"Output shape: {output.shape}")

    # Detect number of classes
    num_classes = args.num_classes
    if num_classes is None:
        num_classes = output.shape[1] - 4
        print(f"Auto-detected {num_classes} classes")

    # Post-process
    detections = postprocess(
        output,
        conf_threshold=args.conf_thresh,
        iou_threshold=args.iou_thresh,
        num_classes=num_classes,
    )[0]

    print(f"Found {len(detections)} detections")

    # Scale boxes back to original image
    if len(detections) > 0:
        detections[:, :4] = scale_boxes(detections[:, :4], scale, padding)

        # Print detections
        for i, det in enumerate(detections):
            x1, y1, x2, y2, score, class_id = det
            class_id = int(class_id)
            if class_names and class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"Class {class_id}"
            print(
                f"  [{i}] {class_name}: {score:.3f} "
                f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
            )

    # Draw and save/display
    result_image = draw_detections(original_image, detections, class_names)

    if args.output:
        cv2.imwrite(args.output, result_image)
        print(f"Saved result to: {args.output}")
    else:
        cv2.imshow("YOLOv9 Detection", result_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    exit(main())
