#!/usr/bin/env python3
"""
TFLite model validation script for YOLOv9.

This script validates TFLite models (FP32, FP16, INT8) on COCO val2017 dataset
and computes mAP metrics. It handles the different output formats between
FP32/FP16 and INT8 TFLite models.

Output tensor structure for YOLOv9:
- Main head (first 9 outputs for FP32/FP16):
  - Identity_0: (1, 80, 80, 80) - cls scale P3
  - Identity_1: (1, 16, 4, 80, 80) - anc scale P3 (ignored)
  - Identity_2: (1, 4, 80, 80) - box scale P3
  - Identity_3: (1, 40, 40, 80) - cls scale P4
  - Identity_4: (1, 16, 4, 40, 40) - anc scale P4 (ignored)
  - Identity_5: (1, 4, 40, 40) - box scale P4
  - Identity_6: (1, 20, 20, 80) - cls scale P5
  - Identity_7: (1, 16, 4, 20, 20) - anc scale P5 (ignored)
  - Identity_8: (1, 4, 20, 20) - box scale P5

Usage:
    python scripts/validate_tflite.py --model exports/v9-t_fp32.tflite
    python scripts/validate_tflite.py --model exports/v9-t_int8.tflite
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms
from tqdm import tqdm


def load_tflite_model(model_path: str):
    """Load TFLite model and return interpreter."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def preprocess_image(image_path: str, input_size: tuple = (640, 640)) -> np.ndarray:
    """Preprocess image for TFLite inference."""
    img = Image.open(image_path).convert("RGB")

    # Resize with letterbox
    original_size = img.size  # (W, H)
    target_w, target_h = input_size

    # Calculate scale
    scale = min(target_w / original_size[0], target_h / original_size[1])
    new_w = int(original_size[0] * scale)
    new_h = int(original_size[1] * scale)

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Create letterbox image
    letterbox = Image.new("RGB", input_size, (114, 114, 114))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    letterbox.paste(img_resized, (paste_x, paste_y))

    # To numpy and normalize
    img_array = np.array(letterbox).astype(np.float32) / 255.0

    # NHWC format for TFLite
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, (original_size, scale, paste_x, paste_y)


def identify_outputs_fp32(output_details):
    """
    Identify cls and box outputs for FP32/FP16 models.
    These have predictable naming: Identity_0, Identity_1, etc.
    Pattern: cls, anc, box repeated for each scale (P3, P4, P5)
    Only use first 9 outputs (main head), ignore aux head (outputs 9-17).
    """
    cls_indices = []
    box_indices = []

    # First 9 outputs are main head: [cls, anc, box] x 3 scales
    for i in range(0, 9, 3):  # 0, 3, 6
        cls_indices.append(i)
        box_indices.append(i + 2)

    return cls_indices, box_indices


def identify_outputs_int8(output_details):
    """
    Identify cls and box outputs for INT8 models.
    INT8 has different ordering, need to identify by shape.

    For YOLOv9-t:
    - cls outputs: shape (1, H, W, 80) - 80 is num_classes
    - box outputs: shape (1, 4, H, W) - 4 is LTRB
    - anc outputs: shape (1, 16, 4, H, W) - 5D tensors, ignored

    Grid sizes: 80x80 (P3), 40x40 (P4), 20x20 (P5)

    Note: There are 6 cls and 6 box outputs (main + aux heads).
    We need to select only 3 (one per scale) from the main head.
    We use the fact that outputs are grouped: take only one per scale.
    """
    cls_by_scale = {}  # grid_size -> list of (index, name)
    box_by_scale = {}  # grid_size -> list of (index, name)

    for i, det in enumerate(output_details):
        shape = det['shape']
        name = det['name']

        # Skip 5D tensors (anchor outputs)
        if len(shape) == 5:
            continue

        if len(shape) != 4:
            continue

        # Check for cls output: (1, H, W, 80) - 80 classes
        if shape[3] == 80 and shape[1] == shape[2]:
            grid_size = shape[1]
            if grid_size not in cls_by_scale:
                cls_by_scale[grid_size] = []
            cls_by_scale[grid_size].append((i, name))

        # Check for box output: (1, 4, H, W) - 4 is LTRB
        elif shape[1] == 4 and shape[2] == shape[3]:
            grid_size = shape[2]
            if grid_size not in box_by_scale:
                box_by_scale[grid_size] = []
            box_by_scale[grid_size].append((i, name))

    # For INT8, we need to select the correct outputs based on valid data ranges
    # Main head outputs have meaningful values, aux head outputs are often constant
    # Hardcoded indices based on analysis for YOLOv9-t INT8 model
    # cls: idx=10 (80x80), idx=7 (40x40), idx=17 (20x20)
    # box: idx=11 (80x80), idx=6 (40x40), idx=13 (20x20)

    # Check if we have the expected structure
    if 80 in cls_by_scale and 40 in cls_by_scale and 20 in cls_by_scale:
        # Use hardcoded indices for YOLOv9-t INT8
        return [10, 7, 17], [11, 6, 13]

    # Fallback: Sort scales descending: 80, 40, 20
    scales = sorted(cls_by_scale.keys(), reverse=True)

    cls_indices = []
    box_indices = []

    for scale in scales:
        # Take second cls and second box for this scale (index 1 if available, else 0)
        if scale in cls_by_scale and cls_by_scale[scale]:
            idx = 1 if len(cls_by_scale[scale]) > 1 else 0
            cls_indices.append(cls_by_scale[scale][idx][0])
        if scale in box_by_scale and box_by_scale[scale]:
            idx = 1 if len(box_by_scale[scale]) > 1 else 0
            box_indices.append(box_by_scale[scale][idx][0])

    return cls_indices, box_indices


def generate_anchors(image_size, strides):
    """Generate anchor grid and scalers."""
    W, H = image_size
    anchors = []
    scalers = []

    for stride in strides:
        h_grid = H // stride
        w_grid = W // stride
        anchor_num = h_grid * w_grid

        scalers.append(np.full((anchor_num,), stride, dtype=np.float32))

        shift = stride // 2
        h = np.arange(0, H, stride) + shift
        w = np.arange(0, W, stride) + shift

        anchor_h, anchor_w = np.meshgrid(h, w, indexing='ij')
        anchor = np.stack([anchor_w.flatten(), anchor_h.flatten()], axis=-1)
        anchors.append(anchor)

    all_anchors = np.concatenate(anchors, axis=0)
    all_scalers = np.concatenate(scalers, axis=0)

    return all_anchors, all_scalers


def dequantize_output(interpreter, output_details, index):
    """Get output tensor and dequantize if INT8."""
    det = output_details[index]
    tensor = interpreter.get_tensor(det['index'])

    if det['dtype'] == np.int8:
        quant_params = det.get('quantization_parameters', {})
        scales = quant_params.get('scales', [1.0])
        zero_points = quant_params.get('zero_points', [0])

        if len(scales) > 0:
            scale = scales[0]
            zp = zero_points[0] if len(zero_points) > 0 else 0
            tensor = (tensor.astype(np.float32) - zp) * scale

    return tensor


def decode_outputs(interpreter, output_details, cls_indices, box_indices, image_size=(640, 640)):
    """
    Decode TFLite outputs to predictions using Vec2Box logic.

    Returns:
        cls_pred: (N, 80) class logits
        box_pred: (N, 4) boxes in xyxy format
    """
    strides = [8, 16, 32]  # P3, P4, P5
    anchor_grid, scaler = generate_anchors(image_size, strides)

    all_cls = []
    all_box = []

    for scale_idx, (cls_idx, box_idx) in enumerate(zip(cls_indices, box_indices)):
        # Get cls output
        cls_out = dequantize_output(interpreter, output_details, cls_idx)
        # Get box output
        box_out = dequantize_output(interpreter, output_details, box_idx)

        cls_shape = output_details[cls_idx]['shape']
        box_shape = output_details[box_idx]['shape']

        # Handle different formats
        # FP32/FP16 cls: (1, H, W, 80) - already NHWC
        # INT8 cls: (1, H, W, 80) - same
        # FP32/FP16 box: (1, 4, H, W) - NCHW
        # INT8 box: (1, 4, H, W) - same

        # cls is already NHWC, reshape to (H*W, 80)
        H_cls, W_cls = cls_shape[1], cls_shape[2]
        cls_flat = cls_out.reshape(-1, 80)  # (H*W, 80)

        # box is NCHW, convert to NHWC then reshape
        # (1, 4, H, W) -> (1, H, W, 4) -> (H*W, 4)
        H_box, W_box = box_shape[2], box_shape[3]
        box_out = box_out.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        box_flat = box_out.reshape(-1, 4)  # (H*W, 4)

        all_cls.append(cls_flat)
        all_box.append(box_flat)

    # Concatenate all scales
    cls_pred = np.concatenate(all_cls, axis=0)  # (8400, 80)
    box_pred = np.concatenate(all_box, axis=0)  # (8400, 4)

    # Decode boxes using Vec2Box logic: LTRB -> xyxy
    # pred_LTRB = box * scaler
    pred_LTRB = box_pred * scaler.reshape(-1, 1)
    lt = pred_LTRB[:, :2]  # left, top distances
    rb = pred_LTRB[:, 2:]  # right, bottom distances

    # xyxy = [anchor - lt, anchor + rb]
    x1y1 = anchor_grid - lt
    x2y2 = anchor_grid + rb
    boxes_xyxy = np.concatenate([x1y1, x2y2], axis=-1)

    return cls_pred, boxes_xyxy


def postprocess_predictions(
    cls_pred: np.ndarray,
    box_pred: np.ndarray,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.65,
    max_detections: int = 300,
    input_size: tuple = (640, 640),
):
    """
    Apply sigmoid, NMS and filter predictions.

    Returns:
        boxes: (N, 4) xyxy format
        scores: (N,) confidence scores
        labels: (N,) class labels
    """
    # Sigmoid for class predictions
    cls_probs = 1 / (1 + np.exp(-np.clip(cls_pred, -50, 50)))

    # Get max class probability and class index
    max_probs = cls_probs.max(axis=1)
    class_ids = cls_probs.argmax(axis=1)

    # Filter by confidence
    mask = max_probs > conf_threshold
    boxes = box_pred[mask]
    scores = max_probs[mask]
    labels = class_ids[mask]

    if len(boxes) == 0:
        return np.array([]).reshape(0, 4), np.array([]), np.array([])

    # Clip boxes to image bounds
    boxes[:, 0] = np.clip(boxes[:, 0], 0, input_size[0])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, input_size[1])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, input_size[0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, input_size[1])

    # Apply NMS
    boxes_t = torch.from_numpy(boxes).float()
    scores_t = torch.from_numpy(scores).float()
    labels_t = torch.from_numpy(labels).long()

    # Use batched NMS (per-class)
    from torchvision.ops import batched_nms
    keep = batched_nms(boxes_t, scores_t, labels_t, iou_threshold)
    keep = keep[:max_detections]

    return boxes[keep.numpy()], scores[keep.numpy()], labels[keep.numpy()]


def scale_boxes_to_original(boxes, preprocess_info, input_size=(640, 640)):
    """Scale boxes from input size back to original image size."""
    if len(boxes) == 0:
        return boxes

    original_size, scale, paste_x, paste_y = preprocess_info

    # Remove letterbox offset
    boxes = boxes.copy()
    boxes[:, 0] -= paste_x
    boxes[:, 1] -= paste_y
    boxes[:, 2] -= paste_x
    boxes[:, 3] -= paste_y

    # Scale back to original size
    boxes = boxes / scale

    # Clip to original image bounds
    boxes[:, 0] = np.clip(boxes[:, 0], 0, original_size[0])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, original_size[1])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, original_size[0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, original_size[1])

    return boxes


def load_coco_annotations(ann_file: str, images_dir: str, num_images: int = None):
    """Load COCO annotations and return list of (image_path, targets) tuples."""
    with open(ann_file) as f:
        coco = json.load(f)

    # Build image id to annotations mapping
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # Build category id to continuous index mapping (COCO ids are not continuous)
    cat_id_to_idx = {}
    for idx, cat in enumerate(sorted(coco['categories'], key=lambda x: x['id'])):
        cat_id_to_idx[cat['id']] = idx

    # Build dataset
    dataset = []
    images = coco['images'][:num_images] if num_images else coco['images']

    for img_info in images:
        img_path = os.path.join(images_dir, img_info['file_name'])
        img_id = img_info['id']

        anns = img_to_anns.get(img_id, [])

        # Convert annotations to targets
        boxes = []
        labels = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue

            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue

            # COCO bbox is [x, y, width, height] -> convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(cat_id_to_idx[ann['category_id']])

        targets = {
            'boxes': np.array(boxes) if boxes else np.array([]).reshape(0, 4),
            'labels': np.array(labels),
            'image_size': (img_info['width'], img_info['height']),
        }

        dataset.append((img_path, targets))

    return dataset


def validate_tflite(
    model_path: str,
    coco_root: str,
    num_images: int = 500,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.65,
    save_results: bool = True,
):
    """Run validation on TFLite model."""
    print(f"Loading model: {model_path}")
    interpreter, input_details, output_details = load_tflite_model(model_path)

    # Print model info
    input_dtype = input_details[0]['dtype']
    is_int8 = input_dtype == np.int8
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_dtype}")
    print(f"Number of outputs: {len(output_details)}")
    print(f"Model type: {'INT8' if is_int8 else 'FP32/FP16'}")

    # Identify outputs based on model type
    if is_int8:
        cls_indices, box_indices = identify_outputs_int8(output_details)
    else:
        cls_indices, box_indices = identify_outputs_fp32(output_details)

    print(f"Classification output indices: {cls_indices}")
    print(f"Box output indices: {box_indices}")

    # Load dataset
    ann_file = os.path.join(coco_root, "annotations/instances_val2017.json")
    images_dir = os.path.join(coco_root, "val2017")

    print(f"Loading COCO annotations from: {ann_file}")
    dataset = load_coco_annotations(ann_file, images_dir, num_images)
    print(f"Loaded {len(dataset)} images")

    # Initialize metrics
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

    # Run validation
    for img_path, targets in tqdm(dataset, desc="Validating"):
        # Preprocess
        img, preprocess_info = preprocess_image(img_path)

        # Handle INT8 input quantization
        if is_int8:
            quant = input_details[0].get('quantization', (1.0, 0))
            scale, zero_point = quant[0], quant[1]
            img = (img / scale + zero_point).astype(np.int8)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        # Decode outputs
        cls_pred, box_pred = decode_outputs(
            interpreter, output_details, cls_indices, box_indices
        )

        # Postprocess
        boxes, scores, labels = postprocess_predictions(
            cls_pred, box_pred,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        # Scale boxes to original image size
        boxes = scale_boxes_to_original(boxes, preprocess_info)

        preds = [{
            'boxes': torch.from_numpy(boxes).float(),
            'scores': torch.from_numpy(scores).float(),
            'labels': torch.from_numpy(labels).long(),
        }]

        # Format targets
        target_list = [{
            'boxes': torch.from_numpy(targets['boxes']).float(),
            'labels': torch.from_numpy(targets['labels']).long(),
        }]

        metric.update(preds, target_list)

    # Compute metrics
    results = metric.compute()

    # Print results
    print("\n" + "=" * 50)
    print(f"Results for {Path(model_path).name}")
    print("=" * 50)
    print(f"mAP@0.5:0.95: {results['map'].item():.4f}")
    print(f"mAP@0.5:      {results['map_50'].item():.4f}")
    print(f"mAP@0.75:     {results['map_75'].item():.4f}")
    print(f"AR@100:       {results['mar_100'].item():.4f}")

    # Save results
    if save_results:
        model_name = Path(model_path).stem
        output_file = Path(model_path).parent / f"{model_name}_metrics.json"

        metrics_dict = {
            "model": f"YOLOv9-t ({model_name})",
            "dataset": f"COCO val2017 ({num_images} images)",
            "mAP_50_95": results['map'].item(),
            "mAP_50": results['map_50'].item(),
            "mAP_75": results['map_75'].item(),
            "AR_100": results['mar_100'].item(),
        }

        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate TFLite model on COCO")
    parser.add_argument("--model", type=str, required=True, help="Path to TFLite model")
    parser.add_argument("--coco-root", type=str, default="coco", help="Path to COCO dataset root")
    parser.add_argument("--num-images", type=int, default=500, help="Number of images to validate")
    parser.add_argument("--conf-threshold", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.65, help="IoU threshold for NMS")

    args = parser.parse_args()

    validate_tflite(
        model_path=args.model,
        coco_root=args.coco_root,
        num_images=args.num_images,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()
