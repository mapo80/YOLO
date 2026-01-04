"""
Detection metrics module for YOLO models.

This module provides comprehensive metrics computation for object detection
using pycocotools for official COCO evaluation and seaborn for visualization.

The implementation follows the COCO evaluation protocol and is independent
from any third-party YOLO implementations.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

# COCO evaluation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of bounding boxes.

    Uses vectorized operations for efficiency.

    Args:
        boxes1: Array of shape (N, 4) with boxes in [x1, y1, x2, y2] format
        boxes2: Array of shape (M, 4) with boxes in [x1, y1, x2, y2] format

    Returns:
        IoU matrix of shape (N, M)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    # Cast to float64 to avoid overflow with large coordinates
    boxes1 = boxes1.astype(np.float64)
    boxes2 = boxes2.astype(np.float64)

    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersection coordinates
    x1_inter = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    y1_inter = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    x2_inter = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    y2_inter = np.minimum(boxes1[:, 3:4], boxes2[:, 3])

    # Compute intersection area
    width_inter = np.clip(x2_inter - x1_inter, 0, None)
    height_inter = np.clip(y2_inter - y1_inter, 0, None)
    intersection = width_inter * height_inter

    # Compute union
    union = area1[:, np.newaxis] + area2 - intersection

    # Avoid division by zero
    return np.where(union > 0, intersection / union, 0)


def gaussian_smooth(values: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to 1D array.

    Args:
        values: Input array
        sigma: Standard deviation for Gaussian kernel

    Returns:
        Smoothed array
    """
    if len(values) < 3:
        return values
    return gaussian_filter1d(values.astype(float), sigma=sigma, mode='nearest')


class COCOFormatConverter:
    """
    Converts detection predictions and ground truths to COCO format
    for evaluation with pycocotools.
    """

    def __init__(self, class_names: Dict[int, str]):
        """
        Initialize converter.

        Args:
            class_names: Dict mapping class indices to names
        """
        self.class_names = class_names
        self.categories = [
            {"id": idx, "name": name}
            for idx, name in class_names.items()
        ]
        self._image_id = 0
        self._annotation_id = 0

        # Storage for COCO format data
        self.gt_annotations = []
        self.dt_annotations = []
        self.images = []

    def add_batch(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        image_size: Tuple[int, int] = (640, 640),
    ) -> None:
        """
        Add a batch of predictions and ground truths.

        Args:
            predictions: List of dicts with 'boxes', 'scores', 'labels'
            targets: List of dicts with 'boxes', 'labels'
            image_size: Image dimensions (width, height)
        """
        for pred, target in zip(predictions, targets):
            self._image_id += 1

            # Add image entry
            self.images.append({
                "id": self._image_id,
                "width": image_size[0],
                "height": image_size[1],
            })

            # Convert ground truth boxes
            gt_boxes = target["boxes"].cpu().numpy() if len(target["boxes"]) else np.zeros((0, 4))
            gt_labels = target["labels"].cpu().numpy() if len(target["labels"]) else np.zeros(0)

            for box, label in zip(gt_boxes, gt_labels):
                self._annotation_id += 1
                x1, y1, x2, y2 = box
                self.gt_annotations.append({
                    "id": self._annotation_id,
                    "image_id": self._image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # COCO format: [x, y, w, h]
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                })

            # Convert predictions
            pred_boxes = pred["boxes"].cpu().numpy() if len(pred["boxes"]) else np.zeros((0, 4))
            pred_scores = pred["scores"].cpu().numpy() if len(pred["scores"]) else np.zeros(0)
            pred_labels = pred["labels"].cpu().numpy() if len(pred["labels"]) else np.zeros(0)

            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                x1, y1, x2, y2 = box
                self.dt_annotations.append({
                    "image_id": self._image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })

    def get_coco_gt(self) -> COCO:
        """Create COCO object for ground truth."""
        gt_dict = {
            "images": self.images,
            "annotations": self.gt_annotations,
            "categories": self.categories,
        }

        # Write to temp file and load (pycocotools requires file)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(gt_dict, f)
            temp_path = f.name

        coco_gt = COCO(temp_path)
        Path(temp_path).unlink()  # Clean up
        return coco_gt

    def get_coco_dt(self, coco_gt: COCO) -> COCO:
        """Create COCO object for detections."""
        if not self.dt_annotations:
            # Return empty results
            return coco_gt.loadRes([])
        return coco_gt.loadRes(self.dt_annotations)

    def reset(self) -> None:
        """Reset all stored data."""
        self._image_id = 0
        self._annotation_id = 0
        self.gt_annotations = []
        self.dt_annotations = []
        self.images = []


class ConfusionMatrix:
    """
    Confusion matrix for object detection.

    Tracks true positives, false positives, and false negatives
    across all classes including background.
    """

    def __init__(self, num_classes: int, class_names: Optional[Dict[int, str]] = None):
        """
        Initialize confusion matrix.

        Args:
            num_classes: Number of object classes
            class_names: Optional mapping of class indices to names
        """
        self.num_classes = num_classes
        self.class_names = class_names or {i: str(i) for i in range(num_classes)}
        # Matrix shape: (num_classes + 1, num_classes + 1)
        # Last row/column is for background (missed detections / false positives)
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    def update(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        iou_threshold: float = 0.5,
    ) -> None:
        """
        Update matrix with a batch of predictions and ground truth.

        Args:
            predictions: Array (N, 6) with [x1, y1, x2, y2, conf, class]
            ground_truth: Array (M, 5) with [class, x1, y1, x2, y2]
            iou_threshold: IoU threshold for matching
        """
        if len(ground_truth) == 0 and len(predictions) == 0:
            return

        if len(ground_truth) == 0:
            # All predictions are false positives
            for pred in predictions:
                pred_class = int(pred[5])
                self.matrix[pred_class, self.num_classes] += 1
            return

        if len(predictions) == 0:
            # All ground truths are missed (false negatives)
            for gt in ground_truth:
                gt_class = int(gt[0])
                self.matrix[self.num_classes, gt_class] += 1
            return

        # Compute IoU matrix
        pred_boxes = predictions[:, :4]
        gt_boxes = ground_truth[:, 1:5]
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

        # Greedy matching: highest IoU first
        matched_gt = set()
        matched_pred = set()

        # Get all valid matches sorted by IoU (descending)
        valid_matches = []
        for i in range(len(predictions)):
            for j in range(len(ground_truth)):
                if iou_matrix[i, j] >= iou_threshold:
                    valid_matches.append((iou_matrix[i, j], i, j))

        valid_matches.sort(reverse=True)

        for _, pred_idx, gt_idx in valid_matches:
            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue

            pred_class = int(predictions[pred_idx, 5])
            gt_class = int(ground_truth[gt_idx, 0])

            self.matrix[pred_class, gt_class] += 1
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)

        # Unmatched predictions -> false positives
        for i in range(len(predictions)):
            if i not in matched_pred:
                pred_class = int(predictions[i, 5])
                self.matrix[pred_class, self.num_classes] += 1

        # Unmatched ground truths -> false negatives
        for j in range(len(ground_truth)):
            if j not in matched_gt:
                gt_class = int(ground_truth[j, 0])
                self.matrix[self.num_classes, gt_class] += 1

    def plot(self, save_path: Path, normalize: bool = True) -> None:
        """
        Plot confusion matrix using seaborn heatmap.

        Args:
            save_path: Path to save the figure
            normalize: Whether to normalize values by column
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data
        matrix_data = self.matrix.astype(float)
        if normalize:
            col_sums = matrix_data.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            matrix_data = matrix_data / col_sums

        # Create labels
        labels = [self.class_names.get(i, str(i)) for i in range(self.num_classes)]
        labels.append("BG")

        # Create figure with seaborn
        plt.figure(figsize=(10, 8))
        sns.set_theme(style="white")

        ax = sns.heatmap(
            matrix_data,
            annot=True,
            fmt=".2f" if normalize else ".0f",
            cmap="YlOrRd",
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            cbar_kws={"label": "Normalized" if normalize else "Count"},
            annot_kws={"size": 8},
        )

        ax.set_xlabel("Ground Truth", fontsize=12)
        ax.set_ylabel("Predicted", fontsize=12)
        ax.set_title("Detection Confusion Matrix", fontsize=14, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def reset(self) -> None:
        """Reset matrix to zeros."""
        self.matrix = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int64)


class MetricStorage:
    """
    Storage container for computed detection metrics.

    Provides properties for accessing precision, recall, F1, and mAP values.
    """

    def __init__(self):
        """Initialize empty storage."""
        self.precision = np.array([])
        self.recall = np.array([])
        self.f1 = np.array([])
        self.ap_per_class = np.array([])  # Shape: (num_classes, num_iou_thresholds)
        self.class_indices = np.array([])

        # Curve data for plotting
        self.precision_curve = np.array([])  # (num_classes, num_points)
        self.recall_curve = np.array([])
        self.f1_curve = np.array([])
        self.confidence_axis = np.array([])

    @property
    def mp(self) -> float:
        """Mean precision."""
        return float(self.precision.mean()) if len(self.precision) else 0.0

    @property
    def mr(self) -> float:
        """Mean recall."""
        return float(self.recall.mean()) if len(self.recall) else 0.0

    @property
    def mf1(self) -> float:
        """Mean F1 score."""
        return float(self.f1.mean()) if len(self.f1) else 0.0

    @property
    def map50(self) -> float:
        """Mean AP at IoU=0.50."""
        if len(self.ap_per_class) == 0:
            return 0.0
        return float(self.ap_per_class[:, 0].mean())

    @property
    def map75(self) -> float:
        """Mean AP at IoU=0.75."""
        if len(self.ap_per_class) == 0 or self.ap_per_class.shape[1] < 6:
            return 0.0
        return float(self.ap_per_class[:, 5].mean())  # Index 5 = IoU 0.75

    @property
    def map(self) -> float:
        """Mean AP averaged over IoU thresholds 0.50:0.95."""
        if len(self.ap_per_class) == 0:
            return 0.0
        return float(self.ap_per_class.mean())


def create_precision_recall_curves(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
    num_points: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curves for all classes.

    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching
        num_points: Number of points in curves

    Returns:
        Tuple of (precision, recall, f1, confidence_axis, ap_per_class)
    """
    # Collect all predictions and ground truths
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []
    all_image_ids = []

    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        pred_boxes = pred["boxes"].cpu().numpy() if len(pred["boxes"]) else np.zeros((0, 4))
        pred_scores = pred["scores"].cpu().numpy() if len(pred["scores"]) else np.zeros(0)
        pred_labels = pred["labels"].cpu().numpy() if len(pred["labels"]) else np.zeros(0)

        gt_boxes = target["boxes"].cpu().numpy() if len(target["boxes"]) else np.zeros((0, 4))
        gt_labels = target["labels"].cpu().numpy() if len(target["labels"]) else np.zeros(0)

        all_pred_boxes.extend(pred_boxes)
        all_pred_scores.extend(pred_scores)
        all_pred_labels.extend(pred_labels)
        all_image_ids.extend([img_idx] * len(pred_boxes))

        all_gt_boxes.append(gt_boxes)
        all_gt_labels.append(gt_labels)

    if not all_pred_scores:
        return (
            np.zeros((num_classes, num_points)),
            np.zeros((num_classes, num_points)),
            np.zeros((num_classes, num_points)),
            np.linspace(0, 1, num_points),
            np.zeros(num_classes),
        )

    all_pred_boxes = np.array(all_pred_boxes)
    all_pred_scores = np.array(all_pred_scores)
    all_pred_labels = np.array(all_pred_labels)
    all_image_ids = np.array(all_image_ids)

    # Sort by confidence (descending)
    sorted_indices = np.argsort(-all_pred_scores)

    confidence_axis = np.linspace(0, 1, num_points)
    precision_curves = np.zeros((num_classes, num_points))
    recall_curves = np.zeros((num_classes, num_points))
    ap_values = np.zeros(num_classes)

    for class_idx in range(num_classes):
        # Get predictions for this class
        class_mask = all_pred_labels[sorted_indices] == class_idx
        class_pred_indices = sorted_indices[class_mask]

        if len(class_pred_indices) == 0:
            continue

        # Count ground truth for this class
        n_gt = sum(
            np.sum(gt_labels == class_idx)
            for gt_labels in all_gt_labels
        )

        if n_gt == 0:
            continue

        # Track which ground truths have been matched
        gt_matched = {img_idx: set() for img_idx in range(len(predictions))}

        tp = np.zeros(len(class_pred_indices))
        fp = np.zeros(len(class_pred_indices))

        for i, pred_idx in enumerate(class_pred_indices):
            img_idx = all_image_ids[pred_idx]
            pred_box = all_pred_boxes[pred_idx]

            gt_boxes_img = all_gt_boxes[img_idx]
            gt_labels_img = all_gt_labels[img_idx]

            # Find matching ground truth
            class_gt_mask = gt_labels_img == class_idx
            class_gt_indices = np.where(class_gt_mask)[0]

            best_iou = 0
            best_gt_idx = -1

            for gt_idx in class_gt_indices:
                if gt_idx in gt_matched[img_idx]:
                    continue

                gt_box = gt_boxes_img[gt_idx]
                iou = compute_iou_matrix(pred_box[np.newaxis], gt_box[np.newaxis])[0, 0]

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp[i] = 1
                gt_matched[img_idx].add(best_gt_idx)
            else:
                fp[i] = 1

        # Compute cumulative TP/FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / n_gt

        # Interpolate at confidence thresholds
        pred_conf = all_pred_scores[class_pred_indices]

        for j, conf_thresh in enumerate(confidence_axis):
            mask = pred_conf >= conf_thresh
            if mask.sum() > 0:
                idx = mask.sum() - 1
                precision_curves[class_idx, j] = precision[idx]
                recall_curves[class_idx, j] = recall[idx]

        # Compute AP using trapezoidal integration
        # Sort by recall
        sort_idx = np.argsort(recall)
        sorted_recall = recall[sort_idx]
        sorted_precision = precision[sort_idx]

        # Add boundary points
        sorted_recall = np.concatenate([[0], sorted_recall, [1]])
        sorted_precision = np.concatenate([[1], sorted_precision, [0]])

        # Make precision monotonically decreasing
        for k in range(len(sorted_precision) - 2, -1, -1):
            sorted_precision[k] = max(sorted_precision[k], sorted_precision[k + 1])

        # Compute area under curve
        ap_values[class_idx] = np.trapz(sorted_precision, sorted_recall)

    # Compute F1 curves
    f1_curves = 2 * precision_curves * recall_curves / (precision_curves + recall_curves + 1e-10)

    return precision_curves, recall_curves, f1_curves, confidence_axis, ap_values


def plot_precision_recall(
    precision_curves: np.ndarray,
    recall_curves: np.ndarray,
    ap_values: np.ndarray,
    class_names: Dict[int, str],
    save_path: Path,
) -> None:
    """
    Plot precision-recall curves using seaborn.

    Args:
        precision_curves: Precision values (num_classes, num_points)
        recall_curves: Recall values (num_classes, num_points)
        ap_values: AP value per class
        class_names: Class name mapping
        save_path: Output path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")

    # Use a colorful palette
    colors = sns.color_palette("husl", n_colors=len(class_names))

    # Plot per-class curves
    for i, (class_idx, class_name) in enumerate(class_names.items()):
        if i < len(recall_curves):
            # Sort by recall for proper curve
            sort_idx = np.argsort(recall_curves[i])
            r = recall_curves[i][sort_idx]
            p = precision_curves[i][sort_idx]

            ap = ap_values[i] if i < len(ap_values) else 0
            plt.plot(r, p, color=colors[i], linewidth=1.5,
                    label=f"{class_name} (AP={ap:.3f})")

    # Plot mean curve
    mean_p = precision_curves.mean(axis=0)
    mean_r = recall_curves.mean(axis=0)
    sort_idx = np.argsort(mean_r)
    plt.plot(mean_r[sort_idx], mean_p[sort_idx], 'k-', linewidth=2.5,
            label=f"Mean (mAP={ap_values.mean():.3f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves", fontsize=14, fontweight='bold')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.legend(loc='lower left', fontsize=9)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_f1_curve(
    f1_curves: np.ndarray,
    confidence_axis: np.ndarray,
    class_names: Dict[int, str],
    save_path: Path,
) -> None:
    """
    Plot F1 vs confidence curves using seaborn.

    Args:
        f1_curves: F1 values (num_classes, num_points)
        confidence_axis: Confidence threshold values
        class_names: Class name mapping
        save_path: Output path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")

    colors = sns.color_palette("husl", n_colors=len(class_names))

    for i, (class_idx, class_name) in enumerate(class_names.items()):
        if i < len(f1_curves):
            plt.plot(confidence_axis, f1_curves[i], color=colors[i],
                    linewidth=1.5, label=class_name)

    # Mean F1 curve (smoothed)
    mean_f1 = gaussian_smooth(f1_curves.mean(axis=0), sigma=3)
    best_conf = confidence_axis[np.argmax(mean_f1)]
    best_f1 = mean_f1.max()

    plt.plot(confidence_axis, mean_f1, 'k-', linewidth=2.5,
            label=f"Mean (best={best_f1:.2f} @ {best_conf:.2f})")
    plt.axvline(x=best_conf, color='red', linestyle='--', linewidth=1, alpha=0.7)

    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("F1-Confidence Curves", fontsize=14, fontweight='bold')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.legend(loc='lower left', fontsize=9)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_precision_curve(
    precision_curves: np.ndarray,
    confidence_axis: np.ndarray,
    class_names: Dict[int, str],
    save_path: Path,
) -> None:
    """Plot precision vs confidence curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")

    colors = sns.color_palette("husl", n_colors=len(class_names))

    for i, (class_idx, class_name) in enumerate(class_names.items()):
        if i < len(precision_curves):
            plt.plot(confidence_axis, precision_curves[i], color=colors[i],
                    linewidth=1.5, label=class_name)

    mean_precision = precision_curves.mean(axis=0)
    plt.plot(confidence_axis, mean_precision, 'k-', linewidth=2.5, label="Mean")

    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Confidence Curves", fontsize=14, fontweight='bold')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.legend(loc='lower left', fontsize=9)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_recall_curve(
    recall_curves: np.ndarray,
    confidence_axis: np.ndarray,
    class_names: Dict[int, str],
    save_path: Path,
) -> None:
    """Plot recall vs confidence curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")

    colors = sns.color_palette("husl", n_colors=len(class_names))

    for i, (class_idx, class_name) in enumerate(class_names.items()):
        if i < len(recall_curves):
            plt.plot(confidence_axis, recall_curves[i], color=colors[i],
                    linewidth=1.5, label=class_name)

    mean_recall = recall_curves.mean(axis=0)
    plt.plot(confidence_axis, mean_recall, 'k-', linewidth=2.5, label="Mean")

    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("Recall", fontsize=12)
    plt.title("Recall-Confidence Curves", fontsize=14, fontweight='bold')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class DetMetrics:
    """
    Complete detection metrics aggregator.

    Collects predictions and ground truths during validation,
    then computes comprehensive metrics using pycocotools
    for official COCO evaluation.
    """

    def __init__(
        self,
        names: Dict[int, str],
        iou_thresholds: Optional[np.ndarray] = None,
    ):
        """
        Initialize detection metrics.

        Args:
            names: Dict mapping class indices to names
            iou_thresholds: IoU thresholds for evaluation (default: COCO 0.50:0.95)
        """
        self.names = names
        self.nc = len(names)
        self.iou_thresholds = (
            iou_thresholds if iou_thresholds is not None
            else np.linspace(0.5, 0.95, 10)
        )

        # Storage
        self.box = MetricStorage()
        self.confusion_matrix = ConfusionMatrix(self.nc, names)
        self.coco_converter = COCOFormatConverter(names)

        # Raw prediction/target storage for curve computation
        self._predictions: List[Dict[str, torch.Tensor]] = []
        self._targets: List[Dict[str, torch.Tensor]] = []

        # Per-class instance counts
        self.nt_per_class: Optional[np.ndarray] = None

    def update(
        self,
        preds: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Update metrics with a batch of predictions and targets.

        Args:
            preds: List of prediction dicts with 'boxes', 'scores', 'labels'
            targets: List of target dicts with 'boxes', 'labels'
        """
        # Store for later processing
        self._predictions.extend(preds)
        self._targets.extend(targets)

        # Update COCO format converter
        self.coco_converter.add_batch(preds, targets)

        # Update confusion matrix
        for pred, target in zip(preds, targets):
            pred_boxes = pred["boxes"].cpu().numpy() if len(pred["boxes"]) else np.zeros((0, 4))
            pred_scores = pred["scores"].cpu().numpy() if len(pred["scores"]) else np.zeros(0)
            pred_labels = pred["labels"].cpu().numpy() if len(pred["labels"]) else np.zeros(0)

            gt_boxes = target["boxes"].cpu().numpy() if len(target["boxes"]) else np.zeros((0, 4))
            gt_labels = target["labels"].cpu().numpy() if len(target["labels"]) else np.zeros(0)

            if len(pred_boxes) > 0:
                predictions_arr = np.column_stack([
                    pred_boxes, pred_scores, pred_labels
                ])
            else:
                predictions_arr = np.zeros((0, 6))

            if len(gt_boxes) > 0:
                gt_arr = np.column_stack([gt_labels, gt_boxes])
            else:
                gt_arr = np.zeros((0, 5))

            self.confusion_matrix.update(predictions_arr, gt_arr)

    def process(
        self,
        save_dir: Optional[Path] = None,
        plot: bool = True,
        conf_prod: float = 0.25,
        threshold_sweep_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Process accumulated data and compute final metrics.

        Uses pycocotools for official COCO evaluation.

        Args:
            save_dir: Directory to save plots
            plot: Whether to generate visualization plots
            conf_prod: Production confidence threshold for operative metrics
            threshold_sweep_values: Confidence thresholds for sweep (default: 0.1-0.5)

        Returns:
            Dict with computed metrics including:
            - COCO standard metrics (map, map50, map75)
            - Size-based metrics (map_small, map_medium, map_large)
            - Average recall metrics (ar_1, ar_10, ar_100, ar_small, ar_medium, ar_large)
            - Operative metrics at conf_prod (precision_at_conf, recall_at_conf, f1_at_conf)
            - Best F1 and optimal threshold (best_f1, best_f1_conf)
            - Error stats (total_fp, total_fn, mean_det_per_img, p95_det_per_img, mean_iou_tp)
            - Per-class data and top confusions
            - Threshold sweep results
        """
        if threshold_sweep_values is None:
            threshold_sweep_values = [0.1, 0.2, 0.3, 0.4, 0.5]

        empty_result = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "map": 0.0,
            "map50": 0.0,
            "map75": 0.0,
            "map_small": 0.0,
            "map_medium": 0.0,
            "map_large": 0.0,
            "ar_1": 0.0,
            "ar_10": 0.0,
            "ar_100": 0.0,
            "ar_small": 0.0,
            "ar_medium": 0.0,
            "ar_large": 0.0,
            "precision_at_conf": 0.0,
            "recall_at_conf": 0.0,
            "f1_at_conf": 0.0,
            "conf_prod": conf_prod,
            "best_f1": 0.0,
            "best_f1_conf": 0.0,
            "total_fp": 0,
            "total_fn": 0,
            "mean_det_per_img": 0.0,
            "p95_det_per_img": 0.0,
            "mean_iou_tp": 0.0,
            "per_class": [],
            "top_confusions": [],
            "threshold_sweep": {t: {"p": 0.0, "r": 0.0, "f1": 0.0} for t in threshold_sweep_values},
        }

        if not self._predictions:
            return empty_result

        from yolo.utils.logger import logger
        import time as _time

        logger.debug(f"[Metrics] Processing {len(self._predictions)} predictions...")

        # Compute COCO metrics
        coco_stats = np.zeros(12)  # 12 standard COCO metrics
        try:
            _t0 = _time.time()
            coco_gt = self.coco_converter.get_coco_gt()
            coco_dt = self.coco_converter.get_coco_dt(coco_gt)
            logger.debug(f"[Metrics] COCO conversion done in {_time.time() - _t0:.2f}s")

            _t0 = _time.time()
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            logger.debug(f"[Metrics] COCO eval done in {_time.time() - _t0:.2f}s")

            # Suppress verbose COCO output - metrics are shown in EvalDashboard
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                coco_eval.summarize()
            finally:
                sys.stdout = old_stdout

            # Extract all 12 metrics from COCO evaluation
            # stats order: AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl
            coco_stats = coco_eval.stats
        except Exception as e:
            logger.warning(f"[Metrics] COCO eval failed: {e}")

        # COCO standard metrics
        map_val = float(coco_stats[0])    # AP @ IoU=0.50:0.95
        map50 = float(coco_stats[1])      # AP @ IoU=0.50
        map75 = float(coco_stats[2])      # AP @ IoU=0.75
        map_small = float(coco_stats[3])  # AP small objects
        map_medium = float(coco_stats[4]) # AP medium objects
        map_large = float(coco_stats[5])  # AP large objects
        ar_1 = float(coco_stats[6])       # AR @ 1 det/img
        ar_10 = float(coco_stats[7])      # AR @ 10 det/img
        ar_100 = float(coco_stats[8])     # AR @ 100 det/img
        ar_small = float(coco_stats[9])   # AR small
        ar_medium = float(coco_stats[10]) # AR medium
        ar_large = float(coco_stats[11])  # AR large

        # Compute precision-recall curves and per-class metrics
        logger.debug("[Metrics] Computing precision-recall curves...")
        _t0 = _time.time()
        precision_curves, recall_curves, f1_curves, conf_axis, ap_values = \
            create_precision_recall_curves(
                self._predictions,
                self._targets,
                self.nc,
                iou_threshold=0.5,
            )
        logger.debug(f"[Metrics] PR curves done in {_time.time() - _t0:.2f}s")

        # Find best F1 threshold (smoothed)
        mean_f1 = gaussian_smooth(f1_curves.mean(axis=0), sigma=3)
        best_idx = np.argmax(mean_f1)
        best_f1 = float(mean_f1[best_idx])
        best_f1_conf = float(conf_axis[best_idx])

        # Store metrics at best F1 threshold (for backward compatibility)
        self.box.precision = precision_curves[:, best_idx]
        self.box.recall = recall_curves[:, best_idx]
        self.box.f1 = f1_curves[:, best_idx]
        self.box.ap_per_class = np.column_stack([ap_values] * 10)  # Replicate for compatibility
        self.box.class_indices = np.arange(self.nc)
        self.box.precision_curve = precision_curves
        self.box.recall_curve = recall_curves
        self.box.f1_curve = f1_curves
        self.box.confidence_axis = conf_axis

        # Compute metrics at production confidence threshold (conf_prod)
        conf_prod_idx = np.argmin(np.abs(conf_axis - conf_prod))
        precision_at_conf = float(precision_curves.mean(axis=0)[conf_prod_idx])
        recall_at_conf = float(recall_curves.mean(axis=0)[conf_prod_idx])
        f1_at_conf = float(f1_curves.mean(axis=0)[conf_prod_idx])

        # Threshold sweep - P/R/F1 at multiple thresholds
        threshold_sweep = {}
        for thresh in threshold_sweep_values:
            thresh_idx = np.argmin(np.abs(conf_axis - thresh))
            threshold_sweep[thresh] = {
                "p": float(precision_curves.mean(axis=0)[thresh_idx]),
                "r": float(recall_curves.mean(axis=0)[thresh_idx]),
                "f1": float(f1_curves.mean(axis=0)[thresh_idx]),
            }

        # Count instances per class
        all_gt_labels = []
        for target in self._targets:
            gt_labels = target["labels"].cpu().numpy() if len(target["labels"]) else np.zeros(0)
            all_gt_labels.extend(gt_labels)
        self.nt_per_class = np.bincount(
            np.array(all_gt_labels).astype(int),
            minlength=self.nc
        )

        # Compute error statistics (FP, FN, det/img, mean IoU)
        total_fp, total_fn, mean_det_per_img, p95_det_per_img, mean_iou_tp = \
            self._compute_error_stats(conf_prod)

        # Build per-class data
        per_class = []
        for i, (class_idx, class_name) in enumerate(self.names.items()):
            p = float(precision_curves[i, conf_prod_idx]) if i < len(precision_curves) else 0.0
            r = float(recall_curves[i, conf_prod_idx]) if i < len(recall_curves) else 0.0
            ap = float(ap_values[i]) if i < len(ap_values) else 0.0
            ap50 = ap  # Our ap_values are at IoU=0.5
            support = int(self.nt_per_class[i]) if i < len(self.nt_per_class) else 0
            per_class.append({
                "name": class_name,
                "ap": ap,
                "ap50": ap50,
                "recall": r,
                "precision": p,
                "support": support,
            })

        # Get top confusions from confusion matrix
        top_confusions = self._get_top_confusions(top_n=10)

        # Generate plots
        if plot and save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            plot_precision_recall(
                precision_curves, recall_curves, ap_values,
                self.names, save_dir / "PR_curve.png"
            )
            plot_f1_curve(
                f1_curves, conf_axis, self.names,
                save_dir / "F1_curve.png"
            )
            plot_precision_curve(
                precision_curves, conf_axis, self.names,
                save_dir / "P_curve.png"
            )
            plot_recall_curve(
                recall_curves, conf_axis, self.names,
                save_dir / "R_curve.png"
            )
            self.confusion_matrix.plot(save_dir / "confusion_matrix.png")

        return {
            # Basic metrics (backward compatible)
            "precision": float(self.box.mp),
            "recall": float(self.box.mr),
            "f1": float(self.box.mf1),

            # COCO standard metrics
            "map": map_val,
            "map50": map50,
            "map75": map75,

            # Size-based AP
            "map_small": map_small,
            "map_medium": map_medium,
            "map_large": map_large,

            # Average Recall
            "ar_1": ar_1,
            "ar_10": ar_10,
            "ar_100": ar_100,
            "ar_small": ar_small,
            "ar_medium": ar_medium,
            "ar_large": ar_large,

            # Operative metrics at conf_prod
            "precision_at_conf": precision_at_conf,
            "recall_at_conf": recall_at_conf,
            "f1_at_conf": f1_at_conf,
            "conf_prod": conf_prod,

            # Best F1 threshold
            "best_f1": best_f1,
            "best_f1_conf": best_f1_conf,

            # Error stats
            "total_fp": total_fp,
            "total_fn": total_fn,
            "mean_det_per_img": mean_det_per_img,
            "p95_det_per_img": p95_det_per_img,
            "mean_iou_tp": mean_iou_tp,

            # Per-class data
            "per_class": per_class,

            # Top confusions
            "top_confusions": top_confusions,

            # Threshold sweep
            "threshold_sweep": threshold_sweep,
        }

    def _compute_error_stats(
        self,
        conf_threshold: float,
        iou_threshold: float = 0.5,
    ) -> Tuple[int, int, float, float, float]:
        """
        Compute error statistics at a given confidence threshold.

        Args:
            conf_threshold: Confidence threshold for filtering predictions
            iou_threshold: IoU threshold for matching

        Returns:
            Tuple of (total_fp, total_fn, mean_det_per_img, p95_det_per_img, mean_iou_tp)
        """
        total_fp = 0
        total_fn = 0
        det_per_img = []
        iou_tp_list = []

        for pred, target in zip(self._predictions, self._targets):
            # Filter predictions by confidence
            pred_boxes = pred["boxes"].cpu().numpy() if len(pred["boxes"]) else np.zeros((0, 4))
            pred_scores = pred["scores"].cpu().numpy() if len(pred["scores"]) else np.zeros(0)
            pred_labels = pred["labels"].cpu().numpy() if len(pred["labels"]) else np.zeros(0)

            conf_mask = pred_scores >= conf_threshold
            pred_boxes = pred_boxes[conf_mask]
            pred_labels = pred_labels[conf_mask]

            gt_boxes = target["boxes"].cpu().numpy() if len(target["boxes"]) else np.zeros((0, 4))
            gt_labels = target["labels"].cpu().numpy() if len(target["labels"]) else np.zeros(0)

            det_per_img.append(len(pred_boxes))

            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue
            elif len(pred_boxes) == 0:
                total_fn += len(gt_boxes)
                continue
            elif len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue

            # Compute IoU matrix
            iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

            # Greedy matching
            matched_gt = set()
            matched_pred = set()

            # Sort by IoU descending
            valid_matches = []
            for i in range(len(pred_boxes)):
                for j in range(len(gt_boxes)):
                    if pred_labels[i] == gt_labels[j] and iou_matrix[i, j] >= iou_threshold:
                        valid_matches.append((iou_matrix[i, j], i, j))

            valid_matches.sort(reverse=True)

            for iou_val, pred_idx, gt_idx in valid_matches:
                if pred_idx in matched_pred or gt_idx in matched_gt:
                    continue
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
                iou_tp_list.append(iou_val)

            # Count errors
            total_fp += len(pred_boxes) - len(matched_pred)
            total_fn += len(gt_boxes) - len(matched_gt)

        # Compute statistics
        mean_det_per_img = float(np.mean(det_per_img)) if det_per_img else 0.0
        p95_det_per_img = float(np.percentile(det_per_img, 95)) if det_per_img else 0.0
        mean_iou_tp = float(np.mean(iou_tp_list)) if iou_tp_list else 0.0

        return total_fp, total_fn, mean_det_per_img, p95_det_per_img, mean_iou_tp

    def _get_top_confusions(self, top_n: int = 10) -> List[Dict[str, Union[str, int]]]:
        """
        Get top confused class pairs from confusion matrix.

        Args:
            top_n: Number of top confusions to return

        Returns:
            List of dicts with 'pred', 'true', 'count' keys
        """
        confusions = []
        matrix = self.confusion_matrix.matrix

        # Iterate over off-diagonal elements (excluding background row/col)
        for pred_idx in range(self.nc):
            for true_idx in range(self.nc):
                if pred_idx != true_idx and matrix[pred_idx, true_idx] > 0:
                    pred_name = self.names.get(pred_idx, str(pred_idx))
                    true_name = self.names.get(true_idx, str(true_idx))
                    confusions.append({
                        "pred": pred_name,
                        "true": true_name,
                        "count": int(matrix[pred_idx, true_idx]),
                    })

        # Sort by count descending
        confusions.sort(key=lambda x: x["count"], reverse=True)
        return confusions[:top_n]

    def reset(self) -> None:
        """Reset all accumulated data."""
        self._predictions = []
        self._targets = []
        self.box = MetricStorage()
        self.confusion_matrix.reset()
        self.coco_converter.reset()
        self.nt_per_class = None

    def summary(self) -> List[Dict[str, Union[str, int, float]]]:
        """
        Generate per-class metrics summary.

        Returns:
            List of dicts with per-class metrics
        """
        results = []
        for i, (class_idx, class_name) in enumerate(self.names.items()):
            p = float(self.box.precision[i]) if i < len(self.box.precision) else 0.0
            r = float(self.box.recall[i]) if i < len(self.box.recall) else 0.0
            f1 = float(self.box.f1[i]) if i < len(self.box.f1) else 0.0
            ap50 = float(self.box.ap_per_class[i, 0]) if i < len(self.box.ap_per_class) else 0.0
            ap = float(self.box.ap_per_class[i].mean()) if i < len(self.box.ap_per_class) else 0.0
            nt = int(self.nt_per_class[class_idx]) if self.nt_per_class is not None else 0

            results.append({
                "Class": class_name,
                "Instances": nt,
                "P": round(p, 4),
                "R": round(r, 4),
                "F1": round(f1, 4),
                "mAP50": round(ap50, 4),
                "mAP50-95": round(ap, 4),
            })
        return results

    def to_csv(self) -> str:
        """Export metrics summary to CSV string."""
        summary = self.summary()
        if not summary:
            return "Class,Instances,P,R,F1,mAP50,mAP50-95\n"

        header = ",".join(summary[0].keys())
        lines = [header]
        for row in summary:
            lines.append(",".join(str(v) for v in row.values()))
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export metrics summary to JSON string."""
        return json.dumps(self.summary(), indent=2)

    @property
    def results_dict(self) -> Dict[str, float]:
        """Get metrics as dictionary for logging."""
        return {
            "metrics/precision": self.box.mp,
            "metrics/recall": self.box.mr,
            "metrics/f1": self.box.mf1,
            "metrics/mAP50": self.box.map50,
            "metrics/mAP75": self.box.map75,
            "metrics/mAP50-95": self.box.map,
        }
