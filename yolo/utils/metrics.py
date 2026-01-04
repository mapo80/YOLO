"""
Detection metrics module for YOLO models.

This module provides comprehensive metrics computation for object detection,
including precision, recall, F1 score, Average Precision (AP), and confusion matrix.
All metrics follow the COCO evaluation protocol with 101-point interpolation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def box_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Calculate Intersection over Union (IoU) between two sets of boxes.

    Args:
        box1: Array of boxes (N, 4) in xyxy format
        box2: Array of boxes (M, 4) in xyxy format
        eps: Small value to avoid division by zero

    Returns:
        IoU matrix of shape (N, M)
    """
    # Get coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0:1], box1[:, 1:2], box1[:, 2:3], box1[:, 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    return inter_area / (union_area + eps)


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """
    Apply box filter smoothing to a 1D array.

    Args:
        y: Input array
        f: Fraction of array length to use as filter size

    Returns:
        Smoothed array
    """
    nf = round(len(y) * f * 2) // 2 + 1  # Ensure odd filter size
    p = np.ones(nf) / nf
    return np.convolve(y, p, mode="same")


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Average Precision from recall-precision curve using 101-point interpolation.

    This follows the COCO evaluation protocol.

    Args:
        recall: Recall values at different thresholds
        precision: Precision values at different thresholds

    Returns:
        Tuple of (AP value, interpolated precision, interpolated recall)
    """
    # Add sentinel values at beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute precision envelope (monotonically decreasing)
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 101-point interpolation (COCO standard)
    x = np.linspace(0, 1, 101)
    # Use numpy.trapezoid for numpy >= 2.0, fall back to trapz for older versions
    try:
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    except AttributeError:
        ap = np.trapz(np.interp(x, mrec, mpre), x)

    return float(ap), mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    names: Dict[int, str],
    plot: bool = False,
    save_dir: Optional[Path] = None,
    eps: float = 1e-16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Average Precision per class at multiple IoU thresholds.

    Args:
        tp: True positives array (N,) or (N, num_iou_thresholds)
        conf: Confidence scores (N,)
        pred_cls: Predicted class indices (N,)
        target_cls: Ground truth class indices (M,)
        names: Dict mapping class indices to names
        plot: Whether to generate plots
        save_dir: Directory to save plots
        eps: Small value to avoid division by zero

    Returns:
        Tuple containing:
            - tp: True positives
            - fp: False positives
            - p: Precision per class
            - r: Recall per class
            - f1: F1 score per class
            - ap: AP per class at each IoU threshold (nc, num_iou)
            - unique_classes: Array of unique class indices
            - p_curve: Precision curves (nc, 1000)
            - r_curve: Recall curves (nc, 1000)
            - f1_curve: F1 curves (nc, 1000)
            - x: X-axis values for curves
    """
    # Sort by confidence (descending)
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes and count targets per class
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = len(unique_classes)

    # Handle case with multiple IoU thresholds
    if tp.ndim == 1:
        tp = tp[:, np.newaxis]
    num_iou = tp.shape[1]

    # Initialize arrays
    x = np.linspace(0, 1, 1000)  # X-axis for curves
    ap = np.zeros((nc, num_iou))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # Number of labels for this class
        n_p = i.sum()  # Number of predictions for this class

        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall curve
        recall = tpc / (n_l + eps)
        # Interpolate recall curve at x points (for plotting)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)

        # Precision curve
        precision = tpc / (tpc + fpc)
        # Interpolate precision curve at x points
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)

        # AP for each IoU threshold
        for j in range(num_iou):
            ap[ci, j], _, _ = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 curve
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

    # Find optimal confidence threshold (max mean F1)
    mean_f1 = smooth(f1_curve.mean(0), 0.1)
    i_best = mean_f1.argmax()

    # Extract metrics at optimal threshold
    p = p_curve[:, i_best]
    r = r_curve[:, i_best]
    f1 = f1_curve[:, i_best]

    # Calculate TP/FP at optimal threshold
    tp_sum = np.zeros(nc)
    fp_sum = np.zeros(nc)
    for ci, c in enumerate(unique_classes):
        mask = pred_cls == c
        conf_mask = conf[mask] >= x[i_best]
        tp_sum[ci] = tp[mask, 0][conf_mask].sum()
        fp_sum[ci] = (~tp[mask, 0].astype(bool))[conf_mask].sum()

    # Generate plots if requested
    if plot and save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        plot_pr_curve(x, p_curve, ap, save_dir / "PR_curve.png", names)
        plot_f1_curve(x, f1_curve, save_dir / "F1_curve.png", names)
        plot_p_curve(x, p_curve, save_dir / "P_curve.png", names)
        plot_r_curve(x, r_curve, save_dir / "R_curve.png", names)

    return tp_sum, fp_sum, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x


def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_path: Path,
    names: Dict[int, str],
) -> None:
    """
    Plot Precision-Recall curve.

    Args:
        px: X-axis values (recall)
        py: Precision values per class (nc, 1000)
        ap: AP values per class (nc, num_iou)
        save_path: Path to save the plot
        names: Dict mapping class indices to names
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    # Plot individual class curves if not too many
    if 0 < len(names) <= 20:
        for i, (name, precision) in enumerate(zip(names.values(), py)):
            ax.plot(px, precision, linewidth=1, label=f"{name} {ap[i, 0]:.3f}")
    else:
        ax.plot(px, py.T, linewidth=1, color="grey", alpha=0.3)

    # Plot mean curve
    mean_precision = py.mean(0)
    mean_ap = ap[:, 0].mean()
    ax.plot(px, mean_precision, linewidth=3, color="blue", label=f"all classes {mean_ap:.3f} mAP@0.5")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


def plot_f1_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_path: Path,
    names: Dict[int, str],
) -> None:
    """
    Plot F1-Confidence curve.

    Args:
        px: X-axis values (confidence)
        py: F1 values per class (nc, 1000)
        save_path: Path to save the plot
        names: Dict mapping class indices to names
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) <= 20:
        for i, (name, f1) in enumerate(zip(names.values(), py)):
            ax.plot(px, f1, linewidth=1, label=name)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey", alpha=0.3)

    # Plot mean curve
    mean_f1 = smooth(py.mean(0), 0.1)
    ax.plot(
        px,
        mean_f1,
        linewidth=3,
        color="blue",
        label=f"all classes {mean_f1.max():.2f} at {px[mean_f1.argmax()]:.3f}",
    )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("F1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("F1-Confidence Curve")
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


def plot_p_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_path: Path,
    names: Dict[int, str],
) -> None:
    """
    Plot Precision-Confidence curve.

    Args:
        px: X-axis values (confidence)
        py: Precision values per class (nc, 1000)
        save_path: Path to save the plot
        names: Dict mapping class indices to names
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) <= 20:
        for i, (name, precision) in enumerate(zip(names.values(), py)):
            ax.plot(px, precision, linewidth=1, label=name)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey", alpha=0.3)

    mean_precision = py.mean(0)
    ax.plot(px, mean_precision, linewidth=3, color="blue", label="all classes")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Confidence Curve")
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


def plot_r_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_path: Path,
    names: Dict[int, str],
) -> None:
    """
    Plot Recall-Confidence curve.

    Args:
        px: X-axis values (confidence)
        py: Recall values per class (nc, 1000)
        save_path: Path to save the plot
        names: Dict mapping class indices to names
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) <= 20:
        for i, (name, recall) in enumerate(zip(names.values(), py)):
            ax.plot(px, recall, linewidth=1, label=name)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey", alpha=0.3)

    mean_recall = py.mean(0)
    ax.plot(px, mean_recall, linewidth=3, color="blue", label="all classes")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Recall-Confidence Curve")
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


class ConfusionMatrix:
    """
    Confusion matrix for object detection evaluation.

    The matrix has shape (nc+1, nc+1) where the last row/column represents
    background (unmatched predictions/ground truths).
    """

    def __init__(self, nc: int, names: Optional[Dict[int, str]] = None):
        """
        Initialize confusion matrix.

        Args:
            nc: Number of classes
            names: Optional dict mapping class indices to names
        """
        self.nc = nc
        self.names = names or {i: str(i) for i in range(nc)}
        self.matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)

    def process_batch(
        self,
        detections: np.ndarray,
        labels: np.ndarray,
        iou_threshold: float = 0.45,
    ) -> None:
        """
        Update confusion matrix for a batch of detections and labels.

        Args:
            detections: Array of detections (N, 6) with [x1, y1, x2, y2, conf, cls]
            labels: Array of ground truth labels (M, 5) with [cls, x1, y1, x2, y2]
            iou_threshold: IoU threshold for matching
        """
        if len(labels) == 0:
            # All detections are false positives
            if len(detections) > 0:
                for det in detections:
                    pred_cls = int(det[5])
                    self.matrix[pred_cls, self.nc] += 1  # FP: pred vs background
            return

        if len(detections) == 0:
            # All labels are false negatives
            for label in labels:
                gt_cls = int(label[0])
                self.matrix[self.nc, gt_cls] += 1  # FN: background vs GT
            return

        # Extract boxes and classes
        det_boxes = detections[:, :4]
        det_classes = detections[:, 5].astype(int)
        gt_boxes = labels[:, 1:5]
        gt_classes = labels[:, 0].astype(int)

        # Compute IoU matrix
        iou = box_iou(det_boxes, gt_boxes)

        # Find matches above threshold
        matches_idx = np.where(iou > iou_threshold)
        if matches_idx[0].shape[0]:
            # Stack matches with IoU values
            matches = np.stack(
                [matches_idx[0], matches_idx[1], iou[matches_idx[0], matches_idx[1]]],
                axis=1,
            )
            # Sort by IoU descending
            matches = matches[matches[:, 2].argsort()[::-1]]

            # Remove duplicate detections (keep best match for each detection)
            _, unique_det_idx = np.unique(matches[:, 0], return_index=True)
            matches = matches[unique_det_idx]

            # Remove duplicate labels (keep best match for each label)
            _, unique_gt_idx = np.unique(matches[:, 1], return_index=True)
            matches = matches[unique_gt_idx]
        else:
            matches = np.zeros((0, 3))

        # Track which detections and labels are matched
        matched_dets = set(matches[:, 0].astype(int))
        matched_gts = set(matches[:, 1].astype(int))

        # Process matches
        for det_idx, gt_idx, _ in matches:
            det_idx, gt_idx = int(det_idx), int(gt_idx)
            pred_cls = det_classes[det_idx]
            gt_cls = gt_classes[gt_idx]
            self.matrix[pred_cls, gt_cls] += 1

        # Unmatched detections -> False Positives
        for det_idx in range(len(detections)):
            if det_idx not in matched_dets:
                pred_cls = det_classes[det_idx]
                self.matrix[pred_cls, self.nc] += 1

        # Unmatched labels -> False Negatives
        for gt_idx in range(len(labels)):
            if gt_idx not in matched_gts:
                gt_cls = gt_classes[gt_idx]
                self.matrix[self.nc, gt_cls] += 1

    def tp_fp(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract true positives and false positives from the matrix.

        Returns:
            Tuple of (tp, fp) arrays of shape (nc,)
        """
        tp = self.matrix.diagonal()[:-1]  # Exclude background
        fp = self.matrix.sum(1)[:-1] - tp
        return tp, fp

    def plot(
        self,
        save_path: Path,
        normalize: bool = True,
        title: str = "Confusion Matrix",
    ) -> None:
        """
        Plot and save the confusion matrix.

        Args:
            save_path: Path to save the plot
            normalize: Whether to normalize by column (ground truth)
            title: Plot title
        """
        save_path = Path(save_path)

        # Prepare matrix for plotting
        matrix = self.matrix.copy().astype(float)
        if normalize:
            # Normalize by column (ground truth totals)
            col_sums = matrix.sum(0, keepdims=True)
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            matrix = matrix / col_sums

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), tight_layout=True)

        # Create labels
        labels = list(self.names.values()) + ["background"]
        n = len(labels)

        # Plot matrix
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        # Add text annotations
        thresh = matrix.max() / 2.0
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                text = f"{value:.2f}" if normalize else f"{int(self.matrix[i, j])}"
                color = "white" if value > thresh else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.set_title(title)

        fig.savefig(save_path, dpi=250)
        plt.close(fig)

    def reset(self) -> None:
        """Reset the confusion matrix to zeros."""
        self.matrix = np.zeros((self.nc + 1, self.nc + 1), dtype=np.int64)

    def to_csv(self, decimals: int = 5) -> str:
        """
        Export confusion matrix to CSV string.

        Args:
            decimals: Number of decimal places for normalized values

        Returns:
            CSV formatted string
        """
        labels = list(self.names.values()) + ["background"]
        lines = ["Predicted," + ",".join(labels)]

        for i, label in enumerate(labels):
            values = [f"{self.matrix[i, j]}" for j in range(len(labels))]
            lines.append(f"{label}," + ",".join(values))

        return "\n".join(lines)


class Metric:
    """
    Container for per-class detection metrics.

    Stores precision, recall, F1, and AP values for each class,
    along with curve data for visualization.
    """

    def __init__(self):
        """Initialize empty metric container."""
        self.p = np.array([])  # Precision per class
        self.r = np.array([])  # Recall per class
        self.f1 = np.array([])  # F1 score per class
        self.all_ap = np.array([])  # AP per class at all IoU thresholds (nc, 10)
        self.ap_class_index = np.array([])  # Class indices with data
        self.nc = 0  # Number of classes

        # Curve data for plotting
        self.p_curve = np.array([])  # Precision curves (nc, 1000)
        self.r_curve = np.array([])  # Recall curves (nc, 1000)
        self.f1_curve = np.array([])  # F1 curves (nc, 1000)
        self.x = np.array([])  # X-axis values (1000,)

    def update(
        self,
        results: Tuple[
            np.ndarray,  # p
            np.ndarray,  # r
            np.ndarray,  # f1
            np.ndarray,  # ap
            np.ndarray,  # unique_classes
            np.ndarray,  # p_curve
            np.ndarray,  # r_curve
            np.ndarray,  # f1_curve
            np.ndarray,  # x
        ],
    ) -> None:
        """
        Update metrics with computed results.

        Args:
            results: Tuple from ap_per_class containing
                (p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x)
        """
        p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x = results
        self.p = p
        self.r = r
        self.f1 = f1
        self.all_ap = ap
        self.ap_class_index = unique_classes
        self.nc = len(unique_classes) if len(unique_classes) > 0 else 0
        self.p_curve = p_curve
        self.r_curve = r_curve
        self.f1_curve = f1_curve
        self.x = x

    @property
    def ap50(self) -> np.ndarray:
        """AP at IoU=0.5 for all classes."""
        return self.all_ap[:, 0] if len(self.all_ap) else np.array([])

    @property
    def ap75(self) -> np.ndarray:
        """AP at IoU=0.75 for all classes."""
        return self.all_ap[:, 5] if len(self.all_ap) else np.array([])

    @property
    def ap(self) -> np.ndarray:
        """AP averaged across IoU thresholds (0.5:0.95) for all classes."""
        return self.all_ap.mean(1) if len(self.all_ap) else np.array([])

    @property
    def mp(self) -> float:
        """Mean precision across all classes."""
        return float(self.p.mean()) if len(self.p) else 0.0

    @property
    def mr(self) -> float:
        """Mean recall across all classes."""
        return float(self.r.mean()) if len(self.r) else 0.0

    @property
    def mf1(self) -> float:
        """Mean F1 score across all classes."""
        return float(self.f1.mean()) if len(self.f1) else 0.0

    @property
    def map50(self) -> float:
        """Mean AP at IoU=0.5 across all classes."""
        return float(self.ap50.mean()) if len(self.ap50) else 0.0

    @property
    def map75(self) -> float:
        """Mean AP at IoU=0.75 across all classes."""
        return float(self.ap75.mean()) if len(self.ap75) else 0.0

    @property
    def map(self) -> float:
        """Mean AP across all classes and IoU thresholds (0.5:0.95)."""
        return float(self.all_ap.mean()) if len(self.all_ap) else 0.0

    def class_result(self, i: int) -> Tuple[float, float, float, float, float]:
        """
        Get metrics for a specific class.

        Args:
            i: Class index (in ap_class_index order)

        Returns:
            Tuple of (precision, recall, f1, ap50, ap)
        """
        return (
            float(self.p[i]) if i < len(self.p) else 0.0,
            float(self.r[i]) if i < len(self.r) else 0.0,
            float(self.f1[i]) if i < len(self.f1) else 0.0,
            float(self.ap50[i]) if i < len(self.ap50) else 0.0,
            float(self.ap[i]) if i < len(self.ap) else 0.0,
        )

    def mean_results(self) -> Tuple[float, float, float, float, float]:
        """
        Get mean metrics across all classes.

        Returns:
            Tuple of (mean_precision, mean_recall, mean_f1, map50, map)
        """
        return self.mp, self.mr, self.mf1, self.map50, self.map


class DetMetrics:
    """
    Complete detection metrics aggregator.

    Collects predictions and ground truths during validation,
    then computes comprehensive metrics including per-class AP,
    confusion matrix, and various curves.
    """

    def __init__(self, names: Dict[int, str], iou_thresholds: Optional[np.ndarray] = None):
        """
        Initialize detection metrics.

        Args:
            names: Dict mapping class indices to names
            iou_thresholds: IoU thresholds for AP calculation (default: 0.5:0.95:0.05)
        """
        self.names = names
        self.nc = len(names)
        self.iou_thresholds = iou_thresholds if iou_thresholds is not None else np.linspace(0.5, 0.95, 10)

        # Metric containers
        self.box = Metric()
        self.confusion_matrix = ConfusionMatrix(self.nc, names)

        # Statistics accumulator
        self.stats: Dict[str, List] = {
            "tp": [],
            "conf": [],
            "pred_cls": [],
            "target_cls": [],
        }

        # Per-class target counts
        self.nt_per_class: Optional[np.ndarray] = None

    def update(
        self,
        preds: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Update metrics with a batch of predictions and targets.

        Args:
            preds: List of prediction dicts with keys 'boxes', 'scores', 'labels'
            targets: List of target dicts with keys 'boxes', 'labels'
        """
        for pred, target in zip(preds, targets):
            # Extract prediction data
            pred_boxes = pred["boxes"].cpu().numpy() if len(pred["boxes"]) else np.zeros((0, 4))
            pred_scores = pred["scores"].cpu().numpy() if len(pred["scores"]) else np.zeros(0)
            pred_labels = pred["labels"].cpu().numpy() if len(pred["labels"]) else np.zeros(0)

            # Extract target data
            gt_boxes = target["boxes"].cpu().numpy() if len(target["boxes"]) else np.zeros((0, 4))
            gt_labels = target["labels"].cpu().numpy() if len(target["labels"]) else np.zeros(0)

            # Skip if no predictions and no targets
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue

            # Update confusion matrix
            if len(pred_boxes) > 0:
                detections = np.concatenate([
                    pred_boxes,
                    pred_scores[:, np.newaxis],
                    pred_labels[:, np.newaxis],
                ], axis=1)
            else:
                detections = np.zeros((0, 6))

            if len(gt_boxes) > 0:
                labels = np.concatenate([
                    gt_labels[:, np.newaxis],
                    gt_boxes,
                ], axis=1)
            else:
                labels = np.zeros((0, 5))

            self.confusion_matrix.process_batch(detections, labels)

            # Record target classes
            self.stats["target_cls"].append(gt_labels)

            # Compute TP/FP for each prediction at all IoU thresholds
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                iou = box_iou(pred_boxes, gt_boxes)

                # For each IoU threshold, determine TP/FP
                tp_all = np.zeros((len(pred_boxes), len(self.iou_thresholds)))

                for t_idx, iou_thresh in enumerate(self.iou_thresholds):
                    # Match predictions to ground truths
                    for p_idx in range(len(pred_boxes)):
                        # Find GT with same class and highest IoU above threshold
                        same_class = gt_labels == pred_labels[p_idx]
                        if not same_class.any():
                            continue

                        iou_vals = iou[p_idx] * same_class
                        best_gt = iou_vals.argmax()

                        if iou_vals[best_gt] >= iou_thresh:
                            tp_all[p_idx, t_idx] = 1

                self.stats["tp"].append(tp_all)
                self.stats["conf"].append(pred_scores)
                self.stats["pred_cls"].append(pred_labels)

            elif len(pred_boxes) > 0:
                # All predictions are FP (no ground truth)
                tp_all = np.zeros((len(pred_boxes), len(self.iou_thresholds)))
                self.stats["tp"].append(tp_all)
                self.stats["conf"].append(pred_scores)
                self.stats["pred_cls"].append(pred_labels)

    def process(
        self,
        save_dir: Optional[Path] = None,
        plot: bool = True,
    ) -> Dict[str, float]:
        """
        Process accumulated statistics and compute final metrics.

        Args:
            save_dir: Directory to save plots
            plot: Whether to generate plots

        Returns:
            Dict of computed metrics
        """
        # Concatenate all statistics
        if not self.stats["tp"]:
            # No predictions were made
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "map50": 0.0,
                "map75": 0.0,
                "map": 0.0,
            }

        tp = np.concatenate(self.stats["tp"], axis=0)
        conf = np.concatenate(self.stats["conf"], axis=0)
        pred_cls = np.concatenate(self.stats["pred_cls"], axis=0)
        target_cls = np.concatenate(self.stats["target_cls"], axis=0)

        # Count targets per class
        self.nt_per_class = np.bincount(target_cls.astype(int), minlength=self.nc)

        # Compute AP per class
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            self.names,
            plot=plot,
            save_dir=save_dir,
        )

        # Update metric container (skip tp_sum and fp_sum)
        self.box.update(results[2:])  # p, r, f1, ap, unique_classes, curves...

        # Generate confusion matrix plot
        if plot and save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.confusion_matrix.plot(save_dir / "confusion_matrix.png")

        return {
            "precision": self.box.mp,
            "recall": self.box.mr,
            "f1": self.box.mf1,
            "map50": self.box.map50,
            "map75": self.box.map75,
            "map": self.box.map,
        }

    def reset(self) -> None:
        """Reset all statistics and metrics."""
        self.stats = {
            "tp": [],
            "conf": [],
            "pred_cls": [],
            "target_cls": [],
        }
        self.box = Metric()
        self.confusion_matrix.reset()
        self.nt_per_class = None

    def summary(self) -> List[Dict[str, Union[str, int, float]]]:
        """
        Generate per-class summary.

        Returns:
            List of dicts with per-class metrics
        """
        results = []
        for i, cls_idx in enumerate(self.box.ap_class_index):
            p, r, f1, ap50, ap = self.box.class_result(i)
            nt = int(self.nt_per_class[cls_idx]) if self.nt_per_class is not None else 0
            results.append({
                "Class": self.names.get(int(cls_idx), str(cls_idx)),
                "Instances": nt,
                "P": round(p, 4),
                "R": round(r, 4),
                "F1": round(f1, 4),
                "mAP50": round(ap50, 4),
                "mAP50-95": round(ap, 4),
            })
        return results

    def to_csv(self) -> str:
        """
        Export metrics summary to CSV string.

        Returns:
            CSV formatted string
        """
        summary = self.summary()
        if not summary:
            return "Class,Instances,P,R,F1,mAP50,mAP50-95\n"

        header = ",".join(summary[0].keys())
        lines = [header]
        for row in summary:
            lines.append(",".join(str(v) for v in row.values()))
        return "\n".join(lines)

    def to_json(self) -> str:
        """
        Export metrics summary to JSON string.

        Returns:
            JSON formatted string
        """
        import json
        return json.dumps(self.summary(), indent=2)

    @property
    def results_dict(self) -> Dict[str, float]:
        """
        Get metrics as a dictionary for logging.

        Returns:
            Dict with metric names and values
        """
        return {
            "metrics/precision": self.box.mp,
            "metrics/recall": self.box.mr,
            "metrics/f1": self.box.mf1,
            "metrics/mAP50": self.box.map50,
            "metrics/mAP75": self.box.map75,
            "metrics/mAP50-95": self.box.map,
        }
