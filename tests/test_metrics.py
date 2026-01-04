"""
Comprehensive unit tests for detection metrics module.

Tests cover all functionality including:
- IoU calculation
- Average Precision computation
- Confusion matrix
- DetMetrics aggregator
- Metric container
- Plot generation
- CSV/JSON export
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from yolo.utils.metrics import (
    ConfusionMatrix,
    DetMetrics,
    Metric,
    ap_per_class,
    box_iou,
    compute_ap,
    plot_f1_curve,
    plot_p_curve,
    plot_pr_curve,
    plot_r_curve,
    smooth,
)


# =============================================================================
# box_iou Tests
# =============================================================================


class TestBoxIoU:
    """Tests for IoU calculation."""

    def test_identical_boxes(self):
        """IoU of identical boxes should be 1.0."""
        boxes = np.array([[0, 0, 100, 100]])
        iou = box_iou(boxes, boxes)
        assert iou.shape == (1, 1)
        np.testing.assert_almost_equal(iou[0, 0], 1.0)

    def test_no_overlap(self):
        """IoU of non-overlapping boxes should be 0.0."""
        box1 = np.array([[0, 0, 50, 50]])
        box2 = np.array([[100, 100, 150, 150]])
        iou = box_iou(box1, box2)
        assert iou.shape == (1, 1)
        np.testing.assert_almost_equal(iou[0, 0], 0.0)

    def test_partial_overlap(self):
        """IoU of partially overlapping boxes."""
        box1 = np.array([[0, 0, 100, 100]])
        box2 = np.array([[50, 50, 150, 150]])
        iou = box_iou(box1, box2)
        # Intersection: 50x50 = 2500, Union: 2*10000 - 2500 = 17500
        expected = 2500 / 17500
        np.testing.assert_almost_equal(iou[0, 0], expected, decimal=4)

    def test_contained_box(self):
        """IoU when one box contains another."""
        box1 = np.array([[0, 0, 100, 100]])
        box2 = np.array([[25, 25, 75, 75]])
        iou = box_iou(box1, box2)
        # Intersection: 50x50 = 2500, Union: 10000 (larger box)
        expected = 2500 / 10000
        np.testing.assert_almost_equal(iou[0, 0], expected, decimal=4)

    def test_multiple_boxes(self):
        """IoU between multiple boxes."""
        box1 = np.array([
            [0, 0, 100, 100],
            [200, 200, 300, 300],
        ])
        box2 = np.array([
            [0, 0, 100, 100],
            [50, 50, 150, 150],
            [200, 200, 300, 300],
        ])
        iou = box_iou(box1, box2)
        assert iou.shape == (2, 3)
        # box1[0] vs box2[0]: 1.0
        np.testing.assert_almost_equal(iou[0, 0], 1.0)
        # box1[1] vs box2[2]: 1.0
        np.testing.assert_almost_equal(iou[1, 2], 1.0)
        # box1[1] vs box2[0]: 0.0
        np.testing.assert_almost_equal(iou[1, 0], 0.0)

    def test_edge_touching_boxes(self):
        """IoU of boxes touching at edges (no overlap)."""
        box1 = np.array([[0, 0, 50, 50]])
        box2 = np.array([[50, 0, 100, 50]])
        iou = box_iou(box1, box2)
        np.testing.assert_almost_equal(iou[0, 0], 0.0)

    def test_empty_input(self):
        """Handle empty input arrays."""
        box1 = np.array([[0, 0, 100, 100]])
        box2 = np.zeros((0, 4))
        iou = box_iou(box1, box2)
        assert iou.shape == (1, 0)


# =============================================================================
# smooth Tests
# =============================================================================


class TestSmooth:
    """Tests for smoothing function."""

    def test_constant_array(self):
        """Smoothing constant array should remain mostly constant (with edge effects)."""
        y = np.ones(100)
        smoothed = smooth(y)
        # Interior values should remain constant (ignore edge effects)
        np.testing.assert_array_almost_equal(smoothed[10:-10], y[10:-10])

    def test_reduces_noise(self):
        """Smoothing should reduce variance."""
        np.random.seed(42)
        y = np.random.randn(100)
        smoothed = smooth(y)
        assert smoothed.std() <= y.std()

    def test_preserves_mean(self):
        """Smoothing should approximately preserve mean."""
        np.random.seed(42)
        y = np.random.randn(100) + 5
        smoothed = smooth(y)
        # Mean should be close (edge effects may cause slight difference)
        np.testing.assert_almost_equal(smoothed.mean(), y.mean(), decimal=0)

    def test_output_shape(self):
        """Output should have same shape as input."""
        y = np.random.randn(50)
        smoothed = smooth(y)
        assert smoothed.shape == y.shape


# =============================================================================
# compute_ap Tests
# =============================================================================


class TestComputeAP:
    """Tests for Average Precision computation."""

    def test_perfect_precision(self):
        """AP should be 1.0 for perfect precision."""
        recall = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        precision = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        ap, _, _ = compute_ap(recall, precision)
        np.testing.assert_almost_equal(ap, 1.0, decimal=2)

    def test_zero_precision(self):
        """AP should be 0.0 for zero precision."""
        recall = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        precision = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        ap, _, _ = compute_ap(recall, precision)
        np.testing.assert_almost_equal(ap, 0.0, decimal=2)

    def test_decreasing_precision(self):
        """AP with monotonically decreasing precision."""
        recall = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        precision = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        ap, _, _ = compute_ap(recall, precision)
        # Should be less than 1.0 but greater than 0
        assert 0.0 < ap < 1.0

    def test_returns_interpolated_values(self):
        """Should return interpolated precision and recall."""
        recall = np.array([0.0, 0.5, 1.0])
        precision = np.array([1.0, 0.8, 0.6])
        ap, mpre, mrec = compute_ap(recall, precision)
        assert len(mpre) > len(precision)  # Interpolated
        assert len(mrec) > len(recall)

    def test_empty_input(self):
        """Handle empty input."""
        recall = np.array([])
        precision = np.array([])
        ap, mpre, mrec = compute_ap(recall, precision)
        assert isinstance(ap, float)


# =============================================================================
# ap_per_class Tests
# =============================================================================


class TestAPPerClass:
    """Tests for per-class AP computation."""

    def test_single_class_perfect(self):
        """Perfect predictions for single class."""
        tp = np.array([1, 1, 1, 1, 1])  # All true positives
        conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        pred_cls = np.array([0, 0, 0, 0, 0])
        target_cls = np.array([0, 0, 0, 0, 0])
        names = {0: "test"}

        results = ap_per_class(tp, conf, pred_cls, target_cls, names, plot=False)
        tp_sum, fp_sum, p, r, f1, ap, unique, p_curve, r_curve, f1_curve, x = results

        # Should have high AP
        assert ap[0, 0] > 0.9

    def test_multiple_classes(self):
        """Compute metrics for multiple classes."""
        tp = np.array([1, 1, 0, 1, 0, 1])
        conf = np.array([0.9, 0.85, 0.8, 0.7, 0.6, 0.5])
        pred_cls = np.array([0, 1, 0, 1, 2, 2])
        target_cls = np.array([0, 0, 1, 1, 2, 2])
        names = {0: "cat", 1: "dog", 2: "bird"}

        results = ap_per_class(tp, conf, pred_cls, target_cls, names, plot=False)
        _, _, p, r, f1, ap, unique_classes, _, _, _, _ = results

        assert len(unique_classes) == 3
        assert len(p) == 3
        assert len(r) == 3
        assert len(f1) == 3

    def test_missing_class(self):
        """Handle class present in ground truth but not predictions."""
        tp = np.array([1, 1])
        conf = np.array([0.9, 0.8])
        pred_cls = np.array([0, 0])
        target_cls = np.array([0, 0, 1])  # Class 1 has no predictions
        names = {0: "cat", 1: "dog"}

        results = ap_per_class(tp, conf, pred_cls, target_cls, names, plot=False)
        _, _, p, r, f1, ap, unique_classes, _, _, _, _ = results

        assert 1 in unique_classes  # Class 1 should be in unique classes

    def test_plot_generation(self, temp_dir):
        """Test that plots are generated when requested."""
        tp = np.array([1, 0, 1, 0])
        conf = np.array([0.9, 0.8, 0.7, 0.6])
        pred_cls = np.array([0, 0, 1, 1])
        target_cls = np.array([0, 0, 1, 1])
        names = {0: "cat", 1: "dog"}

        ap_per_class(tp, conf, pred_cls, target_cls, names, plot=True, save_dir=temp_dir)

        # Check plots were generated
        assert (temp_dir / "PR_curve.png").exists()
        assert (temp_dir / "F1_curve.png").exists()
        assert (temp_dir / "P_curve.png").exists()
        assert (temp_dir / "R_curve.png").exists()


# =============================================================================
# ConfusionMatrix Tests
# =============================================================================


class TestConfusionMatrix:
    """Tests for confusion matrix."""

    def test_initialization(self, class_names):
        """Test proper initialization."""
        cm = ConfusionMatrix(nc=5, names=class_names)
        assert cm.nc == 5
        assert cm.matrix.shape == (6, 6)  # nc + 1 for background
        assert cm.matrix.sum() == 0

    def test_perfect_detections(self):
        """All predictions match ground truth."""
        cm = ConfusionMatrix(nc=3)

        # Perfect match: pred class 0 matches GT class 0
        detections = np.array([[100, 100, 200, 200, 0.9, 0]])
        labels = np.array([[0, 100, 100, 200, 200]])

        cm.process_batch(detections, labels)

        # TP for class 0
        assert cm.matrix[0, 0] == 1
        assert cm.matrix.sum() == 1

    def test_false_positives(self):
        """Predictions with no matching ground truth."""
        cm = ConfusionMatrix(nc=3)

        # Detection with no GT
        detections = np.array([[100, 100, 200, 200, 0.9, 0]])
        labels = np.zeros((0, 5))

        cm.process_batch(detections, labels)

        # FP: pred class 0 vs background
        assert cm.matrix[0, 3] == 1  # pred vs background

    def test_false_negatives(self):
        """Ground truth with no predictions."""
        cm = ConfusionMatrix(nc=3)

        # GT with no detections
        detections = np.zeros((0, 6))
        labels = np.array([[0, 100, 100, 200, 200]])

        cm.process_batch(detections, labels)

        # FN: background vs GT class 0
        assert cm.matrix[3, 0] == 1

    def test_wrong_class_prediction(self):
        """Prediction matches location but wrong class."""
        cm = ConfusionMatrix(nc=3)

        # Pred class 1, GT class 0 at same location
        detections = np.array([[100, 100, 200, 200, 0.9, 1]])
        labels = np.array([[0, 100, 100, 200, 200]])

        cm.process_batch(detections, labels)

        # Confusion: pred class 1 vs GT class 0
        assert cm.matrix[1, 0] == 1

    def test_multiple_detections(self):
        """Multiple detections and labels in one batch."""
        cm = ConfusionMatrix(nc=3)

        detections = np.array([
            [100, 100, 200, 200, 0.9, 0],  # TP class 0
            [300, 300, 400, 400, 0.8, 1],  # TP class 1
            [500, 500, 600, 600, 0.7, 2],  # FP class 2
        ])
        labels = np.array([
            [0, 100, 100, 200, 200],  # GT class 0
            [1, 300, 300, 400, 400],  # GT class 1
        ])

        cm.process_batch(detections, labels)

        assert cm.matrix[0, 0] == 1  # TP class 0
        assert cm.matrix[1, 1] == 1  # TP class 1
        assert cm.matrix[2, 3] == 1  # FP class 2 vs background

    def test_tp_fp_extraction(self):
        """Test TP/FP extraction method."""
        cm = ConfusionMatrix(nc=2)

        # Set up matrix manually
        cm.matrix[0, 0] = 5  # TP class 0
        cm.matrix[0, 1] = 2  # Confusion: pred 0 vs GT 1
        cm.matrix[1, 1] = 3  # TP class 1
        cm.matrix[0, 2] = 1  # FP class 0 vs background

        tp, fp = cm.tp_fp()

        assert tp[0] == 5
        assert tp[1] == 3
        assert fp[0] == 3  # 2 confusion + 1 background
        assert fp[1] == 0

    def test_reset(self, class_names):
        """Test matrix reset."""
        cm = ConfusionMatrix(nc=5, names=class_names)
        cm.matrix[0, 0] = 10
        cm.reset()
        assert cm.matrix.sum() == 0

    def test_plot(self, temp_dir, class_names):
        """Test confusion matrix plotting."""
        cm = ConfusionMatrix(nc=5, names=class_names)

        # Add some data
        cm.matrix[0, 0] = 10
        cm.matrix[1, 1] = 8
        cm.matrix[0, 1] = 2
        cm.matrix[1, 0] = 3

        save_path = temp_dir / "confusion_matrix.png"
        cm.plot(save_path, normalize=True)

        assert save_path.exists()

    def test_plot_unnormalized(self, temp_dir, class_names):
        """Test unnormalized confusion matrix plot."""
        cm = ConfusionMatrix(nc=5, names=class_names)
        cm.matrix[0, 0] = 10

        save_path = temp_dir / "confusion_matrix_raw.png"
        cm.plot(save_path, normalize=False)

        assert save_path.exists()

    def test_to_csv(self, class_names):
        """Test CSV export."""
        cm = ConfusionMatrix(nc=5, names=class_names)
        cm.matrix[0, 0] = 10
        cm.matrix[1, 1] = 5

        csv = cm.to_csv()

        assert "cat" in csv
        assert "dog" in csv
        assert "background" in csv
        assert "10" in csv
        assert "5" in csv


# =============================================================================
# Metric Tests
# =============================================================================


class TestMetric:
    """Tests for Metric container."""

    def test_initialization(self):
        """Test empty initialization."""
        m = Metric()
        assert len(m.p) == 0
        assert len(m.r) == 0
        assert len(m.f1) == 0
        assert m.nc == 0

    def test_update(self):
        """Test updating with results."""
        m = Metric()

        # Simulate results from ap_per_class
        p = np.array([0.9, 0.8])
        r = np.array([0.85, 0.75])
        f1 = 2 * p * r / (p + r)
        ap = np.array([[0.9, 0.85, 0.8], [0.8, 0.75, 0.7]])  # (nc, num_iou)
        unique_classes = np.array([0, 1])
        p_curve = np.random.rand(2, 1000)
        r_curve = np.random.rand(2, 1000)
        f1_curve = np.random.rand(2, 1000)
        x = np.linspace(0, 1, 1000)

        results = (p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x)
        m.update(results)

        assert m.nc == 2
        np.testing.assert_array_equal(m.p, p)
        np.testing.assert_array_equal(m.r, r)

    def test_ap_properties(self):
        """Test AP property accessors."""
        m = Metric()

        # AP at different IoU thresholds
        # Shape (nc, num_iou) where num_iou = 10 for 0.5:0.95:0.05
        ap = np.array([
            [0.9, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72],
            [0.8, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64, 0.62],
        ])

        m.all_ap = ap
        m.p = np.array([0.9, 0.8])
        m.r = np.array([0.85, 0.75])
        m.f1 = np.array([0.87, 0.77])

        # AP50 should be first column
        np.testing.assert_array_equal(m.ap50, ap[:, 0])

        # AP75 should be 6th column (index 5)
        np.testing.assert_array_equal(m.ap75, ap[:, 5])

        # mAP should be mean across all
        np.testing.assert_almost_equal(m.map, ap.mean())

        # Mean metrics
        np.testing.assert_almost_equal(m.mp, 0.85)
        np.testing.assert_almost_equal(m.mr, 0.80)

    def test_empty_properties(self):
        """Properties should return 0 for empty metrics."""
        m = Metric()

        assert m.map == 0.0
        assert m.map50 == 0.0
        assert m.map75 == 0.0
        assert m.mp == 0.0
        assert m.mr == 0.0
        assert m.mf1 == 0.0

    def test_class_result(self):
        """Test per-class result retrieval."""
        m = Metric()
        m.p = np.array([0.9, 0.8, 0.7])
        m.r = np.array([0.85, 0.75, 0.65])
        m.f1 = np.array([0.87, 0.77, 0.67])
        m.all_ap = np.array([
            [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45],
            [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35],
            [0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25],
        ])

        p, r, f1, ap50, ap = m.class_result(0)

        assert p == 0.9
        assert r == 0.85
        assert f1 == 0.87
        assert ap50 == 0.9
        np.testing.assert_almost_equal(ap, m.all_ap[0].mean())

    def test_mean_results(self):
        """Test mean results tuple."""
        m = Metric()
        m.p = np.array([0.9, 0.8])
        m.r = np.array([0.85, 0.75])
        m.f1 = np.array([0.87, 0.77])
        m.all_ap = np.array([
            [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45],
            [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35],
        ])

        mp, mr, mf1, map50, map_val = m.mean_results()

        np.testing.assert_almost_equal(mp, 0.85)
        np.testing.assert_almost_equal(mr, 0.80)


# =============================================================================
# DetMetrics Tests
# =============================================================================


class TestDetMetrics:
    """Tests for DetMetrics aggregator."""

    def test_initialization(self, class_names):
        """Test proper initialization."""
        dm = DetMetrics(names=class_names)

        assert dm.nc == 5
        assert len(dm.stats["tp"]) == 0
        assert isinstance(dm.confusion_matrix, ConfusionMatrix)
        assert isinstance(dm.box, Metric)

    def test_update_single_batch(
        self, class_names, sample_predictions, sample_ground_truth
    ):
        """Test updating with a single batch."""
        dm = DetMetrics(names=class_names)
        dm.update(sample_predictions, sample_ground_truth)

        assert len(dm.stats["tp"]) > 0
        assert len(dm.stats["conf"]) > 0
        assert len(dm.stats["pred_cls"]) > 0
        assert len(dm.stats["target_cls"]) > 0

    def test_update_multiple_batches(
        self, class_names, multi_batch_predictions, multi_batch_ground_truth
    ):
        """Test updating with multiple batches."""
        dm = DetMetrics(names=class_names)

        for preds, gt in zip(multi_batch_predictions, multi_batch_ground_truth):
            dm.update([preds], [gt])

        assert len(dm.stats["tp"]) == 5  # 5 batches

    def test_perfect_predictions(
        self, class_names, perfect_predictions, perfect_ground_truth
    ):
        """Test with perfect predictions."""
        dm = DetMetrics(names=class_names)
        dm.update(perfect_predictions, perfect_ground_truth)
        results = dm.process(plot=False)

        # Should have high precision and recall
        assert results["precision"] > 0.9
        assert results["recall"] > 0.9
        assert results["map50"] > 0.9

    def test_empty_predictions(
        self, class_names, empty_predictions, sample_ground_truth
    ):
        """Test with empty predictions (all FN)."""
        dm = DetMetrics(names=class_names)
        dm.update(empty_predictions, sample_ground_truth)
        results = dm.process(plot=False)

        # No predictions means 0 precision/recall
        assert results["precision"] == 0.0
        assert results["recall"] == 0.0

    def test_empty_ground_truth(
        self, class_names, sample_predictions, empty_ground_truth
    ):
        """Test with empty ground truth (all FP)."""
        dm = DetMetrics(names=class_names)
        dm.update(sample_predictions, empty_ground_truth)
        results = dm.process(plot=False)

        # All predictions are FP
        assert results["precision"] == 0.0

    def test_process_generates_plots(self, class_names, temp_dir):
        """Test that process generates plots when requested."""
        dm = DetMetrics(names=class_names)

        # Add some data
        preds = [{
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "scores": torch.tensor([0.9], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }]
        targets = [{
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }]

        dm.update(preds, targets)
        dm.process(save_dir=temp_dir, plot=True)

        # Check confusion matrix was saved
        assert (temp_dir / "confusion_matrix.png").exists()

    def test_reset(self, class_names, sample_predictions, sample_ground_truth):
        """Test reset functionality."""
        dm = DetMetrics(names=class_names)
        dm.update(sample_predictions, sample_ground_truth)
        dm.process(plot=False)

        dm.reset()

        assert len(dm.stats["tp"]) == 0
        assert dm.confusion_matrix.matrix.sum() == 0

    def test_summary(self, class_names):
        """Test summary generation."""
        dm = DetMetrics(names=class_names)

        # Add data for class 0 and 1
        preds = [{
            "boxes": torch.tensor([
                [100, 100, 200, 200],
                [300, 300, 400, 400],
            ], dtype=torch.float32),
            "scores": torch.tensor([0.9, 0.85], dtype=torch.float32),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }]
        targets = [{
            "boxes": torch.tensor([
                [100, 100, 200, 200],
                [300, 300, 400, 400],
            ], dtype=torch.float32),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }]

        dm.update(preds, targets)
        dm.process(plot=False)

        summary = dm.summary()

        assert len(summary) > 0
        assert "Class" in summary[0]
        assert "P" in summary[0]
        assert "R" in summary[0]
        assert "mAP50" in summary[0]

    def test_to_csv(self, class_names):
        """Test CSV export."""
        dm = DetMetrics(names=class_names)

        preds = [{
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "scores": torch.tensor([0.9], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }]
        targets = [{
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }]

        dm.update(preds, targets)
        dm.process(plot=False)

        csv = dm.to_csv()

        assert "Class" in csv
        assert "P" in csv
        assert "R" in csv
        assert "mAP50" in csv

    def test_to_json(self, class_names):
        """Test JSON export."""
        dm = DetMetrics(names=class_names)

        preds = [{
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "scores": torch.tensor([0.9], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }]
        targets = [{
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }]

        dm.update(preds, targets)
        dm.process(plot=False)

        json_str = dm.to_json()
        data = json.loads(json_str)

        assert isinstance(data, list)
        if len(data) > 0:
            assert "Class" in data[0]

    def test_results_dict(self, class_names):
        """Test results_dict property for logging."""
        dm = DetMetrics(names=class_names)

        preds = [{
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "scores": torch.tensor([0.9], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }]
        targets = [{
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }]

        dm.update(preds, targets)
        dm.process(plot=False)

        results = dm.results_dict

        assert "metrics/precision" in results
        assert "metrics/recall" in results
        assert "metrics/mAP50" in results
        assert "metrics/mAP50-95" in results

    def test_custom_iou_thresholds(self, class_names):
        """Test with custom IoU thresholds."""
        custom_thresholds = np.array([0.5, 0.6, 0.7])
        dm = DetMetrics(names=class_names, iou_thresholds=custom_thresholds)

        assert len(dm.iou_thresholds) == 3
        np.testing.assert_array_equal(dm.iou_thresholds, custom_thresholds)

    def test_overlapping_predictions(
        self, class_names, overlapping_boxes_predictions, overlapping_boxes_ground_truth
    ):
        """Test handling of overlapping predictions."""
        dm = DetMetrics(names=class_names)
        dm.update(overlapping_boxes_predictions, overlapping_boxes_ground_truth)
        results = dm.process(plot=False)

        # Should handle overlapping boxes gracefully
        assert "precision" in results
        assert "recall" in results


# =============================================================================
# Plot Functions Tests
# =============================================================================


class TestPlotFunctions:
    """Tests for plot generation functions."""

    @pytest.fixture
    def curve_data(self):
        """Generate sample curve data for plotting."""
        nc = 3
        x = np.linspace(0, 1, 1000)
        p_curve = np.random.rand(nc, 1000)
        r_curve = np.random.rand(nc, 1000)
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-16)
        ap = np.random.rand(nc, 10)
        names = {0: "cat", 1: "dog", 2: "bird"}
        return x, p_curve, r_curve, f1_curve, ap, names

    def test_plot_pr_curve(self, temp_dir, curve_data):
        """Test PR curve plotting."""
        x, p_curve, r_curve, f1_curve, ap, names = curve_data
        save_path = temp_dir / "pr_curve.png"

        plot_pr_curve(x, p_curve, ap, save_path, names)

        assert save_path.exists()

    def test_plot_f1_curve(self, temp_dir, curve_data):
        """Test F1 curve plotting."""
        x, p_curve, r_curve, f1_curve, ap, names = curve_data
        save_path = temp_dir / "f1_curve.png"

        plot_f1_curve(x, f1_curve, save_path, names)

        assert save_path.exists()

    def test_plot_p_curve(self, temp_dir, curve_data):
        """Test Precision curve plotting."""
        x, p_curve, r_curve, f1_curve, ap, names = curve_data
        save_path = temp_dir / "p_curve.png"

        plot_p_curve(x, p_curve, save_path, names)

        assert save_path.exists()

    def test_plot_r_curve(self, temp_dir, curve_data):
        """Test Recall curve plotting."""
        x, p_curve, r_curve, f1_curve, ap, names = curve_data
        save_path = temp_dir / "r_curve.png"

        plot_r_curve(x, r_curve, save_path, names)

        assert save_path.exists()

    def test_plot_many_classes(self, temp_dir):
        """Test plotting with many classes (>20)."""
        nc = 50
        x = np.linspace(0, 1, 1000)
        p_curve = np.random.rand(nc, 1000)
        ap = np.random.rand(nc, 10)
        names = {i: f"class_{i}" for i in range(nc)}

        save_path = temp_dir / "pr_many_classes.png"
        plot_pr_curve(x, p_curve, ap, save_path, names)

        assert save_path.exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetricsIntegration:
    """Integration tests for the complete metrics pipeline."""

    def test_full_pipeline(self, class_names, temp_dir):
        """Test complete metrics computation pipeline."""
        dm = DetMetrics(names=class_names)

        # Simulate multiple validation batches
        for i in range(10):
            preds = [{
                "boxes": torch.tensor([
                    [100 + i * 5, 100, 200, 200],
                    [300 + i * 5, 300, 400, 400],
                ], dtype=torch.float32),
                "scores": torch.tensor([0.9 - i * 0.05, 0.8 - i * 0.03], dtype=torch.float32),
                "labels": torch.tensor([i % 5, (i + 1) % 5], dtype=torch.long),
            }]
            targets = [{
                "boxes": torch.tensor([
                    [100 + i * 5, 100, 200, 200],
                    [300 + i * 5, 300, 400, 400],
                ], dtype=torch.float32),
                "labels": torch.tensor([i % 5, (i + 1) % 5], dtype=torch.long),
            }]
            dm.update(preds, targets)

        # Process and generate all outputs
        results = dm.process(save_dir=temp_dir, plot=True)

        # Verify results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "map50" in results
        assert "map75" in results
        assert "map" in results

        # Verify files
        assert (temp_dir / "confusion_matrix.png").exists()

        # Verify summary
        summary = dm.summary()
        assert len(summary) > 0

        # Verify CSV
        csv = dm.to_csv()
        assert len(csv) > 0

        # Verify JSON
        json_str = dm.to_json()
        data = json.loads(json_str)
        assert isinstance(data, list)

    def test_edge_case_no_predictions(self, class_names):
        """Test edge case: validation with no predictions."""
        dm = DetMetrics(names=class_names)

        # All empty batches
        for _ in range(5):
            dm.update(
                [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}],
                [{"boxes": torch.tensor([[100, 100, 200, 200]]), "labels": torch.tensor([0])}],
            )

        results = dm.process(plot=False)

        assert results["precision"] == 0.0
        assert results["recall"] == 0.0
        assert results["map50"] == 0.0

    def test_edge_case_no_ground_truth(self, class_names):
        """Test edge case: validation with no ground truth."""
        dm = DetMetrics(names=class_names)

        # All empty GT batches
        for _ in range(5):
            dm.update(
                [{"boxes": torch.tensor([[100, 100, 200, 200]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}],
                [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}],
            )

        results = dm.process(plot=False)

        # All FPs, so precision should be 0
        assert results["precision"] == 0.0

    def test_reset_and_reuse(self, class_names):
        """Test resetting and reusing metrics."""
        dm = DetMetrics(names=class_names)

        # First run
        preds = [{"boxes": torch.tensor([[100, 100, 200, 200]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}]
        targets = [{"boxes": torch.tensor([[100, 100, 200, 200]]), "labels": torch.tensor([0])}]
        dm.update(preds, targets)
        results1 = dm.process(plot=False)

        # Reset
        dm.reset()

        # Second run with different data
        preds = [{"boxes": torch.tensor([[300, 300, 400, 400]]), "scores": torch.tensor([0.8]), "labels": torch.tensor([1])}]
        targets = [{"boxes": torch.tensor([[300, 300, 400, 400]]), "labels": torch.tensor([1])}]
        dm.update(preds, targets)
        results2 = dm.process(plot=False)

        # Results should be independent
        assert isinstance(results1, dict)
        assert isinstance(results2, dict)


# =============================================================================
# Training-Experiment Dataset Integration Test
# =============================================================================


@pytest.mark.skipif(
    not Path(__file__).resolve().parent.parent.joinpath("data", "training-experiment").exists(),
    reason="training-experiment dataset not available",
)
class TestTrainingExperimentDataset:
    """Integration tests with training-experiment dataset."""

    def test_metrics_with_real_data(self, training_experiment_path):
        """Test metrics computation with real dataset format."""
        # This test verifies that our metrics work with real data format
        # without actually loading the full dataset

        # Create metrics with expected classes
        names = {0: "class_0", 1: "class_1"}  # Placeholder
        dm = DetMetrics(names=names)

        # Simulate predictions that would come from real inference
        preds = [{
            "boxes": torch.tensor([[50, 50, 150, 150], [200, 200, 300, 300]], dtype=torch.float32),
            "scores": torch.tensor([0.85, 0.72], dtype=torch.float32),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }]
        targets = [{
            "boxes": torch.tensor([[55, 55, 145, 145], [205, 205, 295, 295]], dtype=torch.float32),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }]

        dm.update(preds, targets)
        results = dm.process(plot=False)

        # Verify we get reasonable results
        assert results["precision"] > 0
        assert results["recall"] > 0
        assert results["map50"] > 0

    def test_dataset_path_exists(self, training_experiment_path, training_experiment_exists):
        """Verify training-experiment dataset path."""
        assert training_experiment_exists
        assert training_experiment_path.exists()
