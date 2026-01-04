"""
Unit tests for detection metrics module.

Tests cover:
- IoU calculation
- Confusion matrix
- DetMetrics aggregator
- Plot generation
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
    MetricStorage,
    compute_iou_matrix,
    gaussian_smooth,
    plot_f1_curve,
    plot_precision_curve,
    plot_precision_recall,
    plot_recall_curve,
)


# =============================================================================
# compute_iou_matrix Tests
# =============================================================================


class TestComputeIoU:
    """Tests for IoU computation."""

    def test_identical_boxes(self):
        """Identical boxes should have IoU of 1.0."""
        box = np.array([[0, 0, 10, 10]])
        iou = compute_iou_matrix(box, box)
        assert iou.shape == (1, 1)
        np.testing.assert_almost_equal(iou[0, 0], 1.0)

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU of 0.0."""
        box1 = np.array([[0, 0, 10, 10]])
        box2 = np.array([[20, 20, 30, 30]])
        iou = compute_iou_matrix(box1, box2)
        np.testing.assert_almost_equal(iou[0, 0], 0.0)

    def test_partial_overlap(self):
        """Partially overlapping boxes should have IoU between 0 and 1."""
        box1 = np.array([[0, 0, 10, 10]])
        box2 = np.array([[5, 5, 15, 15]])
        iou = compute_iou_matrix(box1, box2)
        # Intersection: 5x5=25, Union: 100+100-25=175
        expected = 25 / 175
        np.testing.assert_almost_equal(iou[0, 0], expected, decimal=4)

    def test_contained_box(self):
        """One box contained in another."""
        outer = np.array([[0, 0, 20, 20]])
        inner = np.array([[5, 5, 15, 15]])
        iou = compute_iou_matrix(outer, inner)
        # Intersection: 100, Union: 400
        np.testing.assert_almost_equal(iou[0, 0], 0.25, decimal=4)

    def test_multiple_boxes(self):
        """Test with multiple boxes."""
        boxes1 = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
        boxes2 = np.array([[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]])
        iou = compute_iou_matrix(boxes1, boxes2)
        assert iou.shape == (2, 3)
        np.testing.assert_almost_equal(iou[0, 0], 1.0)  # Identical
        np.testing.assert_almost_equal(iou[1, 2], 1.0)  # Identical

    def test_empty_input(self):
        """Empty inputs should return empty matrix."""
        empty = np.zeros((0, 4))
        box = np.array([[0, 0, 10, 10]])
        iou = compute_iou_matrix(empty, box)
        assert iou.shape == (0, 1)


# =============================================================================
# gaussian_smooth Tests
# =============================================================================


class TestGaussianSmooth:
    """Tests for Gaussian smoothing."""

    def test_constant_array(self):
        """Constant array should remain constant after smoothing."""
        arr = np.ones(100) * 5.0
        smoothed = gaussian_smooth(arr)
        np.testing.assert_array_almost_equal(smoothed, arr, decimal=5)

    def test_reduces_noise(self):
        """Smoothing should reduce noise."""
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        noisy = signal + np.random.normal(0, 0.3, 100)
        smoothed = gaussian_smooth(noisy, sigma=3)
        # Smoothed should be closer to original signal
        assert np.std(smoothed - signal) < np.std(noisy - signal)

    def test_output_shape(self):
        """Output shape should match input."""
        arr = np.random.rand(50)
        smoothed = gaussian_smooth(arr)
        assert smoothed.shape == arr.shape


# =============================================================================
# ConfusionMatrix Tests
# =============================================================================


class TestConfusionMatrix:
    """Tests for ConfusionMatrix class."""

    def test_initialization(self):
        """Test matrix initialization."""
        class_names = {0: "cat", 1: "dog", 2: "bird"}
        cm = ConfusionMatrix(num_classes=3, class_names=class_names)
        assert cm.num_classes == 3
        assert cm.matrix.shape == (4, 4)  # +1 for background
        assert cm.matrix.sum() == 0

    def test_perfect_detections(self):
        """Perfect detections should fill diagonal."""
        cm = ConfusionMatrix(num_classes=3)
        # Detection format: [x1, y1, x2, y2, conf, class]
        predictions = np.array([[0, 0, 10, 10, 0.9, 0]])
        # GT format: [class, x1, y1, x2, y2]
        ground_truth = np.array([[0, 0, 0, 10, 10]])
        cm.update(predictions, ground_truth)
        assert cm.matrix[0, 0] == 1  # True positive

    def test_false_positives(self):
        """Unmatched predictions are false positives."""
        cm = ConfusionMatrix(num_classes=3)
        predictions = np.array([[0, 0, 10, 10, 0.9, 1]])
        ground_truth = np.zeros((0, 5))
        cm.update(predictions, ground_truth)
        assert cm.matrix[1, 3] == 1  # FP (predicted class 1, no GT)

    def test_false_negatives(self):
        """Unmatched ground truths are false negatives."""
        cm = ConfusionMatrix(num_classes=3)
        predictions = np.zeros((0, 6))
        ground_truth = np.array([[2, 0, 0, 10, 10]])
        cm.update(predictions, ground_truth)
        assert cm.matrix[3, 2] == 1  # FN (no pred, GT class 2)

    def test_reset(self):
        """Reset should clear matrix."""
        cm = ConfusionMatrix(num_classes=3)
        cm.matrix[0, 0] = 5
        cm.reset()
        assert cm.matrix.sum() == 0

    def test_plot(self):
        """Test plot generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfusionMatrix(num_classes=3, class_names={0: "a", 1: "b", 2: "c"})
            cm.matrix[0, 0] = 10
            cm.matrix[1, 1] = 8
            save_path = Path(tmpdir) / "cm.png"
            cm.plot(save_path)
            assert save_path.exists()


# =============================================================================
# MetricStorage Tests
# =============================================================================


class TestMetricStorage:
    """Tests for MetricStorage class."""

    def test_initialization(self):
        """Test empty initialization."""
        m = MetricStorage()
        assert len(m.precision) == 0
        assert m.mp == 0.0
        assert m.mr == 0.0
        assert m.mf1 == 0.0

    def test_properties(self):
        """Test property calculations."""
        m = MetricStorage()
        m.precision = np.array([0.8, 0.9])
        m.recall = np.array([0.7, 0.8])
        m.f1 = np.array([0.75, 0.85])
        m.ap_per_class = np.array([[0.6, 0.5], [0.7, 0.6]])

        assert np.isclose(m.mp, 0.85)
        assert np.isclose(m.mr, 0.75)
        assert np.isclose(m.mf1, 0.8)
        assert np.isclose(m.map50, 0.65)  # Mean of first column


# =============================================================================
# DetMetrics Tests
# =============================================================================


class TestDetMetrics:
    """Tests for DetMetrics aggregator."""

    @pytest.fixture
    def class_names(self):
        return {0: "person", 1: "car", 2: "dog"}

    def test_initialization(self, class_names):
        """Test initialization."""
        dm = DetMetrics(names=class_names)
        assert dm.nc == 3
        assert len(dm._predictions) == 0

    def test_update(self, class_names):
        """Test update with predictions."""
        dm = DetMetrics(names=class_names)

        preds = [{
            "boxes": torch.tensor([[0, 0, 10, 10]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }]
        targets = [{
            "boxes": torch.tensor([[0, 0, 10, 10]]),
            "labels": torch.tensor([0]),
        }]

        dm.update(preds, targets)
        assert len(dm._predictions) == 1

    def test_empty_predictions(self, class_names):
        """Test with no predictions."""
        dm = DetMetrics(names=class_names)

        preds = [{
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.long),
        }]
        targets = [{
            "boxes": torch.tensor([[0, 0, 10, 10]]),
            "labels": torch.tensor([0]),
        }]

        dm.update(preds, targets)
        results = dm.process(plot=False)
        assert "map" in results

    def test_process_generates_plots(self, class_names):
        """Test that process generates plots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DetMetrics(names=class_names)

            preds = [{
                "boxes": torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]]),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 1]),
            }]
            targets = [{
                "boxes": torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]]),
                "labels": torch.tensor([0, 1]),
            }]

            dm.update(preds, targets)
            dm.process(save_dir=Path(tmpdir), plot=True)

            assert (Path(tmpdir) / "PR_curve.png").exists()
            assert (Path(tmpdir) / "F1_curve.png").exists()
            assert (Path(tmpdir) / "confusion_matrix.png").exists()

    def test_reset(self, class_names):
        """Test reset clears data."""
        dm = DetMetrics(names=class_names)

        preds = [{
            "boxes": torch.tensor([[0, 0, 10, 10]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }]
        targets = [{
            "boxes": torch.tensor([[0, 0, 10, 10]]),
            "labels": torch.tensor([0]),
        }]

        dm.update(preds, targets)
        dm.reset()
        assert len(dm._predictions) == 0
        assert len(dm._targets) == 0

    def test_summary(self, class_names):
        """Test summary generation."""
        dm = DetMetrics(names=class_names)

        preds = [{
            "boxes": torch.tensor([[0, 0, 10, 10]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }]
        targets = [{
            "boxes": torch.tensor([[0, 0, 10, 10]]),
            "labels": torch.tensor([0]),
        }]

        dm.update(preds, targets)
        dm.process(plot=False)

        summary = dm.summary()
        assert len(summary) == 3
        assert "Class" in summary[0]

    def test_to_csv(self, class_names):
        """Test CSV export."""
        dm = DetMetrics(names=class_names)
        dm.update(
            [{"boxes": torch.tensor([[0, 0, 10, 10]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}],
            [{"boxes": torch.tensor([[0, 0, 10, 10]]), "labels": torch.tensor([0])}]
        )
        dm.process(plot=False)
        csv = dm.to_csv()
        assert "Class" in csv
        assert "mAP50" in csv

    def test_to_json(self, class_names):
        """Test JSON export."""
        dm = DetMetrics(names=class_names)
        dm.update(
            [{"boxes": torch.tensor([[0, 0, 10, 10]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}],
            [{"boxes": torch.tensor([[0, 0, 10, 10]]), "labels": torch.tensor([0])}]
        )
        dm.process(plot=False)
        json_str = dm.to_json()
        data = json.loads(json_str)
        assert isinstance(data, list)

    def test_results_dict(self, class_names):
        """Test results dictionary."""
        dm = DetMetrics(names=class_names)
        dm.update(
            [{"boxes": torch.tensor([[0, 0, 10, 10]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}],
            [{"boxes": torch.tensor([[0, 0, 10, 10]]), "labels": torch.tensor([0])}]
        )
        dm.process(plot=False)
        results = dm.results_dict
        assert "metrics/precision" in results
        assert "metrics/mAP50" in results


# =============================================================================
# Plot Functions Tests
# =============================================================================


class TestPlotFunctions:
    """Tests for plot generation functions."""

    def test_plot_precision_recall(self):
        """Test PR curve generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precision = np.random.rand(3, 101)
            recall = np.random.rand(3, 101)
            ap = np.array([0.8, 0.7, 0.6])
            names = {0: "a", 1: "b", 2: "c"}

            save_path = Path(tmpdir) / "pr.png"
            plot_precision_recall(precision, recall, ap, names, save_path)
            assert save_path.exists()

    def test_plot_f1_curve(self):
        """Test F1 curve generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = np.random.rand(3, 101)
            conf = np.linspace(0, 1, 101)
            names = {0: "a", 1: "b", 2: "c"}

            save_path = Path(tmpdir) / "f1.png"
            plot_f1_curve(f1, conf, names, save_path)
            assert save_path.exists()

    def test_plot_precision_curve(self):
        """Test precision curve generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precision = np.random.rand(3, 101)
            conf = np.linspace(0, 1, 101)
            names = {0: "a", 1: "b", 2: "c"}

            save_path = Path(tmpdir) / "p.png"
            plot_precision_curve(precision, conf, names, save_path)
            assert save_path.exists()

    def test_plot_recall_curve(self):
        """Test recall curve generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recall = np.random.rand(3, 101)
            conf = np.linspace(0, 1, 101)
            names = {0: "a", 1: "b", 2: "c"}

            save_path = Path(tmpdir) / "r.png"
            plot_recall_curve(recall, conf, names, save_path)
            assert save_path.exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetricsIntegration:
    """Integration tests for the full metrics pipeline."""

    def test_full_pipeline(self):
        """Test complete metrics computation pipeline."""
        names = {0: "person", 1: "car"}
        dm = DetMetrics(names=names)

        # Simulate multiple batches
        for _ in range(3):
            preds = [{
                "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 1]),
            }]
            targets = [{
                "boxes": torch.tensor([[12, 12, 48, 48], [62, 62, 98, 98]]),
                "labels": torch.tensor([0, 1]),
            }]
            dm.update(preds, targets)

        results = dm.process(plot=False)

        assert "map" in results
        assert "map50" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results

    def test_no_predictions(self):
        """Test handling of no predictions."""
        names = {0: "person"}
        dm = DetMetrics(names=names)

        preds = [{
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.long),
        }]
        targets = [{
            "boxes": torch.tensor([[10, 10, 50, 50]]),
            "labels": torch.tensor([0]),
        }]

        dm.update(preds, targets)
        results = dm.process(plot=False)
        assert results["map"] == 0.0

    def test_no_ground_truth(self):
        """Test handling of no ground truth."""
        names = {0: "person"}
        dm = DetMetrics(names=names)

        preds = [{
            "boxes": torch.tensor([[10, 10, 50, 50]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }]
        targets = [{
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros(0, dtype=torch.long),
        }]

        dm.update(preds, targets)
        results = dm.process(plot=False)
        # With no GT, metrics should handle gracefully
        assert "map" in results
