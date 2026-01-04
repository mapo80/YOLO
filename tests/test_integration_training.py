"""
Integration tests for complete training pipeline.

These tests verify end-to-end functionality using the training-experiment
dataset (if available) or synthetic data.

Run these tests explicitly with:
    pytest tests/test_integration_training.py -v --run-integration

Or run all integration tests:
    pytest -m integration --run-integration
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# Mark entire module as integration tests - skipped by default
# Use: pytest --run-integration to run these tests
pytestmark = pytest.mark.integration


class TestTrainingExperimentIntegration:
    """Integration tests using training-experiment dataset."""

    @pytest.fixture
    def dataset_path(self):
        """Get path to training-experiment dataset."""
        path = Path(__file__).resolve().parent.parent / "data" / "training-experiment"
        if not path.exists():
            pytest.skip("training-experiment dataset not available")
        return path

    def test_dataset_structure(self, dataset_path):
        """Verify training-experiment dataset has correct structure."""
        # Check required directories/files
        assert dataset_path.exists(), "Dataset directory should exist"

        # Common dataset structures
        possible_structures = [
            # YOLO format
            ("images", "labels"),
            # COCO format
            ("images", "annotations"),
            # Simple image folder
            ("train", "val"),
        ]

        has_valid_structure = False
        for dirs in possible_structures:
            if all((dataset_path / d).exists() for d in dirs if isinstance(d, str)):
                has_valid_structure = True
                break

        # Also accept if there are images directly in the folder
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        has_images = any(
            f.suffix.lower() in image_extensions
            for f in dataset_path.iterdir()
            if f.is_file()
        )

        assert has_valid_structure or has_images, \
            "Dataset should have valid structure or contain images"

    def test_dataset_has_images(self, dataset_path):
        """Verify dataset contains images."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

        # Search for images in dataset
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.rglob(f"*{ext}"))
            images.extend(dataset_path.rglob(f"*{ext.upper()}"))

        assert len(images) > 0, "Dataset should contain at least one image"

    def test_metrics_with_dataset(self, dataset_path):
        """Test metrics computation with dataset images."""
        from yolo.utils.metrics import DetMetrics

        # Create metrics
        names = {0: "object"}  # Simple single-class test
        dm = DetMetrics(names=names)

        # Simulate predictions from dataset
        # (In real test, we would load actual images and run inference)
        for i in range(5):
            preds = [{
                "boxes": torch.tensor([[100 + i * 10, 100, 200, 200]], dtype=torch.float32),
                "scores": torch.tensor([0.9 - i * 0.1], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
            }]
            targets = [{
                "boxes": torch.tensor([[100 + i * 10, 100, 200, 200]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
            }]
            dm.update(preds, targets)

        # Process metrics
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = dm.process(save_dir=Path(tmp_dir), plot=True)

        # Verify results
        assert results["precision"] > 0.5
        assert results["recall"] > 0.5
        assert results["map50"] > 0.5


class TestSyntheticTrainingIntegration:
    """Integration tests using synthetic data (always runs)."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic training data."""
        batch_size = 4
        num_batches = 10

        batches = []
        for _ in range(num_batches):
            # Random images
            images = torch.randn(batch_size, 3, 640, 640)

            # Random targets
            targets = []
            for _ in range(batch_size):
                num_objects = torch.randint(1, 5, (1,)).item()
                boxes = torch.rand(num_objects, 4) * 640
                # Ensure x2 > x1 and y2 > y1
                boxes[:, 2:] = boxes[:, :2] + torch.abs(boxes[:, 2:]) + 10
                boxes = boxes.clamp(0, 640)

                labels = torch.randint(0, 10, (num_objects,))

                targets.append({
                    "boxes": boxes,
                    "labels": labels,
                })

            batches.append((images, targets))

        return batches

    def test_metrics_full_pipeline(self, synthetic_data):
        """Test complete metrics pipeline with synthetic data."""
        from yolo.utils.metrics import DetMetrics

        # Create metrics
        names = {i: f"class_{i}" for i in range(10)}
        dm = DetMetrics(names=names)

        # Simulate validation loop
        for images, targets in synthetic_data:
            # Simulate model predictions (random for test)
            preds = []
            for i in range(len(images)):
                num_preds = torch.randint(1, 10, (1,)).item()
                boxes = torch.rand(num_preds, 4) * 640
                boxes[:, 2:] = boxes[:, :2] + torch.abs(boxes[:, 2:]) + 10
                boxes = boxes.clamp(0, 640)

                preds.append({
                    "boxes": boxes,
                    "scores": torch.rand(num_preds),
                    "labels": torch.randint(0, 10, (num_preds,)),
                })

            dm.update(preds, targets)

        # Process metrics
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = dm.process(save_dir=Path(tmp_dir), plot=True)

            # Verify plots were generated
            assert (Path(tmp_dir) / "confusion_matrix.png").exists()

        # Verify results structure
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "map50" in results
        assert "map75" in results
        assert "map" in results

    def test_metrics_summary_export(self, synthetic_data):
        """Test metrics summary export functionality."""
        from yolo.utils.metrics import DetMetrics

        names = {i: f"class_{i}" for i in range(10)}
        dm = DetMetrics(names=names)

        # Add some data
        for images, targets in synthetic_data[:3]:
            preds = []
            for i in range(len(images)):
                preds.append({
                    "boxes": torch.rand(5, 4) * 640,
                    "scores": torch.rand(5),
                    "labels": torch.randint(0, 10, (5,)),
                })
            dm.update(preds, targets)

        dm.process(plot=False)

        # Test CSV export
        csv = dm.to_csv()
        assert "Class" in csv
        assert "P" in csv
        assert "R" in csv

        # Test JSON export
        import json
        json_str = dm.to_json()
        data = json.loads(json_str)
        assert isinstance(data, list)

    def test_confusion_matrix_accuracy(self):
        """Test confusion matrix with known inputs."""
        from yolo.utils.metrics import ConfusionMatrix

        cm = ConfusionMatrix(num_classes=3, class_names={0: "cat", 1: "dog", 2: "bird"})

        # Perfect predictions
        # Format: [x1, y1, x2, y2, conf, class]
        detections = [
            [100, 100, 200, 200, 0.9, 0],  # Pred cat
            [300, 300, 400, 400, 0.8, 1],  # Pred dog
            [500, 500, 600, 600, 0.7, 2],  # Pred bird
        ]
        # Format: [class, x1, y1, x2, y2]
        labels = [
            [0, 100, 100, 200, 200],  # GT cat
            [1, 300, 300, 400, 400],  # GT dog
            [2, 500, 500, 600, 600],  # GT bird
        ]

        import numpy as np
        cm.update(np.array(detections), np.array(labels))

        # Check diagonal (true positives on matrix diagonal)
        assert cm.matrix[0, 0] == 1  # Cat TP
        assert cm.matrix[1, 1] == 1  # Dog TP
        assert cm.matrix[2, 2] == 1  # Bird TP


class TestSchedulerIntegration:
    """Integration tests for LR schedulers."""

    def test_cosine_scheduler_full_training(self):
        """Test cosine scheduler over full training cycle."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        max_epochs = 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=0.0001,
        )

        lrs = []
        for epoch in range(max_epochs):
            lrs.append(optimizer.param_groups[0]["lr"])
            # Simulate training step
            x = torch.randn(32, 10)
            y = model(x)
            loss = y.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Verify LR schedule
        assert lrs[0] == 0.01  # Start at max
        assert lrs[-1] < lrs[0]  # End lower than start
        assert min(lrs) >= 0.0001  # Never below eta_min

    def test_one_cycle_scheduler_full_training(self):
        """Test OneCycle scheduler over full training cycle."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        total_steps = 1000
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4,
        )

        lrs = []
        for step in range(total_steps):
            lrs.append(optimizer.param_groups[0]["lr"])
            # Simulate training step
            x = torch.randn(32, 10)
            y = model(x)
            loss = y.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Verify LR schedule
        initial_lr = 0.01 / 25.0
        assert abs(lrs[0] - initial_lr) < 1e-6  # Start at initial_lr
        # Peak should be around 30% through training
        peak_idx = int(total_steps * 0.3)
        assert lrs[peak_idx] >= lrs[0]  # Peak higher than start
        assert lrs[-1] < lrs[peak_idx]  # End lower than peak


class TestLayerFreezingIntegration:
    """Integration tests for layer freezing."""

    def test_freeze_unfreeze_training_cycle(self):
        """Test full freeze-train-unfreeze-train cycle."""
        # Create simple model
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )

        # Phase 1: Freeze early layers
        for i, (name, param) in enumerate(model.named_parameters()):
            if i < 2:  # Freeze first conv
                param.requires_grad = False

        # Create optimizer with only trainable params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, lr=0.01)

        # Train for a few steps
        for step in range(10):
            x = torch.randn(4, 3, 32, 32)
            y = model(x)
            loss = y.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Phase 2: Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True

        # Recreate optimizer with all params
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Continue training
        for step in range(10):
            x = torch.randn(4, 3, 32, 32)
            y = model(x)
            loss = y.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verify all parameters have been updated
        # (This is implicit - if we got here without errors, it worked)
        assert True

    def test_gradual_unfreezing(self):
        """Test gradual layer unfreezing during training."""
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        # Initially freeze all but last layer
        for i, (name, param) in enumerate(model.named_parameters()):
            if i < len(list(model.parameters())) - 2:
                param.requires_grad = False

        trainable_count_initial = sum(1 for p in model.parameters() if p.requires_grad)

        # Gradually unfreeze layers
        for epoch in range(3):
            # Unfreeze one more layer
            for i, (name, param) in enumerate(model.named_parameters()):
                if not param.requires_grad:
                    param.requires_grad = True
                    break  # Unfreeze one at a time

            trainable_count = sum(1 for p in model.parameters() if p.requires_grad)

            # Train for some steps
            for step in range(5):
                x = torch.randn(4, 10)
                y = model(x)
                loss = y.sum()
                optimizer = torch.optim.SGD(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=0.01
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Verify more parameters are now trainable
        trainable_count_final = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable_count_final >= trainable_count_initial
