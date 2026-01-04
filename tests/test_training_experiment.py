"""
Integration tests for training-experiment dataset.

Tests all major features:
- COCO format dataset loading
- Metrics computation
- LR schedulers
- Layer freezing
- Model export
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Dataset path
DATASET_PATH = project_root / "training-experiment" / "simpsons-coco-std"
YOLO_DATASET_PATH = project_root / "training-experiment" / "simpsons-yolo"


# Skip tests if dataset not available
pytestmark = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason="training-experiment dataset not available"
)


class TestDatasetLoading:
    """Test dataset loading functionality."""

    def test_coco_dataset_exists(self):
        """Verify COCO format dataset exists."""
        assert DATASET_PATH.exists()
        assert (DATASET_PATH / "images").exists()
        assert (DATASET_PATH / "annotations").exists()

    def test_yolo_dataset_exists(self):
        """Verify YOLO format dataset exists."""
        assert YOLO_DATASET_PATH.exists()
        assert (YOLO_DATASET_PATH / "train").exists()
        assert (YOLO_DATASET_PATH / "valid").exists()

    def test_coco_annotations_valid(self):
        """Test that COCO annotations are valid JSON."""
        import json

        annotations_dir = DATASET_PATH / "annotations"
        for ann_file in annotations_dir.glob("*.json"):
            with open(ann_file) as f:
                data = json.load(f)
                assert "images" in data
                assert "annotations" in data
                assert "categories" in data

    def test_datamodule_setup(self):
        """Test YOLODataModule setup."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(
            root=str(DATASET_PATH),
            train_images="images/train",
            val_images="images/val",
            train_ann="annotations/instances_train.json",
            val_ann="annotations/instances_val.json",
            batch_size=2,
            num_workers=0,
            image_size=[320, 320],
        )
        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.val_dataset) > 0

    def test_datamodule_batch(self):
        """Test datamodule produces valid batches."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(
            root=str(DATASET_PATH),
            train_images="images/train",
            val_images="images/val",
            train_ann="annotations/instances_train.json",
            val_ann="annotations/instances_val.json",
            batch_size=2,
            num_workers=0,
            image_size=[320, 320],
            mosaic_prob=0.0,  # Disable mosaic for simpler testing
        )
        dm.setup(stage="fit")

        # Get a batch
        train_loader = dm.train_dataloader()
        images, targets = next(iter(train_loader))

        assert images.shape[0] == 2  # batch size
        assert images.shape[1] == 3  # RGB
        assert isinstance(targets, list)
        assert len(targets) == 2


class TestMetricsSystem:
    """Test metrics computation on dataset."""

    def test_metrics_with_simpsons_classes(self):
        """Test metrics with Simpsons character classes."""
        from yolo.utils.metrics import DetMetrics

        # Simpsons classes
        names = {
            0: "abraham_grampa_simpson",
            1: "bart_simpson",
            2: "homer_simpson",
            3: "lisa_simpson",
            4: "maggie_simpson",
            5: "marge_simpson",
            6: "ned_flanders",
        }

        dm = DetMetrics(names=names)

        # Simulate predictions
        for i in range(5):
            preds = [{
                "boxes": torch.tensor([[100 + i * 10, 100, 200, 200]], dtype=torch.float32),
                "scores": torch.tensor([0.9 - i * 0.1], dtype=torch.float32),
                "labels": torch.tensor([i % 7], dtype=torch.long),
            }]
            targets = [{
                "boxes": torch.tensor([[100 + i * 10, 100, 200, 200]], dtype=torch.float32),
                "labels": torch.tensor([i % 7], dtype=torch.long),
            }]
            dm.update(preds, targets)

        # Process metrics
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = dm.process(save_dir=Path(tmp_dir), plot=True)

            # Verify plots were generated
            assert (Path(tmp_dir) / "confusion_matrix.png").exists()
            assert (Path(tmp_dir) / "PR_curve.png").exists()

        # Verify metrics
        assert results["precision"] >= 0.0
        assert results["recall"] >= 0.0
        assert results["map50"] >= 0.0

    def test_confusion_matrix_7_classes(self):
        """Test confusion matrix with 7 Simpsons classes."""
        from yolo.utils.metrics import ConfusionMatrix

        names = {i: f"class_{i}" for i in range(7)}
        cm = ConfusionMatrix(num_classes=7, class_names=names)

        # Add some detections
        # Format: [x1, y1, x2, y2, conf, class]
        detections = np.array([
            [100, 100, 200, 200, 0.9, 0],  # Pred class 0
            [300, 300, 400, 400, 0.8, 1],  # Pred class 1
            [500, 500, 600, 600, 0.7, 2],  # Pred class 2
        ])
        # Format: [class, x1, y1, x2, y2]
        labels = np.array([
            [0, 100, 100, 200, 200],  # GT class 0
            [1, 300, 300, 400, 400],  # GT class 1
            [2, 500, 500, 600, 600],  # GT class 2
        ])

        cm.update(detections, labels)

        # Check diagonal has true positives (perfect detections)
        assert cm.matrix[0, 0] >= 1  # class 0 detected correctly
        assert cm.matrix[1, 1] >= 1  # class 1 detected correctly
        assert cm.matrix[2, 2] >= 1  # class 2 detected correctly


class TestLRSchedulers:
    """Test learning rate scheduler functionality."""

    def test_cosine_scheduler_config(self):
        """Test cosine scheduler configuration."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=0.0001,
        )

        # Run a few steps
        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(10):
            optimizer.step()
            scheduler.step()

        # LR should have changed
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_one_cycle_scheduler_config(self):
        """Test one cycle scheduler configuration."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            total_steps=100,
            pct_start=0.3,
        )

        # Run through schedule
        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # Should have warmed up then annealed
        assert max(lrs) > lrs[0]  # Peak higher than start
        assert lrs[-1] < max(lrs)  # End lower than peak


class TestLayerFreezing:
    """Test layer freezing for transfer learning."""

    def test_freeze_backbone_pattern(self):
        """Test freezing backbone layers."""
        # Create simple model with backbone-like naming
        class SimpleYOLO(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone_conv1 = nn.Conv2d(3, 64, 3)
                self.backbone_conv2 = nn.Conv2d(64, 128, 3)
                self.neck_conv = nn.Conv2d(128, 256, 3)
                self.head_conv = nn.Conv2d(256, 7, 1)  # 7 Simpsons classes

            def forward(self, x):
                x = self.backbone_conv1(x)
                x = self.backbone_conv2(x)
                x = self.neck_conv(x)
                return self.head_conv(x)

        model = SimpleYOLO()

        # Freeze backbone
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

        # Check backbone is frozen
        assert not model.backbone_conv1.weight.requires_grad
        assert not model.backbone_conv2.weight.requires_grad

        # Check head is trainable
        assert model.neck_conv.weight.requires_grad
        assert model.head_conv.weight.requires_grad

        # Verify gradient flow
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Backbone should have no gradients
        assert model.backbone_conv1.weight.grad is None

        # Head should have gradients
        assert model.head_conv.weight.grad is not None

    def test_epoch_based_unfreezing(self):
        """Test unfreezing after N epochs."""
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.Linear(64, 7),  # 7 Simpsons classes
        )

        # Freeze first layer
        model[0].weight.requires_grad = False
        freeze_until_epoch = 5

        # Simulate training
        for epoch in range(10):
            if epoch >= freeze_until_epoch:
                model[0].weight.requires_grad = True

            # Check state
            if epoch < freeze_until_epoch:
                assert not model[0].weight.requires_grad
            else:
                assert model[0].weight.requires_grad


class TestModelExport:
    """Test model export functionality."""

    def test_export_module_exists(self):
        """Test that export module is importable."""
        from yolo.tools import export
        assert hasattr(export, "export_onnx")
        assert hasattr(export, "export_tflite")

    def test_letterbox_function(self):
        """Test letterbox image preprocessing."""
        from yolo.tools.export import _letterbox_image
        from PIL import Image

        # Create test image
        img = Image.new("RGB", (800, 600), color="red")

        # Letterbox to square
        result = _letterbox_image(img, target_size=(640, 640))

        assert result.size == (640, 640)


class TestFullPipeline:
    """Integration tests for full training pipeline."""

    def test_config_loading(self):
        """Test loading training config."""
        import yaml

        config_path = project_root / "training-experiment" / "simpsons-train.yaml"
        assert config_path.exists()

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "trainer" in config
        assert "model" in config
        assert "data" in config
        assert config["model"]["num_classes"] == 7

    def test_model_creation_for_simpsons(self):
        """Test creating model with 7 classes for Simpsons."""
        from omegaconf import OmegaConf
        from yolo.model.yolo import create_model

        # Load v9-t config (small model for testing)
        config_path = project_root / "yolo" / "config" / "model" / "v9-t.yaml"
        if not config_path.exists():
            pytest.skip("v9-t.yaml not found")

        cfg = OmegaConf.load(config_path)
        model = create_model(cfg, weight_path=None, class_num=7)

        # Verify model works
        x = torch.randn(1, 3, 320, 320)
        output = model(x)

        assert "Main" in output
        assert len(output["Main"]) == 3  # 3 scales


class TestTrainingExperimentResults:
    """Test and document training experiment results."""

    @pytest.fixture
    def test_results_file(self):
        """Path to store test results."""
        return project_root / "training-experiment" / "TEST_RESULTS.md"

    def test_document_dataset_info(self):
        """Document dataset information."""
        import json

        # Load annotations to get stats
        train_ann = DATASET_PATH / "annotations" / "instances_train.json"
        val_ann = DATASET_PATH / "annotations" / "instances_val.json"

        with open(train_ann) as f:
            train_data = json.load(f)
        with open(val_ann) as f:
            val_data = json.load(f)

        # Dataset stats
        num_train_images = len(train_data["images"])
        num_val_images = len(val_data["images"])
        num_classes = len(train_data["categories"])
        class_names = [c["name"] for c in train_data["categories"]]

        print("\n=== Dataset Information ===")
        print(f"Training images: {num_train_images}")
        print(f"Validation images: {num_val_images}")
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {', '.join(class_names)}")

        # Verify expected structure
        assert num_train_images > 0
        assert num_val_images > 0
        # Dataset has 8 classes (including cartoon-persons background class)
        assert num_classes >= 7
