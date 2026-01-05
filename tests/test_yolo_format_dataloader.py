"""
Extensive tests for YOLO format dataloader.

Tests cover:
- YOLOFormatDataset initialization and loading
- Label parsing from .txt files
- Coordinate conversion (normalized xywh to xyxy pixel coordinates)
- Edge cases (empty labels, missing files, invalid data)
- Integration with MosaicMixupDataset
- YOLOFormatDataModule setup and dataloaders
- Augmentation pipeline
- Collate function
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import shutil

import pytest
import torch
import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Dataset paths
YOLO_DATASET_PATH = project_root / "training-experiment" / "simpsons-yolo"

# Skip tests if dataset not available
pytestmark = pytest.mark.skipif(
    not YOLO_DATASET_PATH.exists(),
    reason="YOLO format dataset not available"
)


class TestYOLOFormatDatasetBasic:
    """Basic tests for YOLOFormatDataset."""

    def test_dataset_initialization(self):
        """Test dataset initializes correctly."""
        from yolo.data.datamodule import YOLOFormatDataset

        dataset = YOLOFormatDataset(
            images_dir=str(YOLO_DATASET_PATH / "train" / "images"),
            labels_dir=str(YOLO_DATASET_PATH / "train" / "labels"),
            image_size=(640, 640),
        )

        assert len(dataset) > 0
        assert dataset.images_dir == YOLO_DATASET_PATH / "train" / "images"
        assert dataset.labels_dir == YOLO_DATASET_PATH / "train" / "labels"

    def test_dataset_length(self):
        """Test dataset returns correct length."""
        from yolo.data.datamodule import YOLOFormatDataset

        train_dataset = YOLOFormatDataset(
            images_dir=str(YOLO_DATASET_PATH / "train" / "images"),
            labels_dir=str(YOLO_DATASET_PATH / "train" / "labels"),
        )
        val_dataset = YOLOFormatDataset(
            images_dir=str(YOLO_DATASET_PATH / "valid" / "images"),
            labels_dir=str(YOLO_DATASET_PATH / "valid" / "labels"),
        )

        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        print(f"\nTrain dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

    def test_getitem_returns_image_and_target(self):
        """Test __getitem__ returns (image, target) tuple."""
        from yolo.data.datamodule import YOLOFormatDataset

        dataset = YOLOFormatDataset(
            images_dir=str(YOLO_DATASET_PATH / "train" / "images"),
            labels_dir=str(YOLO_DATASET_PATH / "train" / "labels"),
        )

        image, target = dataset[0]

        # Image should be PIL Image (no transforms)
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

        # Target should be dict with boxes and labels
        assert isinstance(target, dict)
        assert "boxes" in target
        assert "labels" in target

    def test_target_format(self):
        """Test target dict has correct tensor shapes."""
        from yolo.data.datamodule import YOLOFormatDataset

        dataset = YOLOFormatDataset(
            images_dir=str(YOLO_DATASET_PATH / "train" / "images"),
            labels_dir=str(YOLO_DATASET_PATH / "train" / "labels"),
        )

        # Find an image with boxes
        for i in range(min(10, len(dataset))):
            image, target = dataset[i]
            if len(target["boxes"]) > 0:
                break

        assert target["boxes"].dtype == torch.float32
        assert target["labels"].dtype == torch.long
        assert target["boxes"].ndim == 2
        assert target["boxes"].shape[1] == 4  # xyxy format
        assert target["labels"].ndim == 1
        assert len(target["boxes"]) == len(target["labels"])


class TestYOLOLabelParsing:
    """Tests for YOLO format label parsing."""

    def test_label_file_parsing(self):
        """Test parsing of YOLO format .txt label files."""
        from yolo.data.datamodule import YOLOFormatDataset

        dataset = YOLOFormatDataset(
            images_dir=str(YOLO_DATASET_PATH / "train" / "images"),
            labels_dir=str(YOLO_DATASET_PATH / "train" / "labels"),
        )

        # Get first sample and verify labels parsed correctly
        image, target = dataset[0]
        num_boxes = len(target["boxes"])

        # Read corresponding label file manually
        image_path = dataset.image_files[0]
        label_path = dataset.labels_dir / (image_path.stem + ".txt")

        if label_path.exists():
            with open(label_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            assert num_boxes == len(lines), f"Expected {len(lines)} boxes, got {num_boxes}"

    def test_coordinate_conversion_xywh_to_xyxy(self):
        """Test normalized xywh to xyxy pixel coordinate conversion."""
        from yolo.data.datamodule import YOLOFormatDataset

        dataset = YOLOFormatDataset(
            images_dir=str(YOLO_DATASET_PATH / "train" / "images"),
            labels_dir=str(YOLO_DATASET_PATH / "train" / "labels"),
        )

        image, target = dataset[0]
        img_w, img_h = image.size

        if len(target["boxes"]) > 0:
            box = target["boxes"][0]
            x1, y1, x2, y2 = box.tolist()

            # Verify coordinates are in pixel space (not normalized)
            assert x2 > x1, "x2 should be > x1"
            assert y2 > y1, "y2 should be > y1"

            # Verify coordinates are within image bounds
            assert x1 >= 0 and x1 <= img_w
            assert y1 >= 0 and y1 <= img_h
            assert x2 >= 0 and x2 <= img_w
            assert y2 >= 0 and y2 <= img_h

    def test_class_ids_in_valid_range(self):
        """Test that all class IDs are valid (0-6 for Simpsons dataset)."""
        from yolo.data.datamodule import YOLOFormatDataset

        dataset = YOLOFormatDataset(
            images_dir=str(YOLO_DATASET_PATH / "train" / "images"),
            labels_dir=str(YOLO_DATASET_PATH / "train" / "labels"),
        )

        all_labels = []
        for i in range(min(50, len(dataset))):
            _, target = dataset[i]
            all_labels.extend(target["labels"].tolist())

        assert len(all_labels) > 0, "Should have some labels"
        assert min(all_labels) >= 0, "Class IDs should be >= 0"
        assert max(all_labels) <= 6, "Class IDs should be <= 6 (7 classes)"

        # Print class distribution
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"\nClass distribution in first 50 samples:")
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count}")


class TestYOLOFormatEdgeCases:
    """Tests for edge cases in YOLO format loading."""

    def test_empty_label_file(self):
        """Test handling of empty label files (images with no objects)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create test image
            images_dir = tmp_path / "images"
            labels_dir = tmp_path / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()

            img = Image.new("RGB", (640, 480), color="blue")
            img.save(images_dir / "test.jpg")

            # Create empty label file
            (labels_dir / "test.txt").write_text("")

            from yolo.data.datamodule import YOLOFormatDataset

            dataset = YOLOFormatDataset(
                images_dir=str(images_dir),
                labels_dir=str(labels_dir),
            )

            image, target = dataset[0]

            # Should return empty boxes and labels
            assert len(target["boxes"]) == 0
            assert len(target["labels"]) == 0
            assert target["boxes"].shape == (0, 4)

    def test_missing_label_file(self):
        """Test handling of missing label files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create test image only (no label file)
            images_dir = tmp_path / "images"
            labels_dir = tmp_path / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()

            img = Image.new("RGB", (640, 480), color="green")
            img.save(images_dir / "test.jpg")

            # Don't create label file

            from yolo.data.datamodule import YOLOFormatDataset

            dataset = YOLOFormatDataset(
                images_dir=str(images_dir),
                labels_dir=str(labels_dir),
            )

            image, target = dataset[0]

            # Should return empty boxes and labels
            assert len(target["boxes"]) == 0
            assert len(target["labels"]) == 0

    def test_invalid_label_line_handling(self):
        """Test handling of invalid lines in label files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            images_dir = tmp_path / "images"
            labels_dir = tmp_path / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()

            img = Image.new("RGB", (640, 480), color="red")
            img.save(images_dir / "test.jpg")

            # Create label with mix of valid and invalid lines
            label_content = """0 0.5 0.5 0.2 0.2
invalid line here
1 0.3 0.3 0.1 0.1
too few values
"""
            (labels_dir / "test.txt").write_text(label_content)

            from yolo.data.datamodule import YOLOFormatDataset

            dataset = YOLOFormatDataset(
                images_dir=str(images_dir),
                labels_dir=str(labels_dir),
            )

            image, target = dataset[0]

            # Should only parse valid lines (2 boxes)
            assert len(target["boxes"]) == 2
            assert len(target["labels"]) == 2

    def test_no_images_raises_error(self):
        """Test that empty images directory raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            images_dir = tmp_path / "images"
            labels_dir = tmp_path / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()

            from yolo.data.datamodule import YOLOFormatDataset

            with pytest.raises(ValueError, match="No images found"):
                YOLOFormatDataset(
                    images_dir=str(images_dir),
                    labels_dir=str(labels_dir),
                )

    def test_box_clipping_to_image_bounds(self):
        """Test that boxes are clipped to image boundaries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            images_dir = tmp_path / "images"
            labels_dir = tmp_path / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()

            img = Image.new("RGB", (640, 480), color="white")
            img.save(images_dir / "test.jpg")

            # Create label with box partially outside image (center near edge)
            # normalized coords: center at 0.95, 0.95 with width/height 0.2
            # This would extend beyond image bounds
            label_content = "0 0.95 0.95 0.2 0.2\n"
            (labels_dir / "test.txt").write_text(label_content)

            from yolo.data.datamodule import YOLOFormatDataset

            dataset = YOLOFormatDataset(
                images_dir=str(images_dir),
                labels_dir=str(labels_dir),
            )

            image, target = dataset[0]
            box = target["boxes"][0]

            # Verify box is clipped to image bounds
            assert box[0] >= 0  # x1
            assert box[1] >= 0  # y1
            assert box[2] <= 640  # x2
            assert box[3] <= 480  # y2


class TestYOLOFormatDataModule:
    """Tests for YOLODataModule with format='yolo'."""

    def test_datamodule_initialization(self):
        """Test datamodule initializes with correct parameters."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(
            format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=4,
            num_workers=0,
        )
        dm._image_size = (320, 320)

        assert dm.hparams.batch_size == 4
        assert dm._image_size == (320, 320)
        assert dm.hparams.root == str(YOLO_DATASET_PATH)
        assert dm.hparams.format == "yolo"

    def test_datamodule_setup(self):
        """Test datamodule setup creates datasets."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(
            format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=4,
            num_workers=0,
        )
        dm._image_size = (320, 320)

        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.val_dataset) > 0

    def test_datamodule_train_dataloader(self):
        """Test train dataloader creation."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(
            format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=2,
            num_workers=0,
            mosaic_prob=0.0,  # Disable mosaic for simpler testing
        )
        dm._image_size = (320, 320)

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()

        assert train_loader is not None
        assert train_loader.batch_size == 2

    def test_datamodule_val_dataloader(self):
        """Test validation dataloader creation."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(
            format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=2,
            num_workers=0,
        )
        dm._image_size = (320, 320)

        dm.setup(stage="fit")
        val_loader = dm.val_dataloader()

        assert val_loader is not None
        assert val_loader.batch_size == 2

    def test_datamodule_batch_content(self):
        """Test batch has correct content and shapes."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(
            format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=4,
            num_workers=0,
            mosaic_prob=0.0,
        )
        dm._image_size = (320, 320)

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()
        images, targets = next(iter(train_loader))

        # Check images
        assert images.shape[0] == 4  # batch size
        assert images.shape[1] == 3  # RGB channels
        assert images.shape[2] == 320  # height
        assert images.shape[3] == 320  # width
        assert images.dtype == torch.float32

        # Check targets
        assert isinstance(targets, list)
        assert len(targets) == 4
        for target in targets:
            assert "boxes" in target
            assert "labels" in target


class TestYOLOFormatWithMosaic:
    """Tests for YOLO format dataset with mosaic augmentation."""

    def test_mosaic_integration(self):
        """Test YOLO format works with MosaicMixupDataset."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=2,
            num_workers=0,
            mosaic_prob=1.0,  # Always apply mosaic
            mixup_prob=0.0,
        )
        dm._image_size = (320, 320)

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()
        images, targets = next(iter(train_loader))

        # Should still get valid batches
        assert images.shape == (2, 3, 320, 320)
        assert len(targets) == 2

    def test_mosaic_with_mixup(self):
        """Test YOLO format with both mosaic and mixup."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=2,
            num_workers=0,
            mosaic_prob=1.0,
            mixup_prob=0.5,
            mixup_alpha=32.0,
        )
        dm._image_size = (320, 320)

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()

        # Get multiple batches to test augmentation
        for i, (images, targets) in enumerate(train_loader):
            if i >= 3:
                break
            assert images.shape[1:] == (3, 320, 320)

    def test_different_augmentation_configs(self):
        """Test various augmentation parameter combinations."""
        from yolo.data.datamodule import YOLODataModule

        configs = [
            {"mosaic_prob": 0.0, "mixup_prob": 0.0},
            {"mosaic_prob": 1.0, "mixup_prob": 0.0},
            {"mosaic_prob": 1.0, "mixup_prob": 0.5},
            {"mosaic_prob": 0.5, "mosaic_9_prob": 0.5},
        ]

        for config in configs:
            dm = YOLODataModule(format="yolo",
                root=str(YOLO_DATASET_PATH),
                train_images="train/images",
                train_labels="train/labels",
                val_images="valid/images",
                val_labels="valid/labels",
                batch_size=2,
                num_workers=0,
                **config,
            )
            dm._image_size = (320, 320)

            dm.setup(stage="fit")
            train_loader = dm.train_dataloader()
            images, targets = next(iter(train_loader))

            assert images.shape == (2, 3, 320, 320), f"Failed with config: {config}"


class TestYOLOFormatTransforms:
    """Tests for transforms applied to YOLO format data."""

    def test_validation_transforms(self):
        """Test that validation transforms resize correctly."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=2,
            num_workers=0,
        )
        dm._image_size = (416, 416)  # Different size

        dm.setup(stage="fit")
        val_loader = dm.val_dataloader()
        images, targets = next(iter(val_loader))

        assert images.shape[2:] == (416, 416)

    def test_hsv_augmentation_params(self):
        """Test HSV augmentation parameters are used."""
        from yolo.data.datamodule import YOLODataModule

        # Custom HSV params
        dm = YOLODataModule(format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=2,
            num_workers=0,
            hsv_h=0.05,
            hsv_s=0.9,
            hsv_v=0.6,
            mosaic_prob=0.0,
        )
        dm._image_size = (320, 320)

        dm.setup(stage="fit")

        # Verify params are stored
        assert dm.hparams.hsv_h == 0.05
        assert dm.hparams.hsv_s == 0.9
        assert dm.hparams.hsv_v == 0.6

    def test_geometric_augmentation_params(self):
        """Test geometric augmentation parameters."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=2,
            num_workers=0,
            degrees=15.0,
            translate=0.2,
            scale=0.5,
            shear=5.0,
            flip_lr=0.5,
            flip_ud=0.1,
            mosaic_prob=0.0,
        )
        dm._image_size = (320, 320)

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()
        images, targets = next(iter(train_loader))

        # Should produce valid output
        assert images.shape == (2, 3, 320, 320)


class TestYOLOFormatCollate:
    """Tests for collate function behavior."""

    def test_collate_variable_box_counts(self):
        """Test collate handles images with different numbers of boxes."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=8,  # Larger batch to get variety
            num_workers=0,
            mosaic_prob=0.0,
        )
        dm._image_size = (320, 320)

        dm.setup(stage="fit")
        val_loader = dm.val_dataloader()
        images, targets = next(iter(val_loader))

        # Each target can have different number of boxes
        box_counts = [len(t["boxes"]) for t in targets]
        print(f"\nBox counts in batch: {box_counts}")

        # All should be valid tensors
        for target in targets:
            assert target["boxes"].dtype == torch.float32
            assert target["labels"].dtype == torch.long

    def test_collate_preserves_target_structure(self):
        """Test collate preserves target dict structure."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=4,
            num_workers=0,
            mosaic_prob=0.0,
        )
        dm._image_size = (320, 320)

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()
        images, targets = next(iter(train_loader))

        for i, target in enumerate(targets):
            assert "boxes" in target, f"Target {i} missing 'boxes'"
            assert "labels" in target, f"Target {i} missing 'labels'"
            assert len(target["boxes"]) == len(target["labels"])


class TestYOLOFormatImageExtensions:
    """Tests for different image file extensions."""

    def test_supports_multiple_extensions(self):
        """Test that dataset finds images with various extensions."""
        from yolo.data.datamodule import YOLOFormatDataset

        dataset = YOLOFormatDataset(
            images_dir=str(YOLO_DATASET_PATH / "train" / "images"),
            labels_dir=str(YOLO_DATASET_PATH / "train" / "labels"),
        )

        # Get all extensions in dataset
        extensions = set(f.suffix.lower() for f in dataset.image_files)
        print(f"\nImage extensions found: {extensions}")

        # Should have found images
        assert len(dataset) > 0

    def test_case_insensitive_extensions(self):
        """Test that both .jpg and .JPG are found."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            images_dir = tmp_path / "images"
            labels_dir = tmp_path / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()

            # Create images with different case extensions
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(images_dir / "test1.jpg")
            img.save(images_dir / "test2.JPG")
            img.save(images_dir / "test3.png")
            img.save(images_dir / "test4.PNG")

            # Create label files
            for name in ["test1", "test2", "test3", "test4"]:
                (labels_dir / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            from yolo.data.datamodule import YOLOFormatDataset

            dataset = YOLOFormatDataset(
                images_dir=str(images_dir),
                labels_dir=str(labels_dir),
            )

            assert len(dataset) == 4


class TestYOLOFormatDatasetComparison:
    """Compare YOLO format loading with expected values."""

    def test_compare_with_raw_file_parsing(self):
        """Compare dataset output with manual file parsing."""
        from yolo.data.datamodule import YOLOFormatDataset

        dataset = YOLOFormatDataset(
            images_dir=str(YOLO_DATASET_PATH / "train" / "images"),
            labels_dir=str(YOLO_DATASET_PATH / "train" / "labels"),
        )

        # Get first sample
        image, target = dataset[0]
        image_path = dataset.image_files[0]
        label_path = dataset.labels_dir / (image_path.stem + ".txt")

        # Manually parse label file
        expected_boxes = []
        expected_labels = []

        if label_path.exists():
            img_w, img_h = image.size
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_c = float(parts[1])
                        y_c = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])

                        # Convert to xyxy
                        x1 = max(0, (x_c - w / 2) * img_w)
                        y1 = max(0, (y_c - h / 2) * img_h)
                        x2 = min(img_w, (x_c + w / 2) * img_w)
                        y2 = min(img_h, (y_c + h / 2) * img_h)

                        if x2 > x1 and y2 > y1:
                            expected_boxes.append([x1, y1, x2, y2])
                            expected_labels.append(class_id)

        # Compare
        assert len(target["boxes"]) == len(expected_boxes)
        assert len(target["labels"]) == len(expected_labels)

        if expected_boxes:
            for i in range(len(expected_boxes)):
                # Allow small floating point differences
                for j in range(4):
                    assert abs(target["boxes"][i][j].item() - expected_boxes[i][j]) < 1e-3


class TestYOLOFormatCustomLoader:
    """Tests for custom image loader support."""

    def test_default_loader_used(self):
        """Test that default loader is used when none specified."""
        from yolo.data.datamodule import YOLOFormatDataset
        from yolo.data.loaders import DefaultImageLoader

        dataset = YOLOFormatDataset(
            images_dir=str(YOLO_DATASET_PATH / "train" / "images"),
            labels_dir=str(YOLO_DATASET_PATH / "train" / "labels"),
        )

        assert isinstance(dataset._image_loader, DefaultImageLoader)

    def test_custom_loader_integration(self):
        """Test custom image loader can be used."""
        from yolo.data.datamodule import YOLOFormatDataset
        from yolo.data.loaders import ImageLoader
        from PIL import Image

        class TestLoader(ImageLoader):
            def __init__(self):
                self.call_count = 0

            def __call__(self, path: str) -> Image.Image:
                self.call_count += 1
                return Image.open(path).convert("RGB")

        loader = TestLoader()

        dataset = YOLOFormatDataset(
            images_dir=str(YOLO_DATASET_PATH / "train" / "images"),
            labels_dir=str(YOLO_DATASET_PATH / "train" / "labels"),
            image_loader=loader,
        )

        # Access some samples
        _ = dataset[0]
        _ = dataset[1]

        assert loader.call_count == 2


class TestYOLOFormatDataModuleHyperparams:
    """Tests for datamodule hyperparameter saving."""

    def test_hyperparams_saved(self):
        """Test that hyperparameters are saved correctly."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=16,
            num_workers=4,
            mosaic_prob=0.8,
            mixup_prob=0.2,
        )
        dm._image_size = (640, 640)

        assert dm.hparams.batch_size == 16
        assert dm.hparams.num_workers == 4
        assert dm._image_size == (640, 640)
        assert dm.hparams.mosaic_prob == 0.8
        assert dm.hparams.mixup_prob == 0.2

    def test_close_mosaic_epochs(self):
        """Test close_mosaic_epochs parameter."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=2,
            num_workers=0,
            close_mosaic_epochs=10,
        )
        dm._image_size = (320, 320)

        assert dm.hparams.close_mosaic_epochs == 10
        assert dm._mosaic_enabled is True


class TestYOLOFormatDatasetIntegration:
    """Integration tests with real training pipeline."""

    def test_full_epoch_iteration(self):
        """Test iterating through full dataset epoch."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=8,
            num_workers=0,
            mosaic_prob=0.0,
        )
        dm._image_size = (320, 320)

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()

        total_images = 0
        total_boxes = 0

        for images, targets in train_loader:
            total_images += images.shape[0]
            for target in targets:
                total_boxes += len(target["boxes"])

        print(f"\nTotal images in epoch: {total_images}")
        print(f"Total boxes in epoch: {total_boxes}")
        print(f"Average boxes per image: {total_boxes / max(1, total_images):.2f}")

        assert total_images > 0
        assert total_boxes > 0

    def test_multiple_epochs_iteration(self):
        """Test iterating multiple epochs (shuffle working)."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(format="yolo",
            root=str(YOLO_DATASET_PATH),
            train_images="train/images",
            train_labels="train/labels",
            val_images="valid/images",
            val_labels="valid/labels",
            batch_size=4,
            num_workers=0,
            mosaic_prob=0.0,
        )
        dm._image_size = (320, 320)

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()

        epoch1_first_batch = None
        epoch2_first_batch = None

        # Epoch 1
        for images, _ in train_loader:
            epoch1_first_batch = images.clone()
            break

        # Epoch 2
        for images, _ in train_loader:
            epoch2_first_batch = images.clone()
            break

        # Due to shuffling, batches might be different
        # (not guaranteed, but very likely with enough data)
        assert epoch1_first_batch is not None
        assert epoch2_first_batch is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
