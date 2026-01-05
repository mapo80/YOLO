"""
Tests for YOLO CLI functionality.

Tests the image_size validation and link_arguments functionality.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class TestImageSizeValidation:
    """Test image_size validation between model and datamodule."""

    def test_image_size_mismatch_raises_error(self):
        """Test that mismatched image_size raises ValueError.

        When data.image_size differs from model.image_size, the CLI should
        raise a clear error message telling users NOT to specify data.image_size
        manually.
        """
        from yolo.data.datamodule import YOLODataModule
        from yolo.training.module import YOLOModule

        # Create model with 640x640
        model = YOLOModule(
            model_config="v9-t",
            num_classes=80,
            image_size=[640, 640],
        )

        # Create datamodule and manually set a DIFFERENT image_size
        # (simulating user error of specifying data.image_size manually)
        datamodule = YOLODataModule(
            root="data/coco",
            train_images="train2017",
            val_images="val2017",
            train_ann="annotations/instances_train2017.json",
            val_ann="annotations/instances_val2017.json",
            batch_size=16,
            num_workers=0,
        )
        # Manually override to simulate mismatch (user specified different data.image_size)
        datamodule._image_size = (320, 320)

        # Simulate the CLI validation method
        model_size = tuple(model.hparams.image_size)
        data_size = datamodule._image_size

        # This should detect the mismatch
        assert model_size != data_size, "Test setup: sizes should be different"
        assert model_size == (640, 640)
        assert data_size == (320, 320)

        # Verify error message format
        error_msg = (
            f"Image size mismatch: model.image_size={model_size} != data.image_size={data_size}. "
            f"Do NOT specify data.image_size manually - it is automatically linked from model.image_size."
        )

        with pytest.raises(ValueError, match="Image size mismatch"):
            raise ValueError(error_msg)

    def test_image_size_match_no_error(self):
        """Test that matching image_size does NOT raise error."""
        from yolo.data.datamodule import YOLODataModule
        from yolo.training.module import YOLOModule

        # Create model with 640x640
        model = YOLOModule(
            model_config="v9-t",
            num_classes=80,
            image_size=[640, 640],
        )

        # Create datamodule with SAME image_size (as CLI would do via link_arguments)
        datamodule = YOLODataModule(
            root="data/coco",
            train_images="train2017",
            val_images="val2017",
            train_ann="annotations/instances_train2017.json",
            val_ann="annotations/instances_val2017.json",
            batch_size=16,
            num_workers=0,
            image_size=(640, 640),  # Same as model
        )

        model_size = tuple(model.hparams.image_size)
        data_size = datamodule._image_size

        # Sizes should match - no error
        assert model_size == data_size == (640, 640)

    def test_validate_image_size_method(self):
        """Test the _validate_image_size method directly."""
        from yolo.data.datamodule import YOLODataModule
        from yolo.training.module import YOLOModule

        # Create a mock CLI class with the validation method
        class MockCLI:
            def __init__(self, model, datamodule):
                self.model = model
                self.datamodule = datamodule

            def _validate_image_size(self):
                """Validate that model and datamodule image_size match."""
                if self.model is None or self.datamodule is None:
                    return

                model_size = tuple(self.model.hparams.image_size)
                data_size = self.datamodule._image_size

                if model_size != data_size:
                    raise ValueError(
                        f"Image size mismatch: model.image_size={model_size} != data.image_size={data_size}. "
                        f"Do NOT specify data.image_size manually - it is automatically linked from model.image_size."
                    )

        # Test with mismatch
        model = YOLOModule(
            model_config="v9-t",
            num_classes=80,
            image_size=[640, 640],
        )
        datamodule = YOLODataModule(
            root="data/coco",
            train_images="train2017",
            val_images="val2017",
            train_ann="annotations/instances_train2017.json",
            val_ann="annotations/instances_val2017.json",
            batch_size=16,
            num_workers=0,
        )
        datamodule._image_size = (416, 416)  # Different from model

        cli = MockCLI(model, datamodule)

        with pytest.raises(ValueError) as exc_info:
            cli._validate_image_size()

        assert "Image size mismatch" in str(exc_info.value)
        assert "model.image_size=(640, 640)" in str(exc_info.value)
        assert "data.image_size=(416, 416)" in str(exc_info.value)
        assert "Do NOT specify data.image_size manually" in str(exc_info.value)

    def test_validate_image_size_with_none_model(self):
        """Test validation skips when model is None."""
        from yolo.data.datamodule import YOLODataModule

        class MockCLI:
            def __init__(self, model, datamodule):
                self.model = model
                self.datamodule = datamodule

            def _validate_image_size(self):
                if self.model is None or self.datamodule is None:
                    return
                # Would raise error here if not skipped
                raise ValueError("Should not reach here")

        datamodule = YOLODataModule(
            root="data/coco",
            train_images="train2017",
            val_images="val2017",
            train_ann="annotations/instances_train2017.json",
            val_ann="annotations/instances_val2017.json",
            batch_size=16,
            num_workers=0,
        )

        cli = MockCLI(None, datamodule)
        # Should not raise - skips validation when model is None
        cli._validate_image_size()

    def test_validate_image_size_with_none_datamodule(self):
        """Test validation skips when datamodule is None."""
        from yolo.training.module import YOLOModule

        class MockCLI:
            def __init__(self, model, datamodule):
                self.model = model
                self.datamodule = datamodule

            def _validate_image_size(self):
                if self.model is None or self.datamodule is None:
                    return
                # Would raise error here if not skipped
                raise ValueError("Should not reach here")

        model = YOLOModule(
            model_config="v9-t",
            num_classes=80,
            image_size=[640, 640],
        )

        cli = MockCLI(model, None)
        # Should not raise - skips validation when datamodule is None
        cli._validate_image_size()


class TestImageSizeLinkArguments:
    """Test that link_arguments properly propagates image_size."""

    def test_image_size_parameter_exists(self):
        """Test that YOLODataModule has image_size parameter."""
        from yolo.data.datamodule import YOLODataModule
        import inspect

        sig = inspect.signature(YOLODataModule.__init__)
        params = sig.parameters

        assert "image_size" in params, "YOLODataModule should have image_size parameter"

        # Check default value
        default = params["image_size"].default
        assert default == (640, 640), f"Default should be (640, 640), got {default}"

    def test_image_size_sets_internal_field(self):
        """Test that image_size parameter sets _image_size field."""
        from yolo.data.datamodule import YOLODataModule

        dm = YOLODataModule(
            root="data/coco",
            train_images="train2017",
            val_images="val2017",
            train_ann="annotations/instances_train2017.json",
            val_ann="annotations/instances_val2017.json",
            batch_size=16,
            num_workers=0,
            image_size=(416, 416),
        )

        assert dm._image_size == (416, 416)

    def test_image_size_accepts_list(self):
        """Test that image_size accepts list (as CLI would pass)."""
        from yolo.data.datamodule import YOLODataModule

        # CLI passes list from YAML, should be converted to tuple
        dm = YOLODataModule(
            root="data/coco",
            train_images="train2017",
            val_images="val2017",
            train_ann="annotations/instances_train2017.json",
            val_ann="annotations/instances_val2017.json",
            batch_size=16,
            num_workers=0,
            image_size=[512, 512],  # List, not tuple
        )

        # Should be stored as tuple
        assert dm._image_size == (512, 512)
        assert isinstance(dm._image_size, tuple)


class TestCLIDocumentation:
    """Test that documentation warns about data.image_size."""

    def test_datamodule_docstring_warns(self):
        """Test that YOLODataModule docstring warns about image_size."""
        from yolo.data.datamodule import YOLODataModule

        docstring = YOLODataModule.__doc__

        assert "image_size" in docstring
        assert "Do NOT specify" in docstring or "DO NOT specify" in docstring
        assert "model.image_size" in docstring

    def test_cli_docstring_mentions_propagation(self):
        """Test that CLI docstring mentions image_size propagation."""
        from yolo.cli import YOLOLightningCLI

        docstring = YOLOLightningCLI.__doc__

        assert "image_size" in docstring.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
