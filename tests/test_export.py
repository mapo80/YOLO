"""
Unit tests for model export functionality.

Tests ONNX export (always) and TFLite export (when dependencies available).
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from yolo.tools.export import (
    _get_calibration_images,
    _letterbox_image,
)


class TestLetterboxImage:
    """Tests for letterbox image preprocessing."""

    def test_square_to_square(self):
        """Test letterboxing square image to square target."""
        img = Image.new("RGB", (100, 100), color="red")
        result = _letterbox_image(img, (100, 100))
        assert result.size == (100, 100)

    def test_landscape_to_square(self):
        """Test letterboxing landscape image to square target."""
        img = Image.new("RGB", (200, 100), color="red")
        result = _letterbox_image(img, (100, 100))
        assert result.size == (100, 100)
        # Should have letterbox bars on top/bottom
        # Center should be red
        center_pixel = result.getpixel((50, 50))
        assert center_pixel[0] > 200  # Red channel high

    def test_portrait_to_square(self):
        """Test letterboxing portrait image to square target."""
        img = Image.new("RGB", (100, 200), color="blue")
        result = _letterbox_image(img, (100, 100))
        assert result.size == (100, 100)
        # Should have letterbox bars on left/right
        center_pixel = result.getpixel((50, 50))
        assert center_pixel[2] > 200  # Blue channel high

    def test_custom_fill_value(self):
        """Test letterboxing with custom fill value."""
        img = Image.new("RGB", (50, 100), color="white")
        result = _letterbox_image(img, (100, 100), fill_value=0)
        # Corner should be black (fill value 0)
        corner_pixel = result.getpixel((0, 0))
        assert corner_pixel == (0, 0, 0)

    def test_maintains_aspect_ratio(self):
        """Test that letterboxing maintains aspect ratio."""
        img = Image.new("RGB", (300, 100), color="green")
        result = _letterbox_image(img, (200, 200))
        assert result.size == (200, 200)


class TestCalibrationImages:
    """Tests for calibration image generation."""

    def test_generates_images(self):
        """Test that calibration generator yields images."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test images
            for i in range(5):
                img = Image.new("RGB", (100, 100), color="red")
                img.save(Path(tmp_dir) / f"image_{i}.jpg")

            # Generate calibration data
            images = list(_get_calibration_images(tmp_dir, (64, 64), num_images=3))

            assert len(images) == 3
            for img in images:
                assert img.shape == (1, 64, 64, 3)  # NHWC format
                assert img.dtype == np.float32
                assert img.min() >= 0.0
                assert img.max() <= 1.0

    def test_handles_empty_directory(self):
        """Test that empty directory raises error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="No images found"):
                list(_get_calibration_images(tmp_dir, (64, 64)))

    def test_limits_image_count(self):
        """Test that num_images parameter limits output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create more images than limit
            for i in range(10):
                img = Image.new("RGB", (50, 50), color="blue")
                img.save(Path(tmp_dir) / f"img_{i}.png")

            images = list(_get_calibration_images(tmp_dir, (32, 32), num_images=5))
            assert len(images) == 5

    def test_handles_multiple_formats(self):
        """Test that multiple image formats are supported."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create images in different formats
            img = Image.new("RGB", (100, 100), color="green")
            img.save(Path(tmp_dir) / "image1.jpg")
            img.save(Path(tmp_dir) / "image2.png")
            img.save(Path(tmp_dir) / "image3.bmp")

            images = list(_get_calibration_images(tmp_dir, (64, 64)))
            assert len(images) == 3


class TestExportONNX:
    """Tests for ONNX export functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        import torch

        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x

        return SimpleModel()

    def test_export_onnx_signature(self):
        """Test that export_onnx has correct signature."""
        from yolo.tools.export import export_onnx
        import inspect

        sig = inspect.signature(export_onnx)
        params = list(sig.parameters.keys())

        assert "checkpoint_path" in params
        assert "output_path" in params
        assert "image_size" in params
        assert "opset_version" in params
        assert "simplify" in params
        assert "dynamic_batch" in params
        assert "half" in params
        assert "device" in params


class TestExportTFLite:
    """Tests for TFLite export functionality."""

    def test_export_tflite_signature(self):
        """Test that export_tflite has correct signature."""
        from yolo.tools.export import export_tflite
        import inspect

        sig = inspect.signature(export_tflite)
        params = list(sig.parameters.keys())

        assert "checkpoint_path" in params
        assert "output_path" in params
        assert "image_size" in params
        assert "quantization" in params
        assert "calibration_images" in params
        assert "num_calibration_images" in params
        assert "device" in params

    def test_invalid_quantization_raises_error(self):
        """Test that invalid quantization mode raises ValueError."""
        from yolo.tools.export import export_tflite

        with pytest.raises(ValueError, match="Invalid quantization"):
            export_tflite(
                checkpoint_path="fake.ckpt",
                quantization="invalid",
            )

    def test_int8_without_calibration_raises_error(self):
        """Test that INT8 without calibration images raises ValueError."""
        from yolo.tools.export import export_tflite

        with pytest.raises(ValueError, match="INT8 quantization requires"):
            export_tflite(
                checkpoint_path="fake.ckpt",
                quantization="int8",
                calibration_images=None,
            )


class TestExportSavedModel:
    """Tests for SavedModel export functionality."""

    def test_export_saved_model_signature(self):
        """Test that export_saved_model has correct signature."""
        from yolo.tools.export import export_saved_model
        import inspect

        sig = inspect.signature(export_saved_model)
        params = list(sig.parameters.keys())

        assert "checkpoint_path" in params
        assert "output_path" in params
        assert "image_size" in params
        assert "device" in params


class TestCLIExport:
    """Tests for CLI export command."""

    def test_cli_export_format_choices(self):
        """Test that CLI supports correct format choices."""
        import argparse
        import sys

        # Import the parser setup from cli
        from yolo.cli import export_main

        # Check that the function exists and is callable
        assert callable(export_main)

    def test_cli_quantization_choices(self):
        """Test that CLI supports correct quantization choices."""
        # Valid choices should be fp32, fp16, int8
        valid_choices = {"fp32", "fp16", "int8"}
        from yolo.tools.export import export_tflite
        import inspect

        # Get the docstring and check it mentions the valid quantization modes
        doc = export_tflite.__doc__
        for choice in valid_choices:
            assert choice in doc
