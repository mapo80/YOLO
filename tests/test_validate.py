"""
Tests for validate module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from yolo.tools.validate import (
    get_device,
    BenchmarkResult,
    run_benchmark,
    get_gpu_memory,
)


class TestGetDevice:
    """Tests for get_device function."""

    def test_explicit_cpu(self):
        """Test explicit CPU device."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_explicit_cuda(self):
        """Test explicit CUDA device."""
        if torch.cuda.is_available():
            device = get_device("cuda")
            assert device.type == "cuda"
        else:
            pytest.skip("CUDA not available")

    def test_auto_detect(self):
        """Test auto-detection of device."""
        device = get_device(None)
        assert device.type in ["cuda", "mps", "cpu"]


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_initialization(self):
        """Test benchmark result initialization."""
        result = BenchmarkResult(
            latency_mean_ms=10.5,
            latency_std_ms=1.2,
            fps=95.2,
            memory_mb=256.0,
            model_size_mb=15.5,
        )
        assert result.latency_mean_ms == 10.5
        assert result.latency_std_ms == 1.2
        assert result.fps == 95.2
        assert result.memory_mb == 256.0
        assert result.model_size_mb == 15.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BenchmarkResult(
            latency_mean_ms=10.5,
            latency_std_ms=1.2,
            fps=95.2,
            memory_mb=256.0,
            model_size_mb=15.5,
        )
        d = result.to_dict()
        assert d["latency_mean_ms"] == 10.5
        assert d["latency_std_ms"] == 1.2
        assert d["fps"] == 95.2
        assert d["memory_mb"] == 256.0
        assert d["model_size_mb"] == 15.5

    def test_optional_fields(self):
        """Test optional fields default to None."""
        result = BenchmarkResult(
            latency_mean_ms=10.5,
            latency_std_ms=1.2,
            fps=95.2,
        )
        assert result.memory_mb is None
        assert result.model_size_mb is None


class TestGetGpuMemory:
    """Tests for get_gpu_memory function."""

    def test_returns_float_or_none(self):
        """Test that function returns float or None."""
        result = get_gpu_memory()
        assert result is None or isinstance(result, float)


class TestRunBenchmark:
    """Tests for run_benchmark function."""

    @pytest.fixture
    def mock_model(self):
        """Create a simple mock model."""
        model = Mock()
        model.eval = Mock(return_value=model)
        # Return some dummy tensor when called
        model.return_value = torch.randn(1, 10)
        return model

    def test_benchmark_basic(self, mock_model):
        """Test basic benchmark functionality."""
        device = torch.device("cpu")
        result = run_benchmark(
            model=mock_model,
            device=device,
            image_size=(640, 640),
            batch_size=1,
            warmup=2,
            runs=5,
        )
        assert isinstance(result, BenchmarkResult)
        assert result.latency_mean_ms > 0
        assert result.latency_std_ms >= 0
        assert result.fps > 0

    def test_benchmark_with_checkpoint(self, mock_model, tmp_path):
        """Test benchmark with checkpoint path for model size."""
        # Create a dummy checkpoint file
        checkpoint = tmp_path / "model.ckpt"
        checkpoint.write_bytes(b"x" * 1024 * 1024)  # 1 MB

        device = torch.device("cpu")
        result = run_benchmark(
            model=mock_model,
            device=device,
            image_size=(640, 640),
            batch_size=1,
            warmup=1,
            runs=2,
            checkpoint_path=str(checkpoint),
        )
        assert result.model_size_mb is not None
        assert result.model_size_mb == pytest.approx(1.0, rel=0.1)

    def test_benchmark_different_batch_sizes(self, mock_model):
        """Test benchmark with different batch sizes."""
        device = torch.device("cpu")

        for batch_size in [1, 4]:
            result = run_benchmark(
                model=mock_model,
                device=device,
                image_size=(640, 640),
                batch_size=batch_size,
                warmup=1,
                runs=2,
            )
            assert result.fps > 0

    def test_benchmark_different_image_sizes(self, mock_model):
        """Test benchmark with different image sizes."""
        device = torch.device("cpu")

        for size in [(320, 320), (640, 640)]:
            result = run_benchmark(
                model=mock_model,
                device=device,
                image_size=size,
                batch_size=1,
                warmup=1,
                runs=2,
            )
            assert result.latency_mean_ms > 0


class TestValidateIntegration:
    """Integration tests for validate function.

    These tests require actual model files and are marked as integration tests.
    Run with: pytest --run-integration
    """

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        return [
            {
                "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 1]),
            }
        ]

    @pytest.fixture
    def sample_targets(self):
        """Create sample targets."""
        return [
            {
                "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
                "labels": torch.tensor([0, 1]),
            }
        ]

    def test_format_targets_dict(self):
        """Test _format_targets with dict input."""
        from yolo.tools.validate import _format_targets

        device = torch.device("cpu")
        targets = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]]),
                "labels": torch.tensor([0]),
            }
        ]
        formatted = _format_targets(targets, device)
        assert len(formatted) == 1
        assert "boxes" in formatted[0]
        assert "labels" in formatted[0]

    def test_format_targets_tensor(self):
        """Test _format_targets with tensor input."""
        from yolo.tools.validate import _format_targets

        device = torch.device("cpu")
        # Tensor format: [class, x1, y1, x2, y2]
        targets = [
            torch.tensor([[0, 10, 10, 50, 50], [1, 60, 60, 100, 100]])
        ]
        formatted = _format_targets(targets, device)
        assert len(formatted) == 1
        assert "boxes" in formatted[0]
        assert "labels" in formatted[0]
        assert formatted[0]["boxes"].shape == (2, 4)
        assert formatted[0]["labels"].shape == (2,)

    def test_format_targets_empty(self):
        """Test _format_targets with empty input."""
        from yolo.tools.validate import _format_targets

        device = torch.device("cpu")
        targets = [torch.zeros((0, 5))]
        formatted = _format_targets(targets, device)
        assert len(formatted) == 1
        assert formatted[0]["boxes"].shape == (0, 4)
        assert formatted[0]["labels"].shape == (0,)


class TestDecodePredicitions:
    """Tests for decode_predictions function."""

    def test_decode_empty_batch(self):
        """Test decoding empty predictions."""
        from yolo.tools.validate import decode_predictions

        outputs = torch.zeros((2, 0, 6))
        predictions = decode_predictions(outputs, None)
        assert len(predictions) == 2
        assert all(p.shape[0] == 0 for p in predictions)

    def test_decode_with_confidence_filter(self):
        """Test that confidence threshold filters predictions."""
        from yolo.tools.validate import decode_predictions

        # Create predictions with varying confidence
        outputs = torch.tensor([
            [[10, 10, 50, 50, 0.9, 0]],  # High conf
            [[60, 60, 100, 100, 0.1, 1]],  # Low conf
        ])
        predictions = decode_predictions(
            outputs, None, conf_threshold=0.5
        )
        assert len(predictions) == 2
        assert predictions[0].shape[0] == 1  # High conf kept
        assert predictions[1].shape[0] == 0  # Low conf filtered
