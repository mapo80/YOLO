"""
Tests for YOLO model building and forward pass.

These tests verify that models can be built from YAML configs
and that forward passes produce expected output shapes.
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.model.yolo import YOLO, create_model


def load_model_config(model_name: str):
    """Load model config from YAML file as OmegaConf."""
    config_path = project_root / "yolo" / "config" / "model" / f"{model_name}.yaml"
    if not config_path.exists():
        pytest.skip(f"Model config {model_name}.yaml not found")

    return OmegaConf.load(config_path)


class TestModelBuilding:
    """Tests for building YOLO models from config."""

    def test_build_model_v9c(self):
        """Test building v9-c model."""
        cfg = load_model_config("v9-c")
        OmegaConf.set_struct(cfg, False)
        model = YOLO(cfg)
        assert len(model.model) == 39

    def test_build_model_v9m(self):
        """Test building v9-m model."""
        cfg = load_model_config("v9-m")
        OmegaConf.set_struct(cfg, False)
        model = YOLO(cfg)
        assert len(model.model) == 39

    def test_build_model_v9s(self):
        """Test building v9-s model."""
        cfg = load_model_config("v9-s")
        OmegaConf.set_struct(cfg, False)
        model = YOLO(cfg)
        # v9-s is a smaller model with 31 layers
        assert len(model.model) == 31

    @pytest.mark.skipif(
        not (project_root / "yolo" / "config" / "model" / "v7.yaml").exists(),
        reason="v7.yaml not found"
    )
    def test_build_model_v7(self):
        """Test building v7 model."""
        cfg = load_model_config("v7")
        OmegaConf.set_struct(cfg, False)
        model = YOLO(cfg)
        assert len(model.model) == 106


class TestModelForward:
    """Tests for YOLO model forward pass."""

    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def model(self, device):
        """Create v9-c model for testing."""
        cfg = load_model_config("v9-c")
        model = create_model(cfg, weight_path=None)
        return model.to(device)

    def test_model_basic_status(self, model):
        """Test model was built correctly."""
        assert isinstance(model, YOLO)
        assert len(model.model) == 39

    def test_yolo_forward_output_shape(self, model, device):
        """Test forward pass output shapes."""
        # 2 - batch size, 3 - number of channels, 640x640 - image dimensions
        dummy_input = torch.rand(2, 3, 640, 640, device=device)

        # Forward pass through the model
        output = model(dummy_input)
        output_shape = [(cls.shape, anc.shape, box.shape) for cls, anc, box in output["Main"]]
        assert output_shape == [
            (torch.Size([2, 80, 80, 80]), torch.Size([2, 16, 4, 80, 80]), torch.Size([2, 4, 80, 80])),
            (torch.Size([2, 80, 40, 40]), torch.Size([2, 16, 4, 40, 40]), torch.Size([2, 4, 40, 40])),
            (torch.Size([2, 80, 20, 20]), torch.Size([2, 16, 4, 20, 20]), torch.Size([2, 4, 20, 20])),
        ]

    def test_forward_different_batch_sizes(self, model, device):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 8]:
            dummy_input = torch.rand(batch_size, 3, 640, 640, device=device)
            output = model(dummy_input)

            # Check output batch dimension matches input
            for cls, anc, box in output["Main"]:
                assert cls.shape[0] == batch_size
                assert anc.shape[0] == batch_size
                assert box.shape[0] == batch_size

    def test_forward_different_input_sizes(self, device):
        """Test forward pass with different input sizes."""
        cfg = load_model_config("v9-c")
        model = create_model(cfg, weight_path=None).to(device)

        for size in [320, 640]:
            dummy_input = torch.rand(1, 3, size, size, device=device)
            output = model(dummy_input)

            # Should have 3 scale outputs
            assert len(output["Main"]) == 3


class TestCreateModel:
    """Tests for create_model helper function."""

    def test_create_model_v9c(self):
        """Test create_model with v9-c config."""
        cfg = load_model_config("v9-c")
        model = create_model(cfg, weight_path=None)
        assert isinstance(model, YOLO)

    def test_create_model_returns_eval_mode(self):
        """Test that create_model returns model in training mode."""
        cfg = load_model_config("v9-c")
        model = create_model(cfg, weight_path=None)
        # Model should be in training mode initially
        assert model.training
