"""
Pytest configuration and fixtures for YOLO tests.
"""

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from yolo.config.config import AnchorConfig, ModelConfig
from yolo.model.yolo import YOLO, create_model
from yolo.utils.bounding_box_utils import Vec2Box, create_converter


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: mark test to run only if CUDA is available")


@pytest.fixture(scope="session")
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def model_cfg() -> ModelConfig:
    """Load v9-c model configuration."""
    model_yaml = project_root / "yolo" / "config" / "model" / "v9-c.yaml"
    cfg = OmegaConf.load(model_yaml)
    cfg = OmegaConf.merge(OmegaConf.structured(ModelConfig), cfg)
    return cfg


@pytest.fixture(scope="session")
def model(model_cfg: ModelConfig, device) -> YOLO:
    """Create YOLO model without pretrained weights."""
    model = create_model(model_cfg, weight_path=False, class_num=80)
    return model.to(device)


@pytest.fixture(scope="session")
def vec2box(model_cfg: ModelConfig, model: YOLO, device) -> Vec2Box:
    """Create Vec2Box converter."""
    return Vec2Box(model, model_cfg.anchor, [640, 640], device)


@pytest.fixture(scope="session")
def image_size():
    """Default image size."""
    return [640, 640]


@pytest.fixture
def sample_image(device):
    """Generate sample image tensor."""
    return torch.randn(1, 3, 640, 640, device=device)


@pytest.fixture
def sample_targets():
    """Generate sample targets in COCO format."""
    return [
        {
            "boxes": torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }
    ]
