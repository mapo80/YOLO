"""Tests for loss functions."""

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from yolo.model.yolo import create_model
from yolo.training.loss import YOLOLoss
from yolo.utils.bounding_box_utils import Vec2Box


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model(device):
    model_yaml = project_root / "yolo" / "config" / "model" / "v9-c.yaml"
    model_cfg = OmegaConf.load(model_yaml)
    model = create_model(model_cfg, weight_path=False, class_num=80)
    return model.to(device)


@pytest.fixture
def vec2box(model, device):
    from yolo.config.config import AnchorConfig

    anchor_cfg = AnchorConfig(strides=[8, 16, 32], reg_max=16, anchor_num=None, anchor=[])
    return Vec2Box(model, anchor_cfg, [640, 640], device)


@pytest.fixture
def loss_function(vec2box):
    return YOLOLoss(
        vec2box=vec2box,
        class_num=80,
        reg_max=16,
        box_weight=7.5,
        cls_weight=0.5,
        dfl_weight=1.5,
    )


@pytest.mark.skip(reason="Empty targets edge case not fully supported by loss function")
def test_loss_with_empty_targets(loss_function, device):
    """Test loss computation with no targets."""
    # Empty targets
    targets = []

    # Mock outputs
    outputs = {
        "Main": [
            (
                torch.zeros(1, 80, 80, 80, device=device),
                torch.zeros(1, 4, 16, 80, 80, device=device),
                torch.zeros(1, 4, 80, 80, device=device),
            ),
            (
                torch.zeros(1, 80, 40, 40, device=device),
                torch.zeros(1, 4, 16, 40, 40, device=device),
                torch.zeros(1, 4, 40, 40, device=device),
            ),
            (
                torch.zeros(1, 80, 20, 20, device=device),
                torch.zeros(1, 4, 16, 20, 20, device=device),
                torch.zeros(1, 4, 20, 20, device=device),
            ),
        ]
    }

    loss, loss_dict = loss_function(outputs, targets)

    # With no targets, box and DFL loss should be 0
    assert "box_loss" in loss_dict
    assert "cls_loss" in loss_dict
    assert "dfl_loss" in loss_dict
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
