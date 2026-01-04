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


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (skipped by default)",
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "requires_cuda: mark test to run only if CUDA is available")
    config.addinivalue_line("markers", "integration: mark test as integration test (skipped by default, use --run-integration to run)")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        # --run-integration given: do not skip integration tests
        return

    skip_integration = pytest.mark.skip(reason="Integration tests skipped by default. Use --run-integration to run.")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


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


# =============================================================================
# Metrics Testing Fixtures
# =============================================================================


@pytest.fixture
def class_names():
    """Class names for testing (5 classes)."""
    return {0: "cat", 1: "dog", 2: "bird", 3: "fish", 4: "horse"}


@pytest.fixture
def coco_class_names():
    """COCO 80 class names subset."""
    return {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    }


@pytest.fixture
def sample_predictions():
    """Generate sample predictions for metrics testing."""
    return [
        {
            "boxes": torch.tensor([
                [100, 100, 200, 200],  # TP for class 0
                [300, 300, 400, 400],  # TP for class 1
                [500, 500, 600, 600],  # FP for class 2
            ], dtype=torch.float32),
            "scores": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
            "labels": torch.tensor([0, 1, 2], dtype=torch.long),
        }
    ]


@pytest.fixture
def sample_ground_truth():
    """Generate sample ground truth for metrics testing."""
    return [
        {
            "boxes": torch.tensor([
                [100, 100, 200, 200],  # GT class 0
                [300, 300, 400, 400],  # GT class 1
                [700, 700, 800, 800],  # FN class 3
            ], dtype=torch.float32),
            "labels": torch.tensor([0, 1, 3], dtype=torch.long),
        }
    ]


@pytest.fixture
def perfect_predictions():
    """Predictions that perfectly match ground truth."""
    return [
        {
            "boxes": torch.tensor([
                [100, 100, 200, 200],
                [300, 300, 400, 400],
            ], dtype=torch.float32),
            "scores": torch.tensor([0.95, 0.90], dtype=torch.float32),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }
    ]


@pytest.fixture
def perfect_ground_truth():
    """Ground truth that matches perfect predictions."""
    return [
        {
            "boxes": torch.tensor([
                [100, 100, 200, 200],
                [300, 300, 400, 400],
            ], dtype=torch.float32),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }
    ]


@pytest.fixture
def multi_batch_predictions():
    """Multiple batches of predictions for stress testing."""
    batches = []
    for batch_idx in range(5):
        preds = {
            "boxes": torch.tensor([
                [100 + batch_idx * 10, 100, 200, 200],
                [300 + batch_idx * 10, 300, 400, 400],
            ], dtype=torch.float32),
            "scores": torch.tensor([0.9 - batch_idx * 0.1, 0.8 - batch_idx * 0.05], dtype=torch.float32),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }
        batches.append(preds)
    return batches


@pytest.fixture
def multi_batch_ground_truth():
    """Multiple batches of ground truth for stress testing."""
    batches = []
    for batch_idx in range(5):
        gt = {
            "boxes": torch.tensor([
                [100 + batch_idx * 10, 100, 200, 200],
                [300 + batch_idx * 10, 300, 400, 400],
            ], dtype=torch.float32),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }
        batches.append(gt)
    return batches


@pytest.fixture
def empty_predictions():
    """Empty predictions (no detections)."""
    return [
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros(0, dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.long),
        }
    ]


@pytest.fixture
def empty_ground_truth():
    """Empty ground truth (no objects)."""
    return [
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.long),
        }
    ]


@pytest.fixture
def overlapping_boxes_predictions():
    """Predictions with overlapping boxes (same class)."""
    return [
        {
            "boxes": torch.tensor([
                [100, 100, 200, 200],
                [110, 110, 210, 210],  # Overlaps with first
                [120, 120, 220, 220],  # Overlaps with both
            ], dtype=torch.float32),
            "scores": torch.tensor([0.95, 0.90, 0.85], dtype=torch.float32),
            "labels": torch.tensor([0, 0, 0], dtype=torch.long),  # Same class
        }
    ]


@pytest.fixture
def overlapping_boxes_ground_truth():
    """Ground truth for overlapping predictions."""
    return [
        {
            "boxes": torch.tensor([
                [100, 100, 200, 200],
            ], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }
    ]


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for saving plots and files."""
    return tmp_path


# =============================================================================
# Training Experiment Dataset Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def training_experiment_path():
    """Path to training-experiment dataset."""
    return project_root / "data" / "training-experiment"


@pytest.fixture(scope="session")
def training_experiment_exists(training_experiment_path):
    """Check if training-experiment dataset exists."""
    return training_experiment_path.exists()


@pytest.fixture(scope="session")
def training_experiment_config(training_experiment_path):
    """Load training-experiment dataset configuration if exists."""
    if not training_experiment_path.exists():
        return None

    config_path = training_experiment_path / "dataset.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)
    return None
