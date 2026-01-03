"""
YOLO - Official Implementation with PyTorch Lightning.

Usage:
    # Training
    python -m yolo.cli fit --config yolo/config/experiment/default.yaml

    # Validation
    python -m yolo.cli validate --ckpt_path=best.ckpt

    # Or use the installed command
    yolo fit --config yolo/config/experiment/default.yaml
"""

# Suppress noisy warnings from dependencies
# Must be at the very top before any imports that might trigger them
import warnings

# pkg_resources deprecation warnings from torchmetrics
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*Deprecated call to.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*pkg_resources.*")

from yolo.config.config import ModelConfig, NMSConfig
from yolo.model.yolo import YOLO, create_model
from yolo.utils.bounding_box_utils import Anc2Box, Vec2Box, bbox_nms, create_converter

__all__ = [
    # Model
    "YOLO",
    "create_model",
    # Config
    "ModelConfig",
    "NMSConfig",
    # Utilities
    "Vec2Box",
    "Anc2Box",
    "bbox_nms",
    "create_converter",
]
