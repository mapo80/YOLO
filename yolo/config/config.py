"""
YOLO Configuration Dataclasses.

These are used by the model builder (yolo.py) and utilities.
Training configuration is now handled by LightningCLI YAML files.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn


# =============================================================================
# Model Architecture Config (used by DSL YAML parser)
# =============================================================================


@dataclass
class AnchorConfig:
    """Anchor configuration for detection head."""

    strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    reg_max: Optional[int] = 16
    anchor_num: Optional[int] = None
    anchor: List[List[int]] = field(default_factory=list)


@dataclass
class LayerConfig:
    """Single layer configuration in model DSL."""

    args: Dict = field(default_factory=dict)
    source: Union[int, str, List[int]] = -1
    tags: Optional[str] = None


@dataclass
class BlockConfig:
    """Block of layers configuration."""

    block: List[Dict[str, LayerConfig]] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Complete model configuration from DSL YAML."""

    name: Optional[str] = None
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    model: Dict[str, BlockConfig] = field(default_factory=dict)


# =============================================================================
# Runtime Config (used by utilities)
# =============================================================================


@dataclass
class NMSConfig:
    """Non-Maximum Suppression configuration."""

    min_confidence: float = 0.25
    min_iou: float = 0.45  # yolov9-official uses 0.45 (was 0.65)
    max_bbox: int = 300


@dataclass
class MatcherConfig:
    """Box matcher configuration for loss computation."""

    iou: str = "ciou"
    topk: int = 10
    factor: Dict[str, float] = field(default_factory=lambda: {"iou": 6.0, "cls": 0.5})


# =============================================================================
# Module Base Class
# =============================================================================


@dataclass
class YOLOLayer(nn.Module):
    """Base class for YOLO layers with metadata."""

    source: Union[int, str, List[int]] = -1
    output: bool = False
    tags: Optional[str] = None
    layer_type: str = ""
    usable: bool = False
    external: Optional[dict] = None


# =============================================================================
# COCO Category Mapping
# =============================================================================

# Maps from 0-79 index to COCO category IDs (1-90 with gaps)
IDX_TO_ID = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

# Reverse mapping: COCO category ID to 0-79 index
ID_TO_IDX = {v: i for i, v in enumerate(IDX_TO_ID)}
