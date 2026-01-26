"""
Mock implementations for YOLO training components.

These mock implementations can be swapped in to isolate and debug specific parts
of the training pipeline. Each mock logs its inputs/outputs for debugging.

Usage:
    # In loss.py, replace:
    from yolo.utils.bounding_box_utils import BoxMatcher
    # With:
    from yolo.mock import MockBoxMatcher as BoxMatcher

Configuration via environment variables:
    YOLO_MOCK_VERBOSE=1  - Enable verbose logging
    YOLO_MOCK_LOG_FREQ=10 - Log every N calls (default: 10)
"""

from .losses import MockBCELoss, MockBoxLoss, MockDFLoss, MockYOLOLoss
from .matcher import MockBoxMatcher
from .vec2box import MockVec2Box
from .ema import MockModelEMA, MockEMACallback
from .config import MockConfig

__all__ = [
    "MockBCELoss",
    "MockBoxLoss",
    "MockDFLoss",
    "MockYOLOLoss",
    "MockBoxMatcher",
    "MockVec2Box",
    "MockModelEMA",
    "MockEMACallback",
    "MockConfig",
]
