"""
Integration module for swapping mock components into the training pipeline.

This module provides factory functions that return either the real or mock
implementation based on configuration.

Usage:
    # Set environment variable before importing
    import os
    os.environ["YOLO_MOCK_VERBOSE"] = "1"
    os.environ["YOLO_MOCK_MATCHER"] = "1"

    # Then import the factory functions
    from yolo.mock.integration import get_box_matcher, get_bce_loss

    # These will return mock versions if configured
    matcher = get_box_matcher(cfg, class_num, vec2box, reg_max)
    bce = get_bce_loss()
"""

import os
from typing import Optional

from .config import MockConfig


def get_box_matcher(cfg, class_num: int, vec2box, reg_max: int):
    """Get BoxMatcher - mock or real based on config."""
    config = MockConfig.get_instance()
    if config.mock_matcher or os.environ.get("YOLO_MOCK_MATCHER", "0") == "1":
        from .matcher import MockBoxMatcher
        return MockBoxMatcher(cfg, class_num, vec2box, reg_max)
    else:
        from yolo.utils.bounding_box_utils import BoxMatcher
        return BoxMatcher(cfg, class_num, vec2box, reg_max)


def get_bce_loss():
    """Get BCELoss - mock or real based on config."""
    config = MockConfig.get_instance()
    if config.mock_bce or os.environ.get("YOLO_MOCK_BCE", "0") == "1":
        from .losses import MockBCELoss
        return MockBCELoss()
    else:
        from yolo.training.loss import BCELoss
        return BCELoss()


def get_box_loss():
    """Get BoxLoss - mock or real based on config."""
    config = MockConfig.get_instance()
    if config.mock_box or os.environ.get("YOLO_MOCK_BOX", "0") == "1":
        from .losses import MockBoxLoss
        return MockBoxLoss()
    else:
        from yolo.training.loss import BoxLoss
        return BoxLoss()


def get_dfl_loss(vec2box, reg_max: int):
    """Get DFLoss - mock or real based on config."""
    config = MockConfig.get_instance()
    if config.mock_dfl or os.environ.get("YOLO_MOCK_DFL", "0") == "1":
        from .losses import MockDFLoss
        return MockDFLoss(vec2box, reg_max)
    else:
        from yolo.training.loss import DFLoss
        return DFLoss(vec2box, reg_max)


def get_vec2box(model, anchor_cfg, image_size, device):
    """Get Vec2Box - mock or real based on config."""
    config = MockConfig.get_instance()
    if config.mock_vec2box or os.environ.get("YOLO_MOCK_VEC2BOX", "0") == "1":
        from .vec2box import MockVec2Box
        return MockVec2Box(model, anchor_cfg, image_size, device)
    else:
        from yolo.utils.bounding_box_utils import Vec2Box
        return Vec2Box(model, anchor_cfg, image_size, device)


def get_ema_callback(decay: float = 0.9999, tau: float = 2000.0, enabled: bool = True):
    """Get EMACallback - mock or real based on config."""
    config = MockConfig.get_instance()
    if not enabled:
        from .ema import DisabledEMACallback
        return DisabledEMACallback()
    elif config.mock_ema or os.environ.get("YOLO_MOCK_EMA", "0") == "1":
        from .ema import MockEMACallback
        return MockEMACallback(decay=decay, tau=tau, enabled=enabled)
    else:
        from yolo.training.callbacks import EMACallback
        return EMACallback(decay=decay, tau=tau, enabled=enabled)


def get_yolo_loss(
    vec2box,
    class_num: int = 80,
    reg_max: int = 16,
    box_weight: float = 7.5,
    cls_weight: float = 0.5,
    dfl_weight: float = 1.5,
    matcher_topk: int = 10,
    matcher_iou_weight: float = 6.0,
    matcher_cls_weight: float = 0.5,
):
    """
    Get YOLOLoss - always use mock if any component is mocked.

    If YOLO_MOCK_ALL=1, use full mock loss.
    Otherwise, use real loss (which uses real sub-components).
    """
    if os.environ.get("YOLO_MOCK_ALL", "0") == "1":
        from .losses import MockYOLOLoss
        return MockYOLOLoss(
            vec2box=vec2box,
            class_num=class_num,
            reg_max=reg_max,
            box_weight=box_weight,
            cls_weight=cls_weight,
            dfl_weight=dfl_weight,
            matcher_topk=matcher_topk,
            matcher_iou_weight=matcher_iou_weight,
            matcher_cls_weight=matcher_cls_weight,
        )
    else:
        from yolo.training.loss import YOLOLoss
        return YOLOLoss(
            vec2box=vec2box,
            class_num=class_num,
            reg_max=reg_max,
            box_weight=box_weight,
            cls_weight=cls_weight,
            dfl_weight=dfl_weight,
            matcher_topk=matcher_topk,
            matcher_iou_weight=matcher_iou_weight,
            matcher_cls_weight=matcher_cls_weight,
        )


def print_mock_status():
    """Print which components are mocked."""
    config = MockConfig.get_instance()

    print("\n" + "=" * 60)
    print("YOLO Mock Components Status")
    print("=" * 60)
    print(f"  Verbose logging: {'ON' if config.verbose else 'OFF'}")
    print(f"  Log frequency: every {config.log_freq} calls")
    print()
    print("  Component Status:")
    print(f"    BoxMatcher:  {'MOCK' if config.mock_matcher or os.environ.get('YOLO_MOCK_MATCHER') == '1' else 'REAL'}")
    print(f"    BCELoss:     {'MOCK' if config.mock_bce or os.environ.get('YOLO_MOCK_BCE') == '1' else 'REAL'}")
    print(f"    BoxLoss:     {'MOCK' if config.mock_box or os.environ.get('YOLO_MOCK_BOX') == '1' else 'REAL'}")
    print(f"    DFLoss:      {'MOCK' if config.mock_dfl or os.environ.get('YOLO_MOCK_DFL') == '1' else 'REAL'}")
    print(f"    Vec2Box:     {'MOCK' if config.mock_vec2box or os.environ.get('YOLO_MOCK_VEC2BOX') == '1' else 'REAL'}")
    print(f"    EMA:         {'MOCK' if config.mock_ema or os.environ.get('YOLO_MOCK_EMA') == '1' else 'REAL'}")
    print()
    print("  To enable mock, set environment variable:")
    print("    export YOLO_MOCK_VERBOSE=1")
    print("    export YOLO_MOCK_MATCHER=1")
    print("    export YOLO_MOCK_ALL=1  # Enable all mocks")
    print("=" * 60 + "\n")
