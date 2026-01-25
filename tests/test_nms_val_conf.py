"""
Tests for nms_val_conf_threshold parameter in YOLOModule.

This parameter allows using a lower confidence threshold during validation
to capture all predictions for accurate mAP calculation, while keeping a
higher threshold for inference.
"""

import pytest
import torch


class TestNMSValConfThreshold:
    """Test nms_val_conf_threshold parameter."""

    @pytest.fixture
    def create_module(self):
        """Factory fixture to create YOLOModule with custom parameters."""
        from yolo.training.module import YOLOModule

        def _create(**kwargs):
            defaults = {
                "model_config": "v9-s",
                "num_classes": 13,
                "image_size": [320, 320],
            }
            defaults.update(kwargs)
            return YOLOModule(**defaults)

        return _create

    def test_default_value(self, create_module):
        """Test that nms_val_conf_threshold defaults to 0.001."""
        module = create_module()
        assert module.hparams.nms_val_conf_threshold == 0.001

    def test_custom_value_from_init(self, create_module):
        """Test that nms_val_conf_threshold can be set via init."""
        module = create_module(nms_val_conf_threshold=0.01)
        assert module.hparams.nms_val_conf_threshold == 0.01

    def test_nms_config_uses_val_threshold(self, create_module):
        """Test that _create_nms_config uses nms_val_conf_threshold."""
        module = create_module(nms_val_conf_threshold=0.005)
        nms_config = module._create_nms_config()
        assert nms_config.min_confidence == 0.005

    def test_val_threshold_independent_from_inference_threshold(self, create_module):
        """Test that val and inference thresholds are independent."""
        module = create_module(
            nms_conf_threshold=0.25,
            nms_val_conf_threshold=0.001
        )
        assert module.hparams.nms_conf_threshold == 0.25
        assert module.hparams.nms_val_conf_threshold == 0.001
        # NMS config for validation uses val threshold
        nms_config = module._create_nms_config()
        assert nms_config.min_confidence == 0.001

    def test_nms_config_includes_other_params(self, create_module):
        """Test that NMS config includes iou and max_det parameters."""
        module = create_module(
            nms_val_conf_threshold=0.002,
            nms_iou_threshold=0.5,
            nms_max_detections=100
        )
        nms_config = module._create_nms_config()
        assert nms_config.min_confidence == 0.002
        assert nms_config.min_iou == 0.5
        assert nms_config.max_bbox == 100

    def test_very_low_threshold_value(self, create_module):
        """Test that very low threshold values are accepted."""
        module = create_module(nms_val_conf_threshold=0.0001)
        assert module.hparams.nms_val_conf_threshold == 0.0001
        nms_config = module._create_nms_config()
        assert nms_config.min_confidence == 0.0001

    def test_threshold_zero(self, create_module):
        """Test that zero threshold is accepted (captures all predictions)."""
        module = create_module(nms_val_conf_threshold=0.0)
        assert module.hparams.nms_val_conf_threshold == 0.0
        nms_config = module._create_nms_config()
        assert nms_config.min_confidence == 0.0
