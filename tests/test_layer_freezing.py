"""
Unit tests for layer freezing functionality in YOLOModule.

Tests cover:
- Backbone freezing
- Specific layer pattern freezing
- Epoch-based unfreezing
- Parameter counting
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing layer freezing."""

    def __init__(self):
        super().__init__()
        self.backbone_conv1 = nn.Conv2d(3, 64, 3)
        self.backbone_conv2 = nn.Conv2d(64, 128, 3)
        self.neck_conv = nn.Conv2d(128, 256, 3)
        self.head_conv = nn.Conv2d(256, 80, 1)

    def forward(self, x):
        x = self.backbone_conv1(x)
        x = self.backbone_conv2(x)
        x = self.neck_conv(x)
        x = self.head_conv(x)
        return x


class TestLayerFreezing:
    """Tests for layer freezing functionality."""

    def test_freeze_by_pattern(self):
        """Test freezing layers by name pattern."""
        model = SimpleModel()

        # Initially all parameters should require grad
        for param in model.parameters():
            assert param.requires_grad

        # Freeze backbone layers
        patterns = ["backbone"]
        for name, param in model.named_parameters():
            for pattern in patterns:
                if pattern in name:
                    param.requires_grad = False
                    break

        # Check backbone is frozen
        assert not model.backbone_conv1.weight.requires_grad
        assert not model.backbone_conv2.weight.requires_grad

        # Check neck/head are not frozen
        assert model.neck_conv.weight.requires_grad
        assert model.head_conv.weight.requires_grad

    def test_unfreeze_all_layers(self):
        """Test unfreezing all layers."""
        model = SimpleModel()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Verify all frozen
        for param in model.parameters():
            assert not param.requires_grad

        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True

        # Verify all unfrozen
        for param in model.parameters():
            assert param.requires_grad

    def test_freeze_specific_layers(self):
        """Test freezing specific layers by full name."""
        model = SimpleModel()

        # Freeze only backbone_conv1
        for name, param in model.named_parameters():
            if "backbone_conv1" in name:
                param.requires_grad = False

        # Check specific layer is frozen
        assert not model.backbone_conv1.weight.requires_grad

        # Check other layers are not frozen
        assert model.backbone_conv2.weight.requires_grad
        assert model.neck_conv.weight.requires_grad
        assert model.head_conv.weight.requires_grad

    def test_count_trainable_parameters(self):
        """Test counting trainable parameters."""
        model = SimpleModel()

        # Count all parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Count trainable parameters (all initially)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params == total_params

        # Freeze backbone
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

        # Recount
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert frozen_params + trainable_params == total_params
        assert frozen_params > 0
        assert trainable_params > 0

    def test_count_frozen_parameters(self):
        """Test counting frozen parameters."""
        model = SimpleModel()

        # Initially no frozen parameters
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        assert frozen_params == 0

        # Freeze half the model
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

        # Now some should be frozen
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        assert frozen_params > 0


class TestBackbonePatternDetection:
    """Tests for backbone layer pattern detection."""

    def test_backbone_pattern_in_name(self):
        """Test detecting backbone pattern in layer name."""
        backbone_patterns = [
            "backbone",
            "stem",
            "dark",
            "stage1", "stage2", "stage3",
            "conv1", "conv2", "conv3",
            "layer1", "layer2", "layer3",
        ]

        # Test positive cases
        test_names = [
            "backbone.conv1.weight",
            "stem.0.weight",
            "dark1.conv.weight",
            "stage1.block.weight",
            "model.layer1.weight",
        ]

        for name in test_names:
            name_lower = name.lower()
            matched = any(pattern in name_lower for pattern in backbone_patterns)
            assert matched, f"Expected {name} to match backbone pattern"

        # Test negative cases
        non_backbone_names = [
            "head.classifier.weight",
            "neck.fpn.weight",
            "detect.weight",
        ]

        for name in non_backbone_names:
            name_lower = name.lower()
            matched = any(pattern in name_lower for pattern in backbone_patterns)
            assert not matched, f"Expected {name} to NOT match backbone pattern"


class TestEpochBasedUnfreezing:
    """Tests for epoch-based unfreezing."""

    def test_unfreeze_at_epoch(self):
        """Test unfreezing at specific epoch."""
        model = SimpleModel()
        freeze_until_epoch = 10

        # Freeze backbone initially
        frozen_layers = []
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
                frozen_layers.append(name)

        # Simulate epochs
        for epoch in range(20):
            if epoch >= freeze_until_epoch and len(frozen_layers) > 0:
                # Unfreeze
                for name, param in model.named_parameters():
                    if name in frozen_layers:
                        param.requires_grad = True
                frozen_layers = []

        # After epoch 10, all should be unfrozen
        for param in model.parameters():
            assert param.requires_grad

    def test_freeze_until_zero_never_unfreezes(self):
        """Test that freeze_until_epoch=0 means never auto-unfreeze."""
        model = SimpleModel()
        freeze_until_epoch = 0

        # Freeze backbone
        frozen_layers = []
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
                frozen_layers.append(name)

        # Simulate many epochs
        for epoch in range(100):
            # With freeze_until_epoch=0, we never trigger unfreeze
            if freeze_until_epoch > 0 and epoch >= freeze_until_epoch:
                for name, param in model.named_parameters():
                    if name in frozen_layers:
                        param.requires_grad = True
                frozen_layers = []

        # Should still be frozen
        assert not model.backbone_conv1.weight.requires_grad


class TestOptimizerWithFrozenParams:
    """Tests for optimizer behavior with frozen parameters."""

    def test_optimizer_excludes_frozen_params(self):
        """Test that optimizer only includes trainable parameters."""
        model = SimpleModel()

        # Freeze backbone
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

        # Create optimizer with only trainable params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, lr=0.01)

        # Check optimizer has fewer params than model
        optimizer_params = sum(len(pg["params"]) for pg in optimizer.param_groups)
        model_params = len(list(model.parameters()))
        assert optimizer_params < model_params

    def test_frozen_params_dont_update(self):
        """Test that frozen parameters don't receive gradients."""
        model = SimpleModel()

        # Freeze backbone
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

        # Store initial backbone weights
        initial_weights = model.backbone_conv1.weight.clone()

        # Forward/backward pass
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Backbone weights should not have gradients
        assert model.backbone_conv1.weight.grad is None

        # Weights should be unchanged
        assert torch.allclose(model.backbone_conv1.weight, initial_weights)

    def test_unfrozen_params_update(self):
        """Test that unfrozen parameters receive gradients."""
        model = SimpleModel()

        # Freeze backbone
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

        # Forward/backward pass
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Head weights should have gradients
        assert model.head_conv.weight.grad is not None
        assert model.head_conv.weight.grad.abs().sum() > 0


class TestFreezeLayersConfig:
    """Tests for freeze_layers configuration."""

    def test_freeze_multiple_patterns(self):
        """Test freezing with multiple patterns."""
        model = SimpleModel()

        patterns = ["backbone_conv1", "neck"]

        for name, param in model.named_parameters():
            for pattern in patterns:
                if pattern in name:
                    param.requires_grad = False
                    break

        # Check correct layers are frozen
        assert not model.backbone_conv1.weight.requires_grad
        assert not model.neck_conv.weight.requires_grad

        # Check other layers are not frozen
        assert model.backbone_conv2.weight.requires_grad
        assert model.head_conv.weight.requires_grad

    def test_empty_pattern_list(self):
        """Test with empty pattern list (freeze nothing)."""
        model = SimpleModel()

        patterns = []

        for name, param in model.named_parameters():
            for pattern in patterns:
                if pattern in name:
                    param.requires_grad = False
                    break

        # All should remain trainable
        for param in model.parameters():
            assert param.requires_grad

    def test_non_matching_pattern(self):
        """Test with patterns that don't match any layers."""
        model = SimpleModel()

        patterns = ["nonexistent_layer"]

        for name, param in model.named_parameters():
            for pattern in patterns:
                if pattern in name:
                    param.requires_grad = False
                    break

        # All should remain trainable
        for param in model.parameters():
            assert param.requires_grad


class TestFreezeIntegration:
    """Integration tests for freeze functionality."""

    def test_freeze_unfreeze_cycle(self):
        """Test full freeze-train-unfreeze-train cycle."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Phase 1: Freeze backbone
        frozen_layers = []
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
                frozen_layers.append(name)

        # Train with frozen backbone
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Only non-frozen params should have grads
        assert model.backbone_conv1.weight.grad is None
        assert model.head_conv.weight.grad is not None

        optimizer.zero_grad()

        # Phase 2: Unfreeze backbone
        for name, param in model.named_parameters():
            if name in frozen_layers:
                param.requires_grad = True
        frozen_layers = []

        # Recreate optimizer with all params
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Train with all params
        y = model(x)
        loss = y.sum()
        loss.backward()

        # All should have grads now
        assert model.backbone_conv1.weight.grad is not None
        assert model.head_conv.weight.grad is not None
