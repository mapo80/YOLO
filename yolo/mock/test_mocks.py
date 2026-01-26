#!/usr/bin/env python3
"""
Test script for mock components.

Run this to verify mock implementations are working correctly:
    cd yolo-mit
    YOLO_MOCK_VERBOSE=1 python -m yolo.mock.test_mocks

This creates synthetic data and runs through all mock components.
"""

import os
import sys

# Enable verbose logging for testing
os.environ["YOLO_MOCK_VERBOSE"] = "1"
os.environ["YOLO_MOCK_LOG_FREQ"] = "1"  # Log every call

import torch
import torch.nn as nn

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_mock_bce_loss():
    """Test MockBCELoss."""
    print("\n" + "=" * 60)
    print("Testing MockBCELoss")
    print("=" * 60)

    from yolo.mock.losses import MockBCELoss

    bce = MockBCELoss()

    # Create test data
    batch_size = 2
    num_anchors = 100
    num_classes = 13

    # Predictions (logits)
    predicts_cls = torch.randn(batch_size, num_anchors, num_classes)

    # Soft targets (simulating what BoxMatcher produces)
    targets_cls = torch.zeros(batch_size, num_anchors, num_classes)
    # Set some soft targets for a few anchors
    targets_cls[0, 10, 5] = 0.8  # Soft target
    targets_cls[0, 20, 3] = 0.6
    targets_cls[1, 15, 7] = 0.75

    # Normalization factor (sum of soft targets)
    cls_norm = targets_cls.sum()

    # Run loss
    loss = bce(predicts_cls, targets_cls, cls_norm)
    print(f"\nFinal BCE Loss: {loss.item():.4f}")


def test_mock_box_matcher():
    """Test MockBoxMatcher."""
    print("\n" + "=" * 60)
    print("Testing MockBoxMatcher")
    print("=" * 60)

    from yolo.mock.matcher import MockBoxMatcher
    from yolo.utils.bounding_box_utils import generate_anchors
    from dataclasses import dataclass

    @dataclass
    class MockVec2Box:
        anchor_grid: torch.Tensor
        scaler: torch.Tensor

    @dataclass
    class MatcherConfig:
        iou: str = "ciou"
        topk: int = 10
        factor: dict = None

        def __iter__(self):
            return iter(["iou", "topk", "factor"])

        def __getitem__(self, key):
            return getattr(self, key)

    # Create anchor grid
    image_size = [320, 320]
    strides = [8, 16, 32]
    anchor_grid, scaler = generate_anchors(image_size, strides)

    vec2box = MockVec2Box(anchor_grid=anchor_grid, scaler=scaler)

    cfg = MatcherConfig(
        iou="ciou",
        topk=10,
        factor={"iou": 6.0, "cls": 0.5},
    )

    matcher = MockBoxMatcher(cfg, class_num=13, vec2box=vec2box, reg_max=16)

    # Create test data
    batch_size = 2
    num_anchors = anchor_grid.shape[0]
    num_classes = 13

    # Targets: [batch, num_targets, 5] (class, x1, y1, x2, y2)
    targets = torch.zeros(batch_size, 3, 5)
    targets[0, 0] = torch.tensor([5, 50, 50, 150, 150])  # Class 5, box in image
    targets[0, 1] = torch.tensor([3, 100, 100, 200, 200])
    targets[1, 0] = torch.tensor([7, 80, 80, 180, 180])

    # Predictions
    predict_cls = torch.randn(batch_size, num_anchors, num_classes)
    predict_box = torch.zeros(batch_size, num_anchors, 4)
    # Set some predicted boxes near targets
    predict_box[:, :, 0] = anchor_grid[:, 0] - 20
    predict_box[:, :, 1] = anchor_grid[:, 1] - 20
    predict_box[:, :, 2] = anchor_grid[:, 0] + 20
    predict_box[:, :, 3] = anchor_grid[:, 1] + 20

    # Run matcher
    anchor_matched_targets, valid_mask = matcher(targets, (predict_cls, predict_box))

    print(f"\nMatcher Output:")
    print(f"  anchor_matched_targets shape: {anchor_matched_targets.shape}")
    print(f"  valid_mask shape: {valid_mask.shape}")
    print(f"  valid_mask sum: {valid_mask.sum()}")

    # Extract soft targets
    soft_targets = anchor_matched_targets[..., :num_classes]
    print(f"\nSoft targets:")
    print(f"  min: {soft_targets.min():.4f}")
    print(f"  max: {soft_targets.max():.4f}")
    mean_val = soft_targets[soft_targets > 0].mean().item() if (soft_targets > 0).any() else 0.0
    print(f"  mean (non-zero): {mean_val:.4f}")


def test_mock_yolo_loss():
    """Test full MockYOLOLoss pipeline."""
    print("\n" + "=" * 60)
    print("Testing MockYOLOLoss (full pipeline)")
    print("=" * 60)

    from yolo.mock.losses import MockYOLOLoss
    from yolo.utils.bounding_box_utils import generate_anchors
    from dataclasses import dataclass

    @dataclass
    class MockVec2Box:
        anchor_grid: torch.Tensor
        scaler: torch.Tensor
        strides: list

        def __call__(self, predicts):
            # Simulate vec2box output
            from einops import rearrange
            preds_cls, preds_anc, preds_box = [], [], []
            for layer_output in predicts:
                pred_cls, pred_anc, pred_box = layer_output
                preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
                preds_anc.append(rearrange(pred_anc, "B A R h w -> B (h w) R A"))
                preds_box.append(rearrange(pred_box, "B X h w -> B (h w) X"))
            preds_cls = torch.concat(preds_cls, dim=1)
            preds_anc = torch.concat(preds_anc, dim=1)
            preds_box = torch.concat(preds_box, dim=1)

            pred_LTRB = preds_box * self.scaler.view(1, -1, 1)
            lt, rb = pred_LTRB.chunk(2, dim=-1)
            preds_box = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)
            return preds_cls, preds_anc, preds_box

    # Setup
    image_size = [320, 320]
    strides = [8, 16, 32]
    num_classes = 13
    reg_max = 16

    anchor_grid, scaler = generate_anchors(image_size, strides)
    vec2box = MockVec2Box(anchor_grid=anchor_grid, scaler=scaler, strides=strides)

    # Create loss
    loss_fn = MockYOLOLoss(
        vec2box=vec2box,
        class_num=num_classes,
        reg_max=reg_max,
        box_weight=7.5,
        cls_weight=0.5,
        dfl_weight=1.5,
    )

    # Create model outputs (simulate 3 detection heads)
    batch_size = 2
    outputs = {"Main": []}
    for stride in strides:
        h, w = image_size[0] // stride, image_size[1] // stride
        pred_cls = torch.randn(batch_size, num_classes, h, w)
        pred_anc = torch.randn(batch_size, 4, reg_max, h, w)
        pred_box = torch.randn(batch_size, 4, h, w).abs() * 10  # Positive LTRB values
        outputs["Main"].append((pred_cls, pred_anc, pred_box))

    # Create targets
    targets = [
        {"boxes": torch.tensor([[50, 50, 150, 150], [100, 100, 200, 200]]),
         "labels": torch.tensor([5, 3])},
        {"boxes": torch.tensor([[80, 80, 180, 180]]),
         "labels": torch.tensor([7])},
    ]

    # Run loss
    total_loss, loss_dict = loss_fn(outputs, targets)

    print(f"\nLoss Output:")
    print(f"  total_loss: {total_loss.item():.4f}")
    print(f"  cls_loss: {loss_dict['cls_loss'].item():.4f}")
    print(f"  box_loss: {loss_dict['box_loss'].item():.4f}")
    print(f"  dfl_loss: {loss_dict['dfl_loss'].item():.4f}")


def test_mock_ema():
    """Test MockModelEMA."""
    print("\n" + "=" * 60)
    print("Testing MockModelEMA")
    print("=" * 60)

    from yolo.mock.ema import MockModelEMA

    # Create simple model
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
    )

    # Create EMA
    ema = MockModelEMA(model, decay=0.9999, tau=2000)

    # Simulate some training updates
    for i in range(5):
        # Simulate gradient update
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        # Update EMA
        ema.update(model)

    print("\nEMA test complete")


def main():
    print("\n" + "=" * 60)
    print("YOLO Mock Components Test Suite")
    print("=" * 60)

    # Print mock status
    from yolo.mock.integration import print_mock_status
    print_mock_status()

    # Run tests
    test_mock_bce_loss()
    test_mock_box_matcher()
    test_mock_yolo_loss()
    test_mock_ema()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
