"""Tests for YOLO transforms - RandomFlip box handling."""

import pytest
import torch
from PIL import Image

from yolo.data.transforms import RandomFlip, LetterBox, Compose


class TestRandomFlipBoxes:
    """Test that RandomFlip correctly transforms bounding boxes."""

    def create_test_image_and_target(self, width=320, height=320):
        """Create a test image tensor and target with boxes."""
        # Create image tensor [C, H, W] normalized to [0, 1]
        image = torch.rand(3, height, width)

        # Create boxes in xyxy format
        # Box in top-left quadrant: [10, 10, 100, 80]
        # Box in center: [140, 140, 180, 180]
        boxes = torch.tensor([
            [10.0, 10.0, 100.0, 80.0],
            [140.0, 140.0, 180.0, 180.0],
        ])
        labels = torch.tensor([0, 1])

        return image, {"boxes": boxes, "labels": labels}

    def test_horizontal_flip_transforms_boxes(self):
        """Horizontal flip must correctly transform box x-coordinates."""
        image, target = self.create_test_image_and_target(width=320, height=320)
        w = 320

        # Original boxes
        orig_boxes = target["boxes"].clone()

        # Create flip with 100% probability
        flip = RandomFlip(lr_prob=1.0, ud_prob=0.0)

        # Force the random to flip (set seed or call multiple times)
        # Since lr_prob=1.0, it will always flip
        torch.manual_seed(42)
        image_out, target_out = flip(image, target)

        # Verify box transformation: new_x1 = w - old_x2, new_x2 = w - old_x1
        expected_boxes = orig_boxes.clone()
        expected_boxes[:, 0] = w - orig_boxes[:, 2]  # new_x1 = w - old_x2
        expected_boxes[:, 2] = w - orig_boxes[:, 0]  # new_x2 = w - old_x1

        assert torch.allclose(target_out["boxes"], expected_boxes, atol=1e-5), (
            f"Horizontal flip box mismatch.\n"
            f"Expected: {expected_boxes}\n"
            f"Got: {target_out['boxes']}"
        )

    def test_vertical_flip_transforms_boxes(self):
        """Vertical flip must correctly transform box y-coordinates."""
        image, target = self.create_test_image_and_target(width=320, height=320)
        h = 320

        # Original boxes
        orig_boxes = target["boxes"].clone()

        # Create flip with 100% probability for vertical
        flip = RandomFlip(lr_prob=0.0, ud_prob=1.0)

        torch.manual_seed(42)
        image_out, target_out = flip(image, target)

        # Verify box transformation: new_y1 = h - old_y2, new_y2 = h - old_y1
        expected_boxes = orig_boxes.clone()
        expected_boxes[:, 1] = h - orig_boxes[:, 3]  # new_y1 = h - old_y2
        expected_boxes[:, 3] = h - orig_boxes[:, 1]  # new_y2 = h - old_y1

        assert torch.allclose(target_out["boxes"], expected_boxes, atol=1e-5), (
            f"Vertical flip box mismatch.\n"
            f"Expected: {expected_boxes}\n"
            f"Got: {target_out['boxes']}"
        )

    def test_flip_preserves_box_area(self):
        """Flip must preserve box area."""
        image, target = self.create_test_image_and_target()

        # Calculate original areas
        orig_boxes = target["boxes"]
        orig_areas = (orig_boxes[:, 2] - orig_boxes[:, 0]) * (orig_boxes[:, 3] - orig_boxes[:, 1])

        # Apply both flips
        flip = RandomFlip(lr_prob=1.0, ud_prob=1.0)
        torch.manual_seed(42)
        _, target_out = flip(image, target)

        # Calculate new areas
        new_boxes = target_out["boxes"]
        new_areas = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])

        assert torch.allclose(orig_areas, new_areas, atol=1e-5), (
            f"Flip changed box areas.\n"
            f"Original: {orig_areas}\n"
            f"After flip: {new_areas}"
        )

    def test_flip_with_probability_zero(self):
        """No flip when probability is 0."""
        image, target = self.create_test_image_and_target()
        orig_boxes = target["boxes"].clone()

        # Create flip with 0% probability
        flip = RandomFlip(lr_prob=0.0, ud_prob=0.0)

        # Run multiple times to ensure no flip
        for _ in range(10):
            _, target_out = flip(image, target.copy())
            assert torch.equal(target_out["boxes"], orig_boxes), "Boxes changed with prob=0"

    def test_double_flip_restores_original(self):
        """Double flip (same direction) should restore original boxes."""
        image, target = self.create_test_image_and_target()
        orig_boxes = target["boxes"].clone()

        # Create flip with 100% probability
        flip_lr = RandomFlip(lr_prob=1.0, ud_prob=0.0)
        flip_ud = RandomFlip(lr_prob=0.0, ud_prob=1.0)

        # Double horizontal flip
        torch.manual_seed(42)
        image1, target1 = flip_lr(image, target)
        torch.manual_seed(42)
        image2, target2 = flip_lr(image1, target1)

        assert torch.allclose(target2["boxes"], orig_boxes, atol=1e-5), (
            f"Double horizontal flip should restore original.\n"
            f"Original: {orig_boxes}\n"
            f"After double flip: {target2['boxes']}"
        )

        # Double vertical flip
        torch.manual_seed(42)
        image1, target1 = flip_ud(image, {"boxes": orig_boxes.clone(), "labels": target["labels"]})
        torch.manual_seed(42)
        image2, target2 = flip_ud(image1, target1)

        assert torch.allclose(target2["boxes"], orig_boxes, atol=1e-5), (
            f"Double vertical flip should restore original.\n"
            f"Original: {orig_boxes}\n"
            f"After double flip: {target2['boxes']}"
        )

    def test_flip_handles_empty_boxes(self):
        """Flip must handle empty boxes gracefully."""
        image = torch.rand(3, 320, 320)
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }

        flip = RandomFlip(lr_prob=1.0, ud_prob=1.0)
        torch.manual_seed(42)
        _, target_out = flip(image, target)

        assert len(target_out["boxes"]) == 0
        assert len(target_out["labels"]) == 0

    def test_flip_preserves_labels(self):
        """Flip must preserve label order and values."""
        image, target = self.create_test_image_and_target()
        orig_labels = target["labels"].clone()

        flip = RandomFlip(lr_prob=1.0, ud_prob=1.0)
        torch.manual_seed(42)
        _, target_out = flip(image, target)

        assert torch.equal(target_out["labels"], orig_labels), "Labels changed after flip"

    def test_flip_boxes_stay_within_image(self):
        """Flipped boxes must stay within image bounds."""
        image, target = self.create_test_image_and_target(width=320, height=320)

        flip = RandomFlip(lr_prob=1.0, ud_prob=1.0)
        torch.manual_seed(42)
        _, target_out = flip(image, target)

        boxes = target_out["boxes"]
        assert (boxes[:, 0] >= 0).all(), "x1 < 0 after flip"
        assert (boxes[:, 1] >= 0).all(), "y1 < 0 after flip"
        assert (boxes[:, 2] <= 320).all(), "x2 > width after flip"
        assert (boxes[:, 3] <= 320).all(), "y2 > height after flip"

    def test_flip_maintains_box_validity(self):
        """Flipped boxes must have x2 > x1 and y2 > y1."""
        image, target = self.create_test_image_and_target()

        flip = RandomFlip(lr_prob=1.0, ud_prob=1.0)
        torch.manual_seed(42)
        _, target_out = flip(image, target)

        boxes = target_out["boxes"]
        assert (boxes[:, 2] > boxes[:, 0]).all(), "Invalid box: x2 <= x1"
        assert (boxes[:, 3] > boxes[:, 1]).all(), "Invalid box: y2 <= y1"


class TestLetterBoxWithBoxes:
    """Test LetterBox transform with bounding boxes."""

    def test_letterbox_scales_boxes(self):
        """LetterBox must scale boxes proportionally."""
        # Create PIL image
        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        # Box covering part of image
        boxes = torch.tensor([[100.0, 100.0, 300.0, 200.0]])
        target = {"boxes": boxes, "labels": torch.tensor([0])}

        letterbox = LetterBox(target_size=(320, 320))
        img_out, target_out = letterbox(img, target)

        # Check image size
        assert img_out.shape[1:] == (320, 320), f"Expected (320, 320), got {img_out.shape[1:]}"

        # Boxes should be scaled and shifted
        # Scale factor: min(320/640, 320/480) = 0.5
        # New image: 320x240, padded to 320x320 with top/bottom padding
        assert len(target_out["boxes"]) == 1
        # Box should be within bounds
        assert (target_out["boxes"][:, 2] <= 320).all()
        assert (target_out["boxes"][:, 3] <= 320).all()

    def test_letterbox_handles_empty_boxes(self):
        """LetterBox must handle empty boxes."""
        img = Image.new("RGB", (640, 480))
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }

        letterbox = LetterBox(target_size=(320, 320))
        img_out, target_out = letterbox(img, target)

        assert len(target_out["boxes"]) == 0
        assert len(target_out["labels"]) == 0
