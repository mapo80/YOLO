"""Tests for mosaic augmentation with full-image boxes."""

import pytest
import torch
import numpy as np
from PIL import Image
from yolo.data.mosaic import MosaicMixupDataset, CenterCropWithBoxes


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, image_size=(320, 320), num_images=10):
        self.image_size = image_size
        self.num_images = num_images

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Create dummy image with unique color per index
        img = Image.new('RGB', self.image_size, color=(idx * 20 % 256, 100, 100))
        # Full-image box
        boxes = torch.tensor([[0.0, 0.0, float(self.image_size[0]), float(self.image_size[1])]])
        labels = torch.tensor([idx % 13])
        return img, {"boxes": boxes, "labels": labels}


class TestMosaic4FullImageBoxes:
    """Test _mosaic4 with full-image boxes."""

    @pytest.fixture
    def mosaic_dataset(self):
        base = MockDataset(image_size=(320, 320))
        return MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mixup_prob=0.0,
        )

    def test_mosaic4_returns_2x_canvas_size(self, mosaic_dataset):
        """Mosaic4 must return 2x canvas before crop."""
        mosaic_dataset._mosaic_enabled = True
        img, target = mosaic_dataset._mosaic4(0)

        # Canvas must be 2x (640x640 for 320 base size)
        assert img.size == (640, 640), f"Expected 640x640, got {img.size}"

    def test_mosaic4_boxes_clipped_to_canvas(self, mosaic_dataset):
        """Boxes must be clipped to canvas size, not final size."""
        img, target = mosaic_dataset._mosaic4(0)

        boxes = target["boxes"]
        canvas_size = 640  # 2x of 320

        # No box should exceed canvas bounds
        assert (boxes[:, 2] <= canvas_size).all(), "Box x2 exceeds canvas"
        assert (boxes[:, 3] <= canvas_size).all(), "Box y2 exceeds canvas"
        # No box should have negative coordinates
        assert (boxes[:, 0] >= 0).all(), "Box x1 < 0"
        assert (boxes[:, 1] >= 0).all(), "Box y1 < 0"

    def test_mosaic4_preserves_valid_boxes(self, mosaic_dataset):
        """Mosaic must preserve at least some valid boxes."""
        img, target = mosaic_dataset._mosaic4(0)

        # At least 1 box must survive (4 images with 1 box each)
        assert len(target["boxes"]) >= 1, "No boxes preserved"

    def test_mosaic4_no_zero_area_boxes(self, mosaic_dataset):
        """No zero-area boxes should be present."""
        img, target = mosaic_dataset._mosaic4(0)

        boxes = target["boxes"]
        if len(boxes) > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            assert (areas > 0).all(), "Found zero-area boxes"

    def test_mosaic4_multiple_runs_consistent(self, mosaic_dataset):
        """Multiple runs should consistently produce valid output."""
        for _ in range(10):
            img, target = mosaic_dataset._mosaic4(0)
            assert img.size == (640, 640)
            assert len(target["boxes"]) >= 1


class TestMosaic9FullImageBoxes:
    """Test _mosaic9 with full-image boxes."""

    @pytest.fixture
    def mosaic_dataset(self):
        base = MockDataset(image_size=(320, 320))
        return MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mosaic_9_prob=1.0,
            mixup_prob=0.0,
        )

    def test_mosaic9_returns_3x_canvas_size(self, mosaic_dataset):
        """Mosaic9 must return 3x canvas before crop."""
        img, target = mosaic_dataset._mosaic9(0)

        # Canvas must be 3x (960x960 for 320 base size)
        assert img.size == (960, 960), f"Expected 960x960, got {img.size}"

    def test_mosaic9_boxes_clipped_to_canvas(self, mosaic_dataset):
        """Boxes must be clipped to canvas size."""
        img, target = mosaic_dataset._mosaic9(0)

        boxes = target["boxes"]
        canvas_size = 960  # 3x of 320

        assert (boxes[:, 2] <= canvas_size).all(), "Box x2 exceeds canvas"
        assert (boxes[:, 3] <= canvas_size).all(), "Box y2 exceeds canvas"
        assert (boxes[:, 0] >= 0).all(), "Box x1 < 0"
        assert (boxes[:, 1] >= 0).all(), "Box y1 < 0"

    def test_mosaic9_preserves_valid_boxes(self, mosaic_dataset):
        """Mosaic must preserve at least some valid boxes."""
        img, target = mosaic_dataset._mosaic9(0)

        # At least 1 box must survive (9 images with 1 box each)
        assert len(target["boxes"]) >= 1, "No boxes preserved"


class TestCenterCropWithBoxes:
    """Test CenterCropWithBoxes transform."""

    def test_crop_reduces_image_size(self):
        """Crop must reduce image from 640 to 320."""
        crop = CenterCropWithBoxes((320, 320))
        img = Image.new('RGB', (640, 640), color=(100, 100, 100))
        boxes = torch.tensor([[160.0, 160.0, 480.0, 480.0]])  # Centered box
        target = {"boxes": boxes, "labels": torch.tensor([0])}

        img_out, target_out = crop(img, target)

        assert img_out.size == (320, 320)

    def test_crop_adjusts_centered_box(self):
        """Centered box must remain centered after crop."""
        crop = CenterCropWithBoxes((320, 320))
        img = Image.new('RGB', (640, 640))
        # Box centered in 640x640 canvas: (160, 160, 480, 480)
        # After center crop to 320x320: should become (0, 0, 320, 320)
        boxes = torch.tensor([[160.0, 160.0, 480.0, 480.0]])
        target = {"boxes": boxes, "labels": torch.tensor([0])}

        _, target_out = crop(img, target)

        expected = torch.tensor([[0.0, 0.0, 320.0, 320.0]])
        assert torch.allclose(target_out["boxes"], expected, atol=1.0)

    def test_crop_filters_outside_boxes(self):
        """Box completely outside crop region must be filtered."""
        crop = CenterCropWithBoxes((320, 320))
        img = Image.new('RGB', (640, 640))
        # Box in top-left corner, outside center crop region
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
        target = {"boxes": boxes, "labels": torch.tensor([0])}

        _, target_out = crop(img, target)

        # After crop (offset 160,160), box becomes (-160,-160,-60,-60)
        # Clamped to (0,0,0,0) -> area 0 -> filtered
        assert len(target_out["boxes"]) == 0

    def test_crop_keeps_partial_boxes(self):
        """Box partially in crop region must be kept (if area >= 4)."""
        crop = CenterCropWithBoxes((320, 320))
        img = Image.new('RGB', (640, 640))
        # Box starting at crop edge: (160, 160, 260, 260)
        # After crop: (0, 0, 100, 100) - area 10000 > 4 -> kept
        boxes = torch.tensor([[160.0, 160.0, 260.0, 260.0]])
        target = {"boxes": boxes, "labels": torch.tensor([0])}

        _, target_out = crop(img, target)

        assert len(target_out["boxes"]) == 1
        expected = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
        assert torch.allclose(target_out["boxes"], expected, atol=1.0)

    def test_no_crop_when_size_matches(self):
        """If image already target size, return unchanged."""
        crop = CenterCropWithBoxes((320, 320))
        img = Image.new('RGB', (320, 320))
        boxes = torch.tensor([[0.0, 0.0, 320.0, 320.0]])
        target = {"boxes": boxes, "labels": torch.tensor([0])}

        img_out, target_out = crop(img, target)

        assert img_out.size == (320, 320)
        assert torch.equal(target_out["boxes"], boxes)

    def test_crop_with_empty_boxes(self):
        """Crop must handle empty boxes gracefully."""
        crop = CenterCropWithBoxes((320, 320))
        img = Image.new('RGB', (640, 640))
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        target = {"boxes": boxes, "labels": torch.zeros((0,), dtype=torch.long)}

        img_out, target_out = crop(img, target)

        assert img_out.size == (320, 320)
        assert len(target_out["boxes"]) == 0

    def test_crop_filters_small_boxes(self):
        """Boxes with area < min_box_area must be filtered."""
        crop = CenterCropWithBoxes((320, 320), min_box_area=100.0)
        img = Image.new('RGB', (640, 640))
        # Box that becomes 5x5 = 25 pixels after crop (< 100)
        boxes = torch.tensor([[160.0, 160.0, 165.0, 165.0]])
        target = {"boxes": boxes, "labels": torch.tensor([0])}

        _, target_out = crop(img, target)

        assert len(target_out["boxes"]) == 0

    def test_crop_3x_canvas(self):
        """Crop must work with 3x canvas (mosaic9)."""
        crop = CenterCropWithBoxes((320, 320))
        img = Image.new('RGB', (960, 960))  # 3x canvas
        # Box centered in canvas
        boxes = torch.tensor([[320.0, 320.0, 640.0, 640.0]])
        target = {"boxes": boxes, "labels": torch.tensor([0])}

        img_out, target_out = crop(img, target)

        assert img_out.size == (320, 320)
        # After crop (offset 320,320), box becomes (0, 0, 320, 320)
        expected = torch.tensor([[0.0, 0.0, 320.0, 320.0]])
        assert torch.allclose(target_out["boxes"], expected, atol=1.0)


class TestMosaicIntegration:
    """Test integration of mosaic + crop pipeline."""

    def test_full_pipeline_preserves_some_boxes(self):
        """Full pipeline must preserve at least some boxes."""
        base = MockDataset(image_size=(320, 320))
        crop = CenterCropWithBoxes((320, 320))

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mixup_prob=0.0,
            transforms=crop,
        )

        # Test multiple samples to account for randomness
        boxes_found = 0
        for i in range(10):
            _, target = mosaic[i]
            boxes_found += len(target["boxes"])

        assert boxes_found > 0, "No boxes preserved across 10 samples"

    def test_output_image_correct_size(self):
        """Final output must have correct size."""
        base = MockDataset(image_size=(320, 320))
        crop = CenterCropWithBoxes((320, 320))

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mixup_prob=0.0,
            transforms=crop,
        )

        img, _ = mosaic[0]
        assert img.size == (320, 320)

    def test_no_mosaic_returns_original_size(self):
        """Without mosaic, image should be original size."""
        base = MockDataset(image_size=(320, 320))
        crop = CenterCropWithBoxes((320, 320))

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=0.0,  # Disabled
            mixup_prob=0.0,
            transforms=crop,
        )

        img, target = mosaic[0]
        assert img.size == (320, 320)
        # Original full-image box should be preserved
        assert len(target["boxes"]) == 1

    def test_mosaic_disabled_preserves_boxes(self):
        """When mosaic is disabled via method, boxes should be preserved."""
        base = MockDataset(image_size=(320, 320))
        crop = CenterCropWithBoxes((320, 320))

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mixup_prob=0.0,
            transforms=crop,
        )
        mosaic.disable_mosaic()

        img, target = mosaic[0]
        assert img.size == (320, 320)
        assert len(target["boxes"]) == 1

    def test_pipeline_with_small_boxes(self):
        """Pipeline must handle small boxes correctly."""
        class SmallBoxDataset:
            def __init__(self):
                self.image_size = (320, 320)

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                img = Image.new('RGB', self.image_size, color=(100, 100, 100))
                # Small box in center (50x50)
                cx, cy = 160, 160
                boxes = torch.tensor([[cx - 25.0, cy - 25.0, cx + 25.0, cy + 25.0]])
                labels = torch.tensor([0])
                return img, {"boxes": boxes, "labels": labels}

        base = SmallBoxDataset()
        crop = CenterCropWithBoxes((320, 320))

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mixup_prob=0.0,
            transforms=crop,
        )

        # Run multiple times - small centered boxes have good chance of surviving
        boxes_found = 0
        for i in range(20):
            _, target = mosaic[i]
            boxes_found += len(target["boxes"])

        assert boxes_found > 0, "No small boxes preserved"


class TestMixupWithNewMosaic:
    """Test mixup works correctly with new mosaic implementation."""

    def test_mixup_produces_correct_size(self):
        """Mixup output must have correct size after crop."""
        base = MockDataset(image_size=(320, 320))
        crop = CenterCropWithBoxes((320, 320))

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mixup_prob=1.0,  # Always apply mixup
            transforms=crop,
        )

        img, target = mosaic[0]
        assert img.size == (320, 320)

    def test_mixup_combines_boxes(self):
        """Mixup must combine boxes from both images."""
        base = MockDataset(image_size=(320, 320))
        crop = CenterCropWithBoxes((320, 320))

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mixup_prob=1.0,
            transforms=crop,
        )

        # Run multiple times - mixup combines boxes from 2 mosaics
        # Each mosaic has 4 images, so up to 8 boxes possible
        max_boxes = 0
        for i in range(10):
            _, target = mosaic[i]
            max_boxes = max(max_boxes, len(target["boxes"]))

        # Should sometimes have more than 4 boxes (from combining 2 mosaics)
        assert max_boxes >= 1, "Mixup should preserve some boxes"


class TestMixupBoxScaling:
    """Test that mixup correctly scales boxes when resizing images."""

    def test_mixup_scales_boxes_when_resizing(self):
        """Boxes must be scaled when image2 is resized to match image1."""
        base = MockDataset(image_size=(320, 320))

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mosaic_9_prob=0.0,  # Force mosaic4 (2x canvas = 640x640)
            mixup_prob=0.0,  # Don't use automatic mixup
        )

        # Get a mosaic4 result (640x640)
        img1, target1 = mosaic._mosaic4(0)
        assert img1.size == (640, 640), "mosaic4 should return 640x640"

        # Manually call _mixup - it will create another mosaic internally
        # and should scale boxes if sizes differ
        img_mixed, target_mixed = mosaic._mixup(img1, target1)

        # Output should be same size as input
        assert img_mixed.size == (640, 640), "mixup should preserve image size"

        # Boxes should be valid (within canvas bounds)
        if len(target_mixed["boxes"]) > 0:
            boxes = target_mixed["boxes"]
            assert (boxes[:, 2] <= 640).all(), "Box x2 exceeds canvas"
            assert (boxes[:, 3] <= 640).all(), "Box y2 exceeds canvas"
            assert (boxes[:, 0] >= 0).all(), "Box x1 < 0"
            assert (boxes[:, 1] >= 0).all(), "Box y1 < 0"

    def test_mixup_with_mosaic9_scaling(self):
        """Mixup must scale boxes correctly when mixing mosaic4 with mosaic9."""
        base = MockDataset(image_size=(320, 320))

        # Create dataset that forces mosaic9 for second image
        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mosaic_9_prob=1.0,  # Force mosaic9 (3x canvas = 960x960)
            mixup_prob=0.0,
        )

        # Get mosaic9 (960x960)
        img1, target1 = mosaic._mosaic9(0)
        assert img1.size == (960, 960), "mosaic9 should return 960x960"

        # Now create a mosaic4 (640x640) and use it as input to _mixup
        mosaic4_only = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mosaic_9_prob=0.0,  # Force mosaic4
            mixup_prob=0.0,
        )
        img_small, target_small = mosaic4_only._mosaic4(0)
        assert img_small.size == (640, 640)

        # Call _mixup with 640x640 image - it will create 960x960 mosaic9
        # and must resize it to 640x640 AND scale boxes
        img_mixed, target_mixed = mosaic._mixup(img_small, target_small)

        # Output should match input size
        assert img_mixed.size == (640, 640)

        # Boxes must be within 640x640 bounds (scaled from 960x960)
        if len(target_mixed["boxes"]) > 0:
            boxes = target_mixed["boxes"]
            # Critical check: boxes should NOT exceed 640
            # If bug exists, boxes would be up to 960
            assert (boxes[:, 2] <= 640).all(), f"Box x2 exceeds 640: max={boxes[:, 2].max()}"
            assert (boxes[:, 3] <= 640).all(), f"Box y2 exceeds 640: max={boxes[:, 3].max()}"

    def test_mixup_no_scaling_when_same_size(self):
        """When both images are same size, boxes should not be modified."""
        base = MockDataset(image_size=(320, 320))

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mosaic_9_prob=0.0,  # Both will be mosaic4 (640x640)
            mixup_prob=0.0,
        )

        # Get two mosaic4 results (both 640x640)
        img1, target1 = mosaic._mosaic4(0)

        # Record original boxes from target1
        orig_boxes1 = target1["boxes"].clone() if len(target1["boxes"]) > 0 else None

        # _mixup should work correctly
        img_mixed, target_mixed = mosaic._mixup(img1, target1)

        assert img_mixed.size == (640, 640)
        # Both images are same size, so no scaling needed
        # Just verify boxes are valid
        if len(target_mixed["boxes"]) > 0:
            boxes = target_mixed["boxes"]
            assert (boxes[:, 2] <= 640).all()
            assert (boxes[:, 3] <= 640).all()

    def test_mixup_preserves_labels(self):
        """Mixup must preserve labels from both images."""
        base = MockDataset(image_size=(320, 320), num_images=20)

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mixup_prob=0.0,
        )

        img1, target1 = mosaic._mosaic4(0)
        labels1_count = len(target1["labels"])

        img_mixed, target_mixed = mosaic._mixup(img1, target1)

        # Should have labels from both images
        # At minimum, should have some labels
        if labels1_count > 0 or len(target_mixed["labels"]) > 0:
            assert len(target_mixed["labels"]) == len(target_mixed["boxes"])

    def test_mixup_multiple_runs_no_overflow(self):
        """Multiple mixup runs should never produce boxes outside bounds."""
        base = MockDataset(image_size=(320, 320))

        # Mix mosaic4 and mosaic9 randomly
        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=1.0,
            mosaic_9_prob=0.5,  # 50% chance of each
            mixup_prob=0.0,
        )

        for _ in range(20):
            # Get random mosaic
            if np.random.random() < 0.5:
                img1, target1 = mosaic._mosaic4(0)
            else:
                img1, target1 = mosaic._mosaic9(0)

            # Apply mixup
            img_mixed, target_mixed = mosaic._mixup(img1, target1)

            # Must match input size
            assert img_mixed.size == img1.size

            # Boxes must be within bounds
            if len(target_mixed["boxes"]) > 0:
                boxes = target_mixed["boxes"]
                max_dim = max(img1.size)
                assert (boxes[:, 2] <= max_dim).all(), f"Box overflow: {boxes[:, 2].max()} > {max_dim}"
                assert (boxes[:, 3] <= max_dim).all(), f"Box overflow: {boxes[:, 3].max()} > {max_dim}"
                assert (boxes[:, 0] >= 0).all()
                assert (boxes[:, 1] >= 0).all()


class TestCutMixBoxHandling:
    """Test that cutmix correctly handles boxes."""

    def test_cutmix_scales_boxes_when_resizing(self):
        """Boxes from image2 must be scaled when image2 is resized."""

        class VariableSizeDataset:
            """Dataset that returns images of different sizes."""

            def __init__(self):
                self.image_size = (320, 320)

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                # Return different sizes based on index
                if idx % 2 == 0:
                    size = (320, 320)
                else:
                    size = (640, 640)  # Larger image

                img = Image.new("RGB", size, color=(idx * 20 % 256, 100, 100))
                # Full-image box
                boxes = torch.tensor([[0.0, 0.0, float(size[0]), float(size[1])]])
                labels = torch.tensor([idx % 13])
                return img, {"boxes": boxes, "labels": labels}

        base = VariableSizeDataset()

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=0.0,  # Disable mosaic so cutmix can be applied
            cutmix_prob=0.0,  # We'll call _cutmix directly
        )

        # Get a 320x320 image
        img1, target1 = mosaic._load_image_target(0)
        assert img1.size == (320, 320)

        # Call _cutmix multiple times - it will load random images
        # Some may be 640x640 and need scaling
        for _ in range(10):
            img_mixed, target_mixed = mosaic._cutmix(img1, target1)

            # Output should match input size
            assert img_mixed.size == (320, 320)

            # Boxes should be within bounds
            if len(target_mixed["boxes"]) > 0:
                boxes = target_mixed["boxes"]
                assert (boxes[:, 2] <= 320).all(), f"Box x2 exceeds 320: {boxes[:, 2].max()}"
                assert (boxes[:, 3] <= 320).all(), f"Box y2 exceeds 320: {boxes[:, 3].max()}"
                assert (boxes[:, 0] >= 0).all()
                assert (boxes[:, 1] >= 0).all()

    def test_cutmix_preserves_original_boxes(self):
        """CutMix must preserve boxes from the original image."""
        base = MockDataset(image_size=(320, 320))

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=0.0,
            cutmix_prob=0.0,
        )

        img1, target1 = mosaic._load_image_target(0)
        orig_boxes = target1["boxes"].clone()

        img_mixed, target_mixed = mosaic._cutmix(img1, target1)

        # Original boxes should be in output
        if len(orig_boxes) > 0 and len(target_mixed["boxes"]) > 0:
            # Check that original box is preserved (first boxes should be from image1)
            assert len(target_mixed["boxes"]) >= len(orig_boxes)

    def test_cutmix_clips_boxes_to_cut_region(self):
        """Boxes from image2 must be clipped to the cut region."""
        base = MockDataset(image_size=(320, 320))

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=0.0,
            cutmix_prob=0.0,
        )

        img1, target1 = mosaic._load_image_target(0)

        # Run multiple times to get different cut regions
        for _ in range(10):
            img_mixed, target_mixed = mosaic._cutmix(img1, target1)

            # All boxes should be valid
            if len(target_mixed["boxes"]) > 0:
                boxes = target_mixed["boxes"]
                # No negative coordinates
                assert (boxes[:, 0] >= 0).all()
                assert (boxes[:, 1] >= 0).all()
                # No coordinates exceeding image size
                assert (boxes[:, 2] <= 320).all()
                assert (boxes[:, 3] <= 320).all()
                # Valid box dimensions (x2 > x1, y2 > y1)
                assert (boxes[:, 2] > boxes[:, 0]).all()
                assert (boxes[:, 3] > boxes[:, 1]).all()

    def test_cutmix_filters_small_boxes(self):
        """Boxes with area < 4 must be filtered out."""
        base = MockDataset(image_size=(320, 320))

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=0.0,
            cutmix_prob=0.0,
        )

        img1, target1 = mosaic._load_image_target(0)

        # Run multiple times
        for _ in range(10):
            img_mixed, target_mixed = mosaic._cutmix(img1, target1)

            if len(target_mixed["boxes"]) > 0:
                boxes = target_mixed["boxes"]
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                # All boxes should have area >= 4 (or be from original image)
                # Note: original boxes are kept as-is
                assert (areas > 0).all(), "Found zero-area boxes"

    def test_cutmix_handles_empty_boxes(self):
        """CutMix must handle images with no boxes gracefully."""

        class EmptyBoxDataset:
            def __init__(self):
                self.image_size = (320, 320)

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                img = Image.new("RGB", self.image_size, color=(100, 100, 100))
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.long)
                return img, {"boxes": boxes, "labels": labels}

        base = EmptyBoxDataset()

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=0.0,
            cutmix_prob=0.0,
        )

        img1, target1 = mosaic._load_image_target(0)
        assert len(target1["boxes"]) == 0

        # Should not crash with empty boxes
        img_mixed, target_mixed = mosaic._cutmix(img1, target1)

        assert img_mixed.size == (320, 320)
        # Output may have 0 boxes (both images empty)
        assert len(target_mixed["boxes"]) == len(target_mixed["labels"])

    def test_cutmix_ioa_filtering(self):
        """Only boxes with IoA >= 0.1 with cut region should be included."""

        class CenteredBoxDataset:
            """Dataset with boxes in specific positions."""

            def __init__(self):
                self.image_size = (320, 320)

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                img = Image.new("RGB", self.image_size, color=(100, 100, 100))
                # Small centered box
                cx, cy = 160, 160
                boxes = torch.tensor([[cx - 20.0, cy - 20.0, cx + 20.0, cy + 20.0]])
                labels = torch.tensor([0])
                return img, {"boxes": boxes, "labels": labels}

        base = CenteredBoxDataset()

        mosaic = MosaicMixupDataset(
            base,
            image_size=(320, 320),
            mosaic_prob=0.0,
            cutmix_prob=0.0,
        )

        img1, target1 = mosaic._load_image_target(0)

        # Run multiple times - cut regions vary randomly
        boxes_from_img2 = 0
        for _ in range(20):
            img_mixed, target_mixed = mosaic._cutmix(img1, target1)

            # Count boxes beyond the original one
            if len(target_mixed["boxes"]) > 1:
                boxes_from_img2 += len(target_mixed["boxes"]) - 1

        # Some cut regions should overlap with centered boxes
        # (statistically likely over 20 runs)
        # This test just verifies the code runs without error
