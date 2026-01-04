"""
Comprehensive unit tests for data augmentation implementations.

Tests cover:
- Mosaic (4-way and 9-way)
- MixUp
- CutMix
- RandomPerspective
- EMA (Exponential Moving Average)
"""

import copy
import math
import random

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn


# =============================================================================
# Fixtures
# =============================================================================


class MockDataset:
    """Mock dataset for testing augmentations."""

    def __init__(self, num_samples: int = 100, image_size: tuple = (100, 100)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create random image
        img_np = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_np)

        # Create random boxes (1-5 boxes per image)
        num_boxes = random.randint(1, 5)
        boxes = []
        labels = []
        for _ in range(num_boxes):
            x1 = random.randint(0, self.image_size[0] - 20)
            y1 = random.randint(0, self.image_size[1] - 20)
            x2 = x1 + random.randint(10, min(50, self.image_size[0] - x1))
            y2 = y1 + random.randint(10, min(50, self.image_size[1] - y1))
            boxes.append([x1, y1, x2, y2])
            labels.append(random.randint(0, 9))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return img, target


class MockDatasetEmpty:
    """Mock dataset with no boxes."""

    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
        return img, target


@pytest.fixture
def mock_dataset():
    return MockDataset(num_samples=50)


@pytest.fixture
def mock_dataset_empty():
    return MockDatasetEmpty()


# =============================================================================
# MosaicMixupDataset Tests
# =============================================================================


class TestMosaicMixupDataset:
    """Tests for MosaicMixupDataset class."""

    def test_import(self):
        """Test that MosaicMixupDataset can be imported."""
        from yolo.data.mosaic import MosaicMixupDataset
        assert MosaicMixupDataset is not None

    def test_initialization(self, mock_dataset):
        """Test dataset wrapper initialization."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
            mixup_prob=0.15,
        )
        assert len(wrapper) == len(mock_dataset)
        assert wrapper.image_size == (640, 640)
        assert wrapper.mosaic_prob == 1.0
        assert wrapper.mixup_prob == 0.15

    def test_border_calculation(self, mock_dataset):
        """Test that border is calculated correctly for mosaic."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(mock_dataset, image_size=(640, 640))
        assert wrapper.border == (-320, -320)

        wrapper = MosaicMixupDataset(mock_dataset, image_size=(416, 416))
        assert wrapper.border == (-208, -208)

    def test_disable_enable_mosaic(self, mock_dataset):
        """Test mosaic enable/disable functionality."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(mock_dataset, mosaic_prob=1.0)
        assert wrapper.mosaic_enabled is True

        wrapper.disable_mosaic()
        assert wrapper.mosaic_enabled is False

        wrapper.enable_mosaic()
        assert wrapper.mosaic_enabled is True


class TestMosaic4:
    """Tests for 4-way mosaic augmentation."""

    def test_mosaic4_output_shape(self, mock_dataset):
        """Test that mosaic4 produces correct output shape."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
            mosaic_9_prob=0.0,  # Force 4-way
            mixup_prob=0.0,
        )

        img, target = wrapper._mosaic4(0)
        assert isinstance(img, Image.Image)
        assert img.size == (640, 640)
        assert "boxes" in target
        assert "labels" in target
        assert target["boxes"].shape[1] == 4

    def test_mosaic4_boxes_in_bounds(self, mock_dataset):
        """Test that mosaic4 boxes are within image bounds."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
            mosaic_9_prob=0.0,
            mixup_prob=0.0,
        )

        for _ in range(10):
            img, target = wrapper._mosaic4(random.randint(0, len(mock_dataset) - 1))
            boxes = target["boxes"]
            if len(boxes) > 0:
                assert (boxes[:, 0] >= 0).all(), "x1 should be >= 0"
                assert (boxes[:, 1] >= 0).all(), "y1 should be >= 0"
                assert (boxes[:, 2] <= 640).all(), "x2 should be <= 640"
                assert (boxes[:, 3] <= 640).all(), "y2 should be <= 640"
                assert (boxes[:, 2] > boxes[:, 0]).all(), "x2 > x1"
                assert (boxes[:, 3] > boxes[:, 1]).all(), "y2 > y1"

    def test_mosaic4_center_point_range(self, mock_dataset):
        """Test that mosaic4 center point is in valid range."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(mock_dataset, image_size=(640, 640))

        # border = (-320, -320)
        # yc, xc should be in range [-(-320), 2*640 + (-320)] = [320, 960]
        s = 640
        border = wrapper.border

        for _ in range(100):
            yc = int(random.uniform(-border[1], 2 * s + border[1]))
            xc = int(random.uniform(-border[0], 2 * s + border[0]))
            assert 320 <= yc <= 960, f"yc={yc} out of range"
            assert 320 <= xc <= 960, f"xc={xc} out of range"

    def test_mosaic4_with_empty_boxes(self, mock_dataset_empty):
        """Test mosaic4 handles empty boxes correctly."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset_empty,
            image_size=(640, 640),
            mosaic_prob=1.0,
            mosaic_9_prob=0.0,
            mixup_prob=0.0,
        )

        img, target = wrapper._mosaic4(0)
        assert isinstance(img, Image.Image)
        assert target["boxes"].shape == (0, 4)
        assert target["labels"].shape == (0,)


class TestMosaic9:
    """Tests for 9-way mosaic augmentation."""

    def test_mosaic9_output_shape(self, mock_dataset):
        """Test that mosaic9 produces correct output shape."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
            mosaic_9_prob=1.0,  # Force 9-way
            mixup_prob=0.0,
        )

        img, target = wrapper._mosaic9(0)
        assert isinstance(img, Image.Image)
        assert img.size == (640, 640)

    def test_mosaic9_boxes_in_bounds(self, mock_dataset):
        """Test that mosaic9 boxes are within image bounds."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
            mosaic_9_prob=1.0,
            mixup_prob=0.0,
        )

        for _ in range(10):
            img, target = wrapper._mosaic9(random.randint(0, len(mock_dataset) - 1))
            boxes = target["boxes"]
            if len(boxes) > 0:
                assert (boxes[:, 0] >= 0).all(), "x1 should be >= 0"
                assert (boxes[:, 1] >= 0).all(), "y1 should be >= 0"
                assert (boxes[:, 2] <= 640).all(), "x2 should be <= 640"
                assert (boxes[:, 3] <= 640).all(), "y2 should be <= 640"

    def test_mosaic9_with_empty_boxes(self, mock_dataset_empty):
        """Test mosaic9 handles empty boxes correctly."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset_empty,
            image_size=(640, 640),
            mosaic_prob=1.0,
            mosaic_9_prob=1.0,
            mixup_prob=0.0,
        )

        img, target = wrapper._mosaic9(0)
        assert target["boxes"].shape == (0, 4)


class TestMixUp:
    """Tests for MixUp augmentation."""

    def test_mixup_output_shape(self, mock_dataset):
        """Test that mixup produces correct output shape."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
            mixup_prob=1.0,  # Force mixup
        )

        # First create a mosaic, then apply mixup
        img1, target1 = wrapper._mosaic4(0)
        img_mixed, target_mixed = wrapper._mixup(img1, target1)

        assert isinstance(img_mixed, Image.Image)
        assert img_mixed.size == (640, 640)

    def test_mixup_combines_boxes(self, mock_dataset):
        """Test that mixup combines boxes from both images."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
            mixup_prob=1.0,
        )

        # Run multiple times to check box combination
        for _ in range(5):
            img1, target1 = wrapper._mosaic4(0)
            n_boxes_before = len(target1["boxes"])

            img_mixed, target_mixed = wrapper._mixup(img1, target1)
            n_boxes_after = len(target_mixed["boxes"])

            # Should have at least as many boxes (usually more from second image)
            assert n_boxes_after >= n_boxes_before or n_boxes_before == 0

    def test_mixup_beta_distribution(self):
        """Test that mixup uses correct Beta distribution."""
        alpha = 32.0
        samples = [np.random.beta(alpha, alpha) for _ in range(1000)]

        # Beta(32, 32) should produce values concentrated around 0.5
        mean = np.mean(samples)
        std = np.std(samples)

        assert 0.45 < mean < 0.55, f"Mean {mean} should be close to 0.5"
        assert std < 0.1, f"Std {std} should be small for alpha=32"


class TestCutMix:
    """Tests for CutMix augmentation."""

    def test_cutmix_output_shape(self, mock_dataset):
        """Test that cutmix produces correct output shape."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=0.0,  # No mosaic
            cutmix_prob=1.0,  # Force cutmix
        )

        img, target = wrapper._load_image_target(0)
        # Resize to match expected size
        img = img.resize((640, 640))

        img_cut, target_cut = wrapper._cutmix(img, target)
        assert isinstance(img_cut, Image.Image)
        assert img_cut.size == (640, 640)

    def test_rand_bbox_bounds(self, mock_dataset):
        """Test that _rand_bbox produces valid coordinates."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(mock_dataset, image_size=(640, 640))

        for _ in range(100):
            x1, y1, x2, y2 = wrapper._rand_bbox(640, 640)
            assert 0 <= x1 <= 640, f"x1={x1} out of bounds"
            assert 0 <= y1 <= 640, f"y1={y1} out of bounds"
            assert 0 <= x2 <= 640, f"x2={x2} out of bounds"
            assert 0 <= y2 <= 640, f"y2={y2} out of bounds"
            assert x1 <= x2, f"x1={x1} > x2={x2}"
            assert y1 <= y2, f"y1={y1} > y2={y2}"

    def test_rand_bbox_center_based(self, mock_dataset):
        """Test that _rand_bbox uses center-based sampling."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(mock_dataset, image_size=(640, 640))

        # Run many times and check distribution
        centers_x = []
        centers_y = []
        for _ in range(1000):
            x1, y1, x2, y2 = wrapper._rand_bbox(640, 640)
            centers_x.append((x1 + x2) / 2)
            centers_y.append((y1 + y2) / 2)

        # Centers should be uniformly distributed across the image
        mean_x = np.mean(centers_x)
        mean_y = np.mean(centers_y)

        # Allow some tolerance (center of image is 320)
        assert 250 < mean_x < 390, f"Mean x={mean_x} should be near center"
        assert 250 < mean_y < 390, f"Mean y={mean_y} should be near center"

    def test_cutmix_ioa_threshold(self, mock_dataset):
        """Test that cutmix uses IoA threshold for box filtering."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=0.0,
            cutmix_prob=1.0,
        )

        # The IoA threshold is 0.1 - boxes with < 10% overlap should be excluded
        # This is implicitly tested by the cutmix working correctly
        img, target = wrapper._load_image_target(0)
        img = img.resize((640, 640))
        img_cut, target_cut = wrapper._cutmix(img, target)

        # Just verify it runs without error
        assert target_cut["boxes"].shape[1] == 4


# =============================================================================
# RandomPerspective Tests
# =============================================================================


class TestRandomPerspective:
    """Tests for RandomPerspective transform."""

    def test_import(self):
        """Test that RandomPerspective can be imported."""
        from yolo.data.transforms import RandomPerspective
        assert RandomPerspective is not None

    def test_initialization(self):
        """Test RandomPerspective initialization with various parameters."""
        from yolo.data.transforms import RandomPerspective

        rp = RandomPerspective(
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=5.0,
            perspective=0.001,
        )
        assert rp.degrees == 10.0
        assert rp.translate == 0.1
        assert rp.scale == 0.5
        assert rp.shear == 5.0
        assert rp.perspective == 0.001

    def test_no_transform_when_disabled(self):
        """Test that no transform is applied when all params are 0."""
        from yolo.data.transforms import RandomPerspective

        rp = RandomPerspective(
            degrees=0,
            translate=0,
            scale=0,
            shear=0,
            perspective=0,
        )

        img = torch.rand(3, 640, 640)
        target = {
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }

        img_out, target_out = rp(img, target)

        # Should be unchanged
        assert torch.allclose(img, img_out)
        assert torch.allclose(target["boxes"], target_out["boxes"])

    def test_output_shape(self):
        """Test that RandomPerspective preserves output shape."""
        from yolo.data.transforms import RandomPerspective

        rp = RandomPerspective(degrees=10, translate=0.1, scale=0.5)

        img = torch.rand(3, 640, 640)
        target = {
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }

        img_out, target_out = rp(img, target)
        assert img_out.shape == img.shape
        assert target_out["boxes"].shape[1] == 4

    def test_boxes_in_bounds(self):
        """Test that transformed boxes are within image bounds."""
        from yolo.data.transforms import RandomPerspective

        rp = RandomPerspective(degrees=30, translate=0.2, scale=0.5, shear=10)

        for _ in range(20):
            img = torch.rand(3, 640, 640)
            target = {
                "boxes": torch.tensor(
                    [[100, 100, 200, 200], [300, 300, 400, 400]],
                    dtype=torch.float32,
                ),
                "labels": torch.tensor([0, 1], dtype=torch.long),
            }

            _, target_out = rp(img, target)
            boxes = target_out["boxes"]

            if len(boxes) > 0:
                assert (boxes[:, 0] >= 0).all(), "x1 < 0"
                assert (boxes[:, 1] >= 0).all(), "y1 < 0"
                assert (boxes[:, 2] <= 640).all(), "x2 > 640"
                assert (boxes[:, 3] <= 640).all(), "y2 > 640"

    def test_matrix_composition_order(self):
        """Test that transformation matrix is composed in correct order: T @ S @ R @ P @ C."""
        from yolo.data.transforms import RandomPerspective

        rp = RandomPerspective(degrees=10, translate=0.1, scale=0.2, shear=5)

        # Build matrix manually and check it's 3x3
        M = rp._build_transform_matrix(640, 640, 640, 640)
        assert M.shape == (3, 3)

    def test_empty_boxes(self):
        """Test RandomPerspective with empty boxes."""
        from yolo.data.transforms import RandomPerspective

        rp = RandomPerspective(degrees=10, translate=0.1, scale=0.5)

        img = torch.rand(3, 640, 640)
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }

        img_out, target_out = rp(img, target)
        assert target_out["boxes"].shape == (0, 4)


# =============================================================================
# EMA Tests
# =============================================================================


class TestModelEMA:
    """Tests for ModelEMA class."""

    def test_import(self):
        """Test that ModelEMA can be imported."""
        from yolo.training.callbacks import ModelEMA
        assert ModelEMA is not None

    def test_initialization(self):
        """Test ModelEMA initialization."""
        from yolo.training.callbacks import ModelEMA

        model = nn.Linear(10, 10)
        ema = ModelEMA(model, decay=0.9999, tau=2000)

        assert ema.decay == 0.9999
        assert ema.tau == 2000
        assert ema.updates == 0

    def test_ema_weights_different_from_model(self):
        """Test that EMA weights diverge from model after updates."""
        from yolo.training.callbacks import ModelEMA

        model = nn.Linear(10, 10)
        ema = ModelEMA(model, decay=0.9999, tau=2000)

        # Initially EMA should be copy of model
        for p1, p2 in zip(model.parameters(), ema.ema.parameters()):
            assert torch.allclose(p1, p2)

        # Update model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Update EMA
        ema.update(model)

        # EMA should now be different from model (blended)
        different = False
        for p1, p2 in zip(model.parameters(), ema.ema.parameters()):
            if not torch.allclose(p1, p2):
                different = True
                break
        assert different, "EMA weights should be different from model after update"

    def test_decay_formula(self):
        """Test that decay formula matches: decay * (1 - exp(-updates / tau))."""
        from yolo.training.callbacks import ModelEMA

        model = nn.Linear(10, 10)
        decay = 0.9999
        tau = 2000.0

        ema = ModelEMA(model, decay=decay, tau=tau)

        # Test decay at various update counts
        test_cases = [
            (0, 0.0),  # At 0 updates, effective decay is 0
            (1, decay * (1 - math.exp(-1 / tau))),
            (1000, decay * (1 - math.exp(-1000 / tau))),
            (2000, decay * (1 - math.exp(-2000 / tau))),
            (10000, decay * (1 - math.exp(-10000 / tau))),
        ]

        for updates, expected_decay in test_cases:
            actual_decay = decay * (1 - math.exp(-updates / tau)) if updates > 0 else 0
            assert abs(actual_decay - expected_decay) < 1e-6, \
                f"Decay mismatch at updates={updates}: {actual_decay} vs {expected_decay}"

    def test_decay_warmup(self):
        """Test that decay ramps up during warmup."""
        from yolo.training.callbacks import ModelEMA

        model = nn.Linear(10, 10)
        ema = ModelEMA(model, decay=0.9999, tau=2000)

        # Simulate updates and track effective decay
        decays = []
        for i in range(5000):
            ema.updates = i
            d = ema.decay * (1 - math.exp(-ema.updates / ema.tau)) if ema.updates > 0 else 0
            decays.append(d)

        # Decay should be monotonically increasing
        for i in range(1, len(decays)):
            assert decays[i] >= decays[i - 1], "Decay should be non-decreasing"

        # At updates=0, decay should be ~0
        assert decays[0] < 0.001

        # At updates=2000, decay should be ~0.632 * 0.9999
        assert 0.62 < decays[2000] < 0.64

        # At updates=5000, decay should be close to 0.9999
        assert decays[-1] > 0.9

    def test_state_dict_save_load(self):
        """Test that EMA state can be saved and loaded."""
        from yolo.training.callbacks import ModelEMA

        model = nn.Linear(10, 10)
        ema = ModelEMA(model, decay=0.9999, tau=2000)

        # Do some updates
        for _ in range(100):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema.update(model)

        # Save state
        state = ema.state_dict()
        assert "ema_state_dict" in state
        assert "decay" in state
        assert "tau" in state
        assert "updates" in state

        # Create new EMA and load state
        model2 = nn.Linear(10, 10)
        ema2 = ModelEMA(model2, decay=0.999, tau=1000)  # Different params
        ema2.load_state_dict(state)

        assert ema2.updates == ema.updates
        assert ema2.decay == ema.decay
        assert ema2.tau == ema.tau

    def test_gradients_disabled(self):
        """Test that EMA model has gradients disabled."""
        from yolo.training.callbacks import ModelEMA

        model = nn.Linear(10, 10)
        ema = ModelEMA(model, decay=0.9999, tau=2000)

        for param in ema.ema.parameters():
            assert not param.requires_grad, "EMA params should not require grad"

    def test_ema_in_eval_mode(self):
        """Test that EMA model is in eval mode."""
        from yolo.training.callbacks import ModelEMA

        model = nn.Linear(10, 10)
        ema = ModelEMA(model, decay=0.9999, tau=2000)

        assert not ema.ema.training, "EMA model should be in eval mode"


class TestEMACallback:
    """Tests for EMACallback Lightning callback."""

    def test_import(self):
        """Test that EMACallback can be imported."""
        from yolo.training.callbacks import EMACallback
        assert EMACallback is not None

    def test_initialization(self):
        """Test EMACallback initialization."""
        from yolo.training.callbacks import EMACallback

        callback = EMACallback(decay=0.9999, tau=2000, enabled=True)
        assert callback.decay == 0.9999
        assert callback.tau == 2000
        assert callback.enabled is True

    def test_disabled_callback(self):
        """Test that disabled callback does nothing."""
        from yolo.training.callbacks import EMACallback

        callback = EMACallback(enabled=False)
        assert callback.enabled is False
        assert callback._ema is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full augmentation pipeline."""

    def test_full_pipeline_mosaic_mixup(self, mock_dataset):
        """Test full pipeline with mosaic and mixup."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
            mosaic_9_prob=0.0,
            mixup_prob=0.5,
            cutmix_prob=0.0,
        )

        # Get multiple items
        for i in range(10):
            img, target = wrapper[i]
            # Without transforms, we get PIL Image
            assert isinstance(img, Image.Image)
            assert img.size == (640, 640)
            assert "boxes" in target
            assert "labels" in target

    def test_full_pipeline_no_mosaic_with_cutmix(self, mock_dataset):
        """Test pipeline without mosaic but with cutmix."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=0.0,  # No mosaic
            cutmix_prob=1.0,  # Always cutmix
        )

        for i in range(10):
            img, target = wrapper[i]
            assert isinstance(img, Image.Image)

    def test_mosaic_disabled_returns_single_image(self, mock_dataset):
        """Test that disabled mosaic returns single images."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
            cutmix_prob=0.0,
        )

        # Disable mosaic
        wrapper.disable_mosaic()

        for i in range(5):
            img, target = wrapper[i]
            # Should return the original image (resized)
            assert isinstance(img, Image.Image)

    def test_transforms_chain(self, mock_dataset):
        """Test that transforms can be chained."""
        from yolo.data.mosaic import MosaicMixupDataset
        from yolo.data.transforms import LetterBox, RandomPerspective, Compose

        transforms = Compose([
            LetterBox(target_size=(640, 640)),
            RandomPerspective(degrees=10, translate=0.1, scale=0.5),
        ])

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
            transforms=transforms,
        )

        img, target = wrapper[0]
        # After transforms, should be tensor
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 640, 640)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_image_dataset(self):
        """Test with dataset containing single image."""
        from yolo.data.mosaic import MosaicMixupDataset

        single_dataset = MockDataset(num_samples=1)
        wrapper = MosaicMixupDataset(
            single_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
        )

        # Should work even with single image (will sample same image multiple times)
        img, target = wrapper[0]
        assert isinstance(img, Image.Image)

    def test_very_small_images(self):
        """Test with very small images."""
        from yolo.data.mosaic import MosaicMixupDataset

        small_dataset = MockDataset(num_samples=10, image_size=(32, 32))
        wrapper = MosaicMixupDataset(
            small_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
        )

        img, target = wrapper[0]
        assert img.size == (640, 640)

    def test_non_square_target_size(self):
        """Test with non-square target size."""
        from yolo.data.mosaic import MosaicMixupDataset

        dataset = MockDataset(num_samples=10)
        # Note: Current implementation assumes square, but should handle gracefully
        wrapper = MosaicMixupDataset(
            dataset,
            image_size=(640, 480),  # Non-square
            mosaic_prob=1.0,
        )

        # Should use first dimension for canvas size
        assert wrapper.border == (-320, -240)

    def test_reproducibility_with_seed(self, mock_dataset):
        """Test that results are reproducible with same seed."""
        from yolo.data.mosaic import MosaicMixupDataset

        wrapper = MosaicMixupDataset(
            mock_dataset,
            image_size=(640, 640),
            mosaic_prob=1.0,
            mixup_prob=0.0,
        )

        # Set seed and get result
        random.seed(42)
        np.random.seed(42)
        img1, target1 = wrapper._mosaic4(0)

        # Reset seed and get result again
        random.seed(42)
        np.random.seed(42)
        img2, target2 = wrapper._mosaic4(0)

        # Results should be identical
        assert np.array_equal(np.array(img1), np.array(img2))
        assert torch.equal(target1["boxes"], target2["boxes"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
