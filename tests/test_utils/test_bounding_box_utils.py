"""
Tests for bounding box utilities.

These tests verify IoU calculations, bbox transformations,
NMS, anchors, and mAP calculations.
"""

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import allclose, float32, isclose, tensor

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo import NMSConfig, create_model
from yolo.config.config import AnchorConfig
from yolo.utils.bounding_box_utils import (
    Anc2Box,
    Vec2Box,
    bbox_nms,
    calculate_iou,
    calculate_map,
    generate_anchors,
    transform_bbox,
)

EPS = 1e-4


def load_model_config(model_name: str):
    """Load model config from YAML file as OmegaConf."""
    config_path = project_root / "yolo" / "config" / "model" / f"{model_name}.yaml"
    if not config_path.exists():
        pytest.skip(f"Model config {model_name}.yaml not found")

    return OmegaConf.load(config_path)


@pytest.fixture
def dummy_bboxes():
    bbox1 = tensor([[50, 80, 150, 140], [30, 20, 100, 80]], dtype=float32)
    bbox2 = tensor([[90, 70, 160, 160], [40, 40, 90, 120]], dtype=float32)
    return bbox1, bbox2


def test_calculate_iou_2d(dummy_bboxes):
    bbox1, bbox2 = dummy_bboxes
    iou = calculate_iou(bbox1, bbox2)
    expected_iou = tensor([[0.4138, 0.1905], [0.0096, 0.3226]])
    assert iou.shape == (2, 2)
    assert allclose(iou, expected_iou, atol=EPS)


def test_calculate_iou_3d(dummy_bboxes):
    bbox1, bbox2 = dummy_bboxes
    iou = calculate_iou(bbox1[None], bbox2[None])
    expected_iou = tensor([[0.4138, 0.1905], [0.0096, 0.3226]])
    assert iou.shape == (1, 2, 2)
    assert allclose(iou, expected_iou, atol=EPS)


def test_calculate_diou(dummy_bboxes):
    bbox1, bbox2 = dummy_bboxes
    iou = calculate_iou(bbox1, bbox2, "diou")
    expected_diou = tensor([[0.3816, 0.0943], [-0.2048, 0.2622]])

    assert iou.shape == (2, 2)
    assert allclose(iou, expected_diou, atol=EPS)


def test_calculate_ciou(dummy_bboxes):
    bbox1, bbox2 = dummy_bboxes
    iou = calculate_iou(bbox1, bbox2, metrics="ciou")
    # TODO: check result!
    expected_ciou = tensor([[0.3769, 0.0853], [-0.2050, 0.2602]])
    assert iou.shape == (2, 2)
    assert allclose(iou, expected_ciou, atol=EPS)

    bbox1 = tensor([[50, 80, 150, 140], [30, 20, 100, 80]], dtype=float32)
    bbox2 = tensor([[90, 70, 160, 160], [40, 40, 90, 120]], dtype=float32)


def test_transform_bbox_xywh_to_Any(dummy_bboxes):
    bbox1, _ = dummy_bboxes
    transformed_bbox = transform_bbox(bbox1, "xywh -> xyxy")
    expected_bbox = tensor([[50.0, 80.0, 200.0, 220.0], [30.0, 20.0, 130.0, 100.0]])
    assert allclose(transformed_bbox, expected_bbox)


def test_transform_bbox_xycwh_to_Any(dummy_bboxes):
    bbox1, bbox2 = dummy_bboxes
    transformed_bbox = transform_bbox(bbox1, "xycwh -> xycwh")
    assert allclose(transformed_bbox, bbox1)

    transformed_bbox = transform_bbox(bbox2, "xyxy -> xywh")
    expected_bbox = tensor([[90.0, 70.0, 70.0, 90.0], [40.0, 40.0, 50.0, 80.0]])
    assert allclose(transformed_bbox, expected_bbox)


def test_transform_bbox_xyxy_to_Any(dummy_bboxes):
    bbox1, bbox2 = dummy_bboxes
    transformed_bbox = transform_bbox(bbox1, "xyxy -> xyxy")
    assert allclose(transformed_bbox, bbox1)

    transformed_bbox = transform_bbox(bbox2, "xyxy -> xycwh")
    expected_bbox = tensor([[125.0, 115.0, 70.0, 90.0], [65.0, 80.0, 50.0, 80.0]])
    assert allclose(transformed_bbox, expected_bbox)


def test_transform_bbox_invalid_format(dummy_bboxes):
    bbox, _ = dummy_bboxes

    # Test invalid input format
    with pytest.raises(ValueError, match="Invalid input or output format"):
        transform_bbox(bbox, "invalid->xyxy")

    # Test invalid output format
    with pytest.raises(ValueError, match="Invalid input or output format"):
        transform_bbox(bbox, "xywh->invalid")


def test_generate_anchors():
    image_size = [256, 256]
    strides = [8, 16, 32]
    anchors, scalers = generate_anchors(image_size, strides)
    assert anchors.shape[0] == scalers.shape[0]
    assert anchors.shape[1] == 2


def test_vec2box_autoanchor():
    """Test Vec2Box anchor generation."""
    cfg = load_model_config("v9-m")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg, weight_path=None)

    # Build anchor config from model config
    anchor_cfg = cfg.get("anchor", {})
    image_size = [640, 640]

    vec2box = Vec2Box(model, anchor_cfg, image_size, device)
    assert vec2box.strides == [8, 16, 32]

    vec2box.update((320, 640))
    assert vec2box.anchor_grid.shape == (4200, 2)
    assert vec2box.scaler.shape == tuple([4200])


@pytest.mark.skipif(
    not (project_root / "yolo" / "config" / "model" / "v7.yaml").exists(),
    reason="v7.yaml not found"
)
def test_anc2box_autoanchor():
    """Test Anc2Box anchor generation for v7 model."""
    cfg = load_model_config("v7")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg, weight_path=None).to(device)

    # Anc2Box expects an OmegaConf object with attributes
    anchor_cfg = cfg.get("anchor", {})
    # Remove strides for auto-detection
    if "strides" in anchor_cfg:
        anchor_cfg = OmegaConf.create({k: v for k, v in OmegaConf.to_container(anchor_cfg).items() if k != "strides"})

    image_size = [640, 640]
    anc2box = Anc2Box(model, anchor_cfg, image_size, device)
    assert anc2box.strides == [8, 16, 32]

    anc2box.update((320, 640))
    anchor_grids_shape = [anchor_grid.shape for anchor_grid in anc2box.anchor_grids]
    assert anchor_grids_shape == [
        torch.Size([1, 1, 80, 40, 2]),
        torch.Size([1, 1, 40, 20, 2]),
        torch.Size([1, 1, 20, 10, 2]),
    ]
    assert anc2box.anchor_scale.shape == torch.Size([3, 1, 3, 1, 1, 2])


def test_bbox_nms():
    cls_dist = torch.tensor(
        [
            [
                [0.7, 0.1, 0.2],  # High confidence, class 0
                [0.3, 0.6, 0.1],  # High confidence, class 1
                [-3.0, -2.0, -1.0],  # low confidence, class 2
                [0.6, 0.2, 0.2],  # Medium confidence, class 0
            ],
            [
                [0.55, 0.25, 0.2],  # Medium confidence, class 0
                [-4.0, -0.5, -2.0],  # low confidence, class 1
                [0.15, 0.2, 0.65],  # Medium confidence, class 2
                [0.8, 0.1, 0.1],  # High confidence, class 0
            ],
        ],
        dtype=float32,
    )

    bbox = torch.tensor(
        [
            [
                [0, 0, 160, 120],  # Overlaps with box 4
                [160, 120, 320, 240],
                [0, 120, 160, 240],
                [16, 12, 176, 132],
            ],
            [
                [0, 0, 160, 120],  # Overlaps with box 4
                [160, 120, 320, 240],
                [0, 120, 160, 240],
                [16, 12, 176, 132],
            ],
        ],
        dtype=float32,
    )

    nms_cfg = NMSConfig(min_confidence=0.5, min_iou=0.5, max_bbox=400)

    # Batch 1:
    #  - box 1 is kept with classes 0 and 2 as it overlaps with box 4 and has a higher confidence for classes 0 and 2.
    #  - box 2 is kept with classes 0, 1, 2 as it does not overlap with any other box.
    #  - box 3 is rejected by the confidence filter.
    #  - box 4 is kept with class 1 as it overlaps with box 1 and has a higher confidence for class 1.
    # Batch 2:
    #  - box 1 is kept with classes 1 and 2 as it overlaps with box 1 and has a higher confidence for classes 1 and 2.
    #  - box 2 is rejected by the confidence filter.
    #  - box 3 is kept with classes 0, 1, 2 as it does not overlap with any other box.
    #  - box 4 is kept with class 0 as it overlaps with box 1 and has a higher confidence for class 0.
    expected_output = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 160.0, 120.0, 0.6682],
                [1.0, 160.0, 120.0, 320.0, 240.0, 0.6457],
                [0.0, 160.0, 120.0, 320.0, 240.0, 0.5744],
                [2.0, 0.0, 0.0, 160.0, 120.0, 0.5498],
                [1.0, 16.0, 12.0, 176.0, 132.0, 0.5498],
                [2.0, 160.0, 120.0, 320.0, 240.0, 0.5250],
            ],
            [
                [0.0, 16.0, 12.0, 176.0, 132.0, 0.6900],
                [2.0, 0.0, 120.0, 160.0, 240.0, 0.6570],
                [1.0, 0.0, 0.0, 160.0, 120.0, 0.5622],
                [2.0, 0.0, 0.0, 160.0, 120.0, 0.5498],
                [1.0, 0.0, 120.0, 160.0, 240.0, 0.5498],
                [0.0, 0.0, 120.0, 160.0, 240.0, 0.5374],
            ],
        ]
    )

    output = bbox_nms(cls_dist, bbox, nms_cfg)
    for out, exp in zip(output, expected_output):
        assert allclose(out, exp, atol=1e-4), f"Output: {out} Expected: {exp}"


def test_calculate_map():
    predictions = tensor([[0, 60, 60, 160, 160, 0.5], [0, 40, 40, 120, 120, 0.5]])  # [class, x1, y1, x2, y2]
    ground_truths = tensor([[0, 50, 50, 150, 150], [0, 30, 30, 100, 100]])  # [class, x1, y1, x2, y2]

    mAP = calculate_map(predictions, ground_truths)
    expected_ap50 = tensor(0.5050)
    expected_ap50_95 = tensor(0.2020)

    assert isclose(mAP["map_50"], expected_ap50, atol=1e-4), f"AP50 mismatch"
    assert isclose(mAP["map"], expected_ap50_95, atol=1e-4), f"Mean AP mismatch"


# =============================================================================
# BoxMatcher Tests - Full Image Boxes Support
# =============================================================================

class TestBoxMatcherLargeBoxes:
    """
    Tests for BoxMatcher.get_valid_matrix() with large boxes.

    These tests verify that boxes covering the entire image (or most of it)
    are correctly matched with anchors. Previously, this was broken because
    the reg_max constraint would reject large boxes.
    """

    @pytest.fixture
    def create_vec2box_mock(self):
        """Create a mock Vec2Box with configurable anchor grid."""
        class Vec2BoxMock:
            def __init__(self, image_size, strides):
                # Generate anchor grid similar to real Vec2Box
                anchors_list = []
                scalers_list = []
                for stride in strides:
                    h, w = image_size[0] // stride, image_size[1] // stride
                    grid_y, grid_x = torch.meshgrid(
                        torch.arange(h) * stride + stride / 2,
                        torch.arange(w) * stride + stride / 2,
                        indexing='ij'
                    )
                    anchors = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
                    anchors_list.append(anchors)
                    scalers_list.append(torch.full((anchors.shape[0],), stride, dtype=torch.float32))

                self.anchor_grid = torch.cat(anchors_list, dim=0)
                self.scaler = torch.cat(scalers_list, dim=0)

        return Vec2BoxMock

    @pytest.fixture
    def box_matcher(self, create_vec2box_mock):
        """Create a BoxMatcher with standard configuration."""
        from yolo.utils.bounding_box_utils import BoxMatcher
        from omegaconf import OmegaConf

        vec2box = create_vec2box_mock(image_size=(320, 320), strides=[8, 16, 32])
        cfg = OmegaConf.create({
            "topk": 10,
            "alpha": 0.5,
            "beta": 6.0,
        })
        return BoxMatcher(cfg, class_num=13, vec2box=vec2box, reg_max=16)

    def test_full_image_box_matches_anchors(self, box_matcher):
        """
        Test that a box covering the entire image matches with anchors.

        This is the critical test case that was failing before the fix.
        A box [0, 0, 320, 320] should match with all anchors whose centers
        are inside the box (which is all of them for a full-image box).
        """
        # Full image box: [0, 0, 320, 320] in xyxy format
        # Shape: [batch=1, targets=1, coords=4]
        target_bbox = torch.tensor([[[0.0, 0.0, 320.0, 320.0]]])

        valid_matrix = box_matcher.get_valid_matrix(target_bbox)

        # Shape should be [batch=1, targets=1, num_anchors]
        assert valid_matrix.shape[0] == 1
        assert valid_matrix.shape[1] == 1

        # ALL anchors should be valid for a full-image box
        num_valid = valid_matrix.sum().item()
        total_anchors = valid_matrix.shape[2]

        assert num_valid == total_anchors, (
            f"Full-image box should match ALL anchors. "
            f"Got {num_valid}/{total_anchors} valid anchors."
        )

    def test_large_box_matches_most_anchors(self, box_matcher):
        """
        Test that a box covering 90% of the image matches most anchors.
        """
        # Box covering 90% of image: [16, 16, 304, 304]
        target_bbox = torch.tensor([[[16.0, 16.0, 304.0, 304.0]]])

        valid_matrix = box_matcher.get_valid_matrix(target_bbox)

        num_valid = valid_matrix.sum().item()
        total_anchors = valid_matrix.shape[2]

        # At least 80% of anchors should be valid
        min_expected = total_anchors * 0.8
        assert num_valid >= min_expected, (
            f"Large box should match most anchors. "
            f"Got {num_valid}/{total_anchors} valid anchors (expected >= {min_expected})."
        )

    def test_small_box_matches_few_anchors(self, box_matcher):
        """
        Test that a small box matches only nearby anchors.
        """
        # Small box: 32x32 pixels in the center
        cx, cy = 160, 160
        target_bbox = torch.tensor([[[cx - 16.0, cy - 16.0, cx + 16.0, cy + 16.0]]])

        valid_matrix = box_matcher.get_valid_matrix(target_bbox)

        num_valid = valid_matrix.sum().item()
        total_anchors = valid_matrix.shape[2]

        # Small box should match relatively few anchors
        max_expected = total_anchors * 0.1
        assert num_valid <= max_expected, (
            f"Small box should match few anchors. "
            f"Got {num_valid}/{total_anchors} valid anchors (expected <= {max_expected})."
        )
        # But should match at least some
        assert num_valid > 0, "Small box should match at least some anchors."

    def test_box_outside_image_matches_nothing(self, box_matcher):
        """
        Test that a box completely outside the image matches no anchors.
        """
        # Box outside image: [400, 400, 500, 500]
        target_bbox = torch.tensor([[[400.0, 400.0, 500.0, 500.0]]])

        valid_matrix = box_matcher.get_valid_matrix(target_bbox)

        num_valid = valid_matrix.sum().item()
        assert num_valid == 0, (
            f"Box outside image should match NO anchors. "
            f"Got {num_valid} valid anchors."
        )

    def test_multiple_boxes_mixed_sizes(self, box_matcher):
        """
        Test batch with multiple boxes of different sizes.
        """
        # Batch of 2 images, each with 3 targets
        target_bbox = torch.tensor([
            [
                [0.0, 0.0, 320.0, 320.0],   # Full image
                [100.0, 100.0, 200.0, 200.0],  # Medium box
                [150.0, 150.0, 170.0, 170.0],  # Small box
            ],
            [
                [10.0, 10.0, 310.0, 310.0],   # Nearly full image
                [0.0, 0.0, 160.0, 160.0],     # Quarter image
                [500.0, 500.0, 600.0, 600.0],  # Outside image
            ],
        ])

        valid_matrix = box_matcher.get_valid_matrix(target_bbox)

        # Shape should be [batch=2, targets=3, num_anchors]
        assert valid_matrix.shape[0] == 2
        assert valid_matrix.shape[1] == 3

        # Full image boxes should match all anchors
        total_anchors = valid_matrix.shape[2]
        assert valid_matrix[0, 0].sum().item() == total_anchors

        # Outside box should match nothing
        assert valid_matrix[1, 2].sum().item() == 0

    def test_anchor_center_inside_logic(self, box_matcher):
        """
        Verify the anchor-center-inside-box logic works correctly.
        """
        # Box from (100, 100) to (200, 200)
        target_bbox = torch.tensor([[[100.0, 100.0, 200.0, 200.0]]])

        valid_matrix = box_matcher.get_valid_matrix(target_bbox)

        # Check specific anchors
        anchor_grid = box_matcher.vec2box.anchor_grid

        for i, (ax, ay) in enumerate(anchor_grid):
            is_inside = (100 <= ax <= 200) and (100 <= ay <= 200)
            assert valid_matrix[0, 0, i].item() == is_inside, (
                f"Anchor {i} at ({ax}, {ay}) should be "
                f"{'inside' if is_inside else 'outside'} box [100, 100, 200, 200]"
            )
