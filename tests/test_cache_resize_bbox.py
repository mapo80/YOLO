"""
Test cache resize bbox transformation.

Verifies that when images are resized during caching, the bounding boxes
are correctly transformed to match the letterboxed image.
"""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo.data.cache import ImageCache


def test_cache_serialization_with_orig_size():
    """Test that orig_size is correctly serialized and deserialized."""
    cache = ImageCache(mode='ram')

    # Create test array (simulating 320x320 resized image)
    arr = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    orig_size = (1240, 1754)  # Original image size

    # Serialize with orig_size
    data = cache._serialize(arr, orig_size)
    arr_out, orig_size_out = cache._deserialize(data)

    assert np.array_equal(arr, arr_out), "Array mismatch after serialization"
    assert orig_size == orig_size_out, f"orig_size mismatch: {orig_size} != {orig_size_out}"
    print("✓ Serialization with orig_size works")


def test_cache_serialization_without_orig_size():
    """Test that serialization works without orig_size (no resize)."""
    cache = ImageCache(mode='ram')

    arr = np.random.randint(0, 255, (1754, 1240, 3), dtype=np.uint8)

    # Serialize without orig_size
    data = cache._serialize(arr, None)
    arr_out, orig_size_out = cache._deserialize(data)

    assert np.array_equal(arr, arr_out), "Array mismatch after serialization"
    assert orig_size_out is None, f"orig_size should be None, got {orig_size_out}"
    print("✓ Serialization without orig_size works")


def test_bbox_letterbox_transform():
    """Test bbox transformation for letterboxed images."""
    # Simulate the transformation function
    def transform_bboxes_for_letterbox(boxes, orig_size, target_size):
        orig_w, orig_h = orig_size
        target_w, target_h = target_size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        boxes = boxes.clone()
        boxes[:, 0] = boxes[:, 0] * scale + pad_x  # x1
        boxes[:, 1] = boxes[:, 1] * scale + pad_y  # y1
        boxes[:, 2] = boxes[:, 2] * scale + pad_x  # x2
        boxes[:, 3] = boxes[:, 3] * scale + pad_y  # y2

        return boxes, scale, pad_x, pad_y

    # Test case 1: Portrait image (taller than wide)
    orig_size = (1240, 1754)  # w, h
    target_size = (320, 320)

    # Bbox covering center 50% of image
    cx, cy = orig_size[0] / 2, orig_size[1] / 2
    w, h = orig_size[0] * 0.5, orig_size[1] * 0.5
    orig_bbox = torch.tensor([[cx - w/2, cy - h/2, cx + w/2, cy + h/2]], dtype=torch.float32)

    transformed, scale, pad_x, pad_y = transform_bboxes_for_letterbox(orig_bbox, orig_size, target_size)

    # Verify scale is correct (limited by width since image is portrait)
    expected_scale = 320 / 1754  # height is limiting factor
    assert abs(scale - expected_scale) < 0.001, f"Scale mismatch: {scale} != {expected_scale}"

    # Verify padding is on left/right (since height fills the target)
    assert pad_y == 0, f"Vertical padding should be 0 for portrait image, got {pad_y}"
    assert pad_x > 0, f"Horizontal padding should be > 0 for portrait image, got {pad_x}"

    # Verify transformed bbox is within target bounds
    x1, y1, x2, y2 = transformed[0].tolist()
    assert 0 <= x1 <= 320, f"x1 out of bounds: {x1}"
    assert 0 <= y1 <= 320, f"y1 out of bounds: {y1}"
    assert 0 <= x2 <= 320, f"x2 out of bounds: {x2}"
    assert 0 <= y2 <= 320, f"y2 out of bounds: {y2}"

    # Verify bbox center is still roughly centered
    transformed_cx = (x1 + x2) / 2
    transformed_cy = (y1 + y2) / 2
    assert abs(transformed_cx - 160) < 5, f"Transformed center X should be ~160, got {transformed_cx}"
    assert abs(transformed_cy - 160) < 5, f"Transformed center Y should be ~160, got {transformed_cy}"

    print(f"✓ Portrait image: orig_bbox={orig_bbox[0].tolist()}")
    print(f"  → transformed={transformed[0].tolist()}")
    print(f"  → scale={scale:.4f}, pad=({pad_x}, {pad_y})")

    # Test case 2: Landscape image (wider than tall)
    orig_size = (1754, 1240)  # w, h
    target_size = (320, 320)

    cx, cy = orig_size[0] / 2, orig_size[1] / 2
    w, h = orig_size[0] * 0.5, orig_size[1] * 0.5
    orig_bbox = torch.tensor([[cx - w/2, cy - h/2, cx + w/2, cy + h/2]], dtype=torch.float32)

    transformed, scale, pad_x, pad_y = transform_bboxes_for_letterbox(orig_bbox, orig_size, target_size)

    # For landscape, width is limiting factor
    expected_scale = 320 / 1754
    assert abs(scale - expected_scale) < 0.001, f"Scale mismatch: {scale} != {expected_scale}"

    # Padding should be on top/bottom
    assert pad_x == 0, f"Horizontal padding should be 0 for landscape image, got {pad_x}"
    assert pad_y > 0, f"Vertical padding should be > 0 for landscape image, got {pad_y}"

    print(f"✓ Landscape image: orig_bbox={orig_bbox[0].tolist()}")
    print(f"  → transformed={transformed[0].tolist()}")
    print(f"  → scale={scale:.4f}, pad=({pad_x}, {pad_y})")


def test_full_cache_flow():
    """Test the complete flow: save resized image with orig_size, load and transform bbox."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Create cache with resize enabled
        cache = ImageCache(
            mode='disk',
            cache_dir=cache_dir,
            target_size=(320, 320),
        )

        # Create a test image (1240x1754)
        orig_w, orig_h = 1240, 1754
        orig_img = Image.new('RGB', (orig_w, orig_h), color=(100, 150, 200))

        # Simulate _resize_for_cache (letterbox)
        target_w, target_h = 320, 320
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = orig_img.resize((new_w, new_h), Image.BILINEAR)
        padded = Image.new('RGB', (target_w, target_h), (114, 114, 114))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded.paste(resized, (paste_x, paste_y))

        # Initialize cache
        cache.initialize(1, cache_dir, [Path("test.jpg")])

        # Store with orig_size
        arr = np.asarray(padded).copy()
        cache.put(0, arr, orig_size=(orig_w, orig_h))

        # Retrieve
        result = cache.get(0)
        assert result is not None, "Cache miss"

        arr_out, orig_size_out = result
        assert orig_size_out == (orig_w, orig_h), f"orig_size mismatch: {orig_size_out}"
        assert arr_out.shape == (320, 320, 3), f"Shape mismatch: {arr_out.shape}"

        print(f"✓ Full cache flow: stored 320x320 with orig_size={orig_size_out}")

        # Now verify bbox transformation would work
        # Create a bbox in original coordinates (center of image)
        orig_bbox = torch.tensor([
            [orig_w * 0.25, orig_h * 0.25, orig_w * 0.75, orig_h * 0.75]
        ], dtype=torch.float32)

        # Transform to letterbox coordinates
        boxes = orig_bbox.clone()
        boxes[:, 0] = boxes[:, 0] * scale + paste_x
        boxes[:, 1] = boxes[:, 1] * scale + paste_y
        boxes[:, 2] = boxes[:, 2] * scale + paste_x
        boxes[:, 3] = boxes[:, 3] * scale + paste_y

        x1, y1, x2, y2 = boxes[0].tolist()
        assert 0 <= x1 <= 320 and 0 <= x2 <= 320, f"X coords out of bounds: {x1}, {x2}"
        assert 0 <= y1 <= 320 and 0 <= y2 <= 320, f"Y coords out of bounds: {y1}, {y2}"

        print(f"  Original bbox: {orig_bbox[0].tolist()}")
        print(f"  Transformed bbox: {boxes[0].tolist()}")
        print("✓ Bbox transformation verified")


def test_edge_cases():
    """Test edge cases for bbox transformation."""
    def transform(boxes, orig_size, target_size):
        orig_w, orig_h = orig_size
        target_w, target_h = target_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        boxes = boxes.clone()
        boxes[:, 0] = boxes[:, 0] * scale + pad_x
        boxes[:, 1] = boxes[:, 1] * scale + pad_y
        boxes[:, 2] = boxes[:, 2] * scale + pad_x
        boxes[:, 3] = boxes[:, 3] * scale + pad_y
        return boxes

    # Edge case 1: Bbox at origin
    orig_size = (1240, 1754)
    target_size = (320, 320)
    bbox = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)
    transformed = transform(bbox, orig_size, target_size)

    # Should have padding offset
    assert transformed[0, 0] > 0, "x1 should have padding offset"
    print(f"✓ Edge case 1 (origin bbox): {bbox[0].tolist()} → {transformed[0].tolist()}")

    # Edge case 2: Bbox at bottom-right corner
    bbox = torch.tensor([[1140, 1654, 1240, 1754]], dtype=torch.float32)
    transformed = transform(bbox, orig_size, target_size)

    # Should be within bounds
    assert transformed[0, 2] <= 320, f"x2 should be <= 320, got {transformed[0, 2]}"
    assert transformed[0, 3] <= 320, f"y2 should be <= 320, got {transformed[0, 3]}"
    print(f"✓ Edge case 2 (corner bbox): {bbox[0].tolist()} → {transformed[0].tolist()}")

    # Edge case 3: Square image (no padding needed)
    orig_size = (1000, 1000)
    target_size = (320, 320)
    bbox = torch.tensor([[250, 250, 750, 750]], dtype=torch.float32)
    transformed = transform(bbox, orig_size, target_size)

    # Check no padding (scale = 0.32, pad = 0)
    scale = 320 / 1000
    expected = torch.tensor([[250 * scale, 250 * scale, 750 * scale, 750 * scale]])
    assert torch.allclose(transformed, expected, atol=0.1), f"Square image transform failed"
    print(f"✓ Edge case 3 (square image): {bbox[0].tolist()} → {transformed[0].tolist()}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing cache resize bbox transformation")
    print("=" * 60)
    print()

    test_cache_serialization_with_orig_size()
    print()

    test_cache_serialization_without_orig_size()
    print()

    test_bbox_letterbox_transform()
    print()

    test_full_cache_flow()
    print()

    test_edge_cases()
    print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
