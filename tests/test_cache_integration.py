"""
Integration test for cache resize with bbox transformation.

Tests the actual datamodule flow with real-like data.
"""

import tempfile
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo.data.datamodule import CocoDetectionWrapper
from yolo.data.cache import ImageCache


def create_test_coco_dataset(tmpdir: Path, num_images: int = 5):
    """Create a minimal COCO format dataset for testing."""
    images_dir = tmpdir / "images"
    images_dir.mkdir(parents=True)

    categories = [
        {"id": 1, "name": "document", "supercategory": "object"},
    ]

    images = []
    annotations = []
    ann_id = 1

    for i in range(num_images):
        # Create images with different aspect ratios
        if i % 2 == 0:
            # Portrait
            orig_w, orig_h = 1240, 1754
        else:
            # Landscape
            orig_w, orig_h = 1754, 1240

        # Create test image
        img = Image.new('RGB', (orig_w, orig_h), color=(100 + i * 20, 150, 200))
        img_filename = f"test_image_{i:03d}.jpg"
        img.save(images_dir / img_filename)

        images.append({
            "id": i + 1,
            "file_name": img_filename,
            "width": orig_w,
            "height": orig_h,
        })

        # Create bbox in center of image (50% of image area)
        cx, cy = orig_w / 2, orig_h / 2
        w, h = orig_w * 0.5, orig_h * 0.5
        x1, y1 = cx - w / 2, cy - h / 2

        annotations.append({
            "id": ann_id,
            "image_id": i + 1,
            "category_id": 1,
            "bbox": [x1, y1, w, h],  # COCO format: x, y, w, h
            "area": w * h,
            "iscrowd": 0,
        })
        ann_id += 1

    # Save annotations
    ann_file = tmpdir / "annotations.json"
    with open(ann_file, 'w') as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }, f)

    return images_dir, ann_file, images, annotations


def test_dataset_with_cache_resize():
    """Test that dataset correctly transforms bboxes when cache has resized images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        images_dir, ann_file, images_meta, annotations_meta = create_test_coco_dataset(tmpdir)

        target_size = (320, 320)

        # Create cache with resize enabled
        cache = ImageCache(
            mode='ram',
            cache_dir=tmpdir,
            target_size=target_size,
            refresh=True,
        )

        # Create dataset with cache
        dataset = CocoDetectionWrapper(
            root=str(images_dir),
            annFile=str(ann_file),
            transforms=None,
            image_size=target_size,
            image_cache=cache,
        )

        print(f"Dataset size: {len(dataset)}")

        # Test each image
        for i in range(len(dataset)):
            image, target = dataset[i]
            boxes = target["boxes"]
            labels = target["labels"]

            img_meta = images_meta[i]
            orig_w, orig_h = img_meta["width"], img_meta["height"]
            orig_ann = annotations_meta[i]
            orig_bbox_xywh = orig_ann["bbox"]

            # Convert original COCO bbox to xyxy
            orig_x1 = orig_bbox_xywh[0]
            orig_y1 = orig_bbox_xywh[1]
            orig_x2 = orig_bbox_xywh[0] + orig_bbox_xywh[2]
            orig_y2 = orig_bbox_xywh[1] + orig_bbox_xywh[3]

            # Calculate expected transformed bbox
            scale = min(target_size[0] / orig_w, target_size[1] / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pad_x = (target_size[0] - new_w) // 2
            pad_y = (target_size[1] - new_h) // 2

            expected_x1 = orig_x1 * scale + pad_x
            expected_y1 = orig_y1 * scale + pad_y
            expected_x2 = orig_x2 * scale + pad_x
            expected_y2 = orig_y2 * scale + pad_y

            # Get actual transformed bbox
            actual_x1, actual_y1, actual_x2, actual_y2 = boxes[0].tolist()

            # Verify image size is target size
            assert image.size == target_size, f"Image size mismatch: {image.size} != {target_size}"

            # Verify bbox is within image bounds
            assert 0 <= actual_x1 <= target_size[0], f"x1 out of bounds: {actual_x1}"
            assert 0 <= actual_y1 <= target_size[1], f"y1 out of bounds: {actual_y1}"
            assert 0 <= actual_x2 <= target_size[0], f"x2 out of bounds: {actual_x2}"
            assert 0 <= actual_y2 <= target_size[1], f"y2 out of bounds: {actual_y2}"

            # Verify bbox is approximately correct (with tolerance for rounding)
            tolerance = 2.0
            assert abs(actual_x1 - expected_x1) < tolerance, f"x1 mismatch: {actual_x1} != {expected_x1}"
            assert abs(actual_y1 - expected_y1) < tolerance, f"y1 mismatch: {actual_y1} != {expected_y1}"
            assert abs(actual_x2 - expected_x2) < tolerance, f"x2 mismatch: {actual_x2} != {expected_x2}"
            assert abs(actual_y2 - expected_y2) < tolerance, f"y2 mismatch: {actual_y2} != {expected_y2}"

            # Verify bbox center is roughly in center of image
            actual_cx = (actual_x1 + actual_x2) / 2
            actual_cy = (actual_y1 + actual_y2) / 2
            assert abs(actual_cx - 160) < 10, f"Center X should be ~160, got {actual_cx}"
            assert abs(actual_cy - 160) < 10, f"Center Y should be ~160, got {actual_cy}"

            print(f"✓ Image {i}: orig={orig_w}x{orig_h}, bbox=[{actual_x1:.1f}, {actual_y1:.1f}, {actual_x2:.1f}, {actual_y2:.1f}]")

        print("\n✓ All images have correctly transformed bboxes!")


def test_dataset_without_cache_resize():
    """Test that dataset works correctly without cache resize."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        images_dir, ann_file, images_meta, annotations_meta = create_test_coco_dataset(tmpdir)

        image_size = (640, 640)

        # Create cache WITHOUT resize (target_size=None)
        cache = ImageCache(
            mode='ram',
            cache_dir=tmpdir,
            target_size=None,  # No resize
            refresh=True,
        )

        dataset = CocoDetectionWrapper(
            root=str(images_dir),
            annFile=str(ann_file),
            transforms=None,
            image_size=image_size,  # This doesn't affect cache, only dataset output
            image_cache=cache,
        )

        for i in range(len(dataset)):
            image, target = dataset[i]
            boxes = target["boxes"]

            img_meta = images_meta[i]
            orig_w, orig_h = img_meta["width"], img_meta["height"]

            # Image should be original size (no cache resize)
            assert image.size == (orig_w, orig_h), f"Image size mismatch: {image.size} != ({orig_w}, {orig_h})"

            # Bbox should be in original coordinates
            orig_ann = annotations_meta[i]
            orig_bbox_xywh = orig_ann["bbox"]
            orig_x1 = orig_bbox_xywh[0]
            orig_y1 = orig_bbox_xywh[1]
            orig_x2 = orig_bbox_xywh[0] + orig_bbox_xywh[2]
            orig_y2 = orig_bbox_xywh[1] + orig_bbox_xywh[3]

            actual_x1, actual_y1, actual_x2, actual_y2 = boxes[0].tolist()

            tolerance = 1.0
            assert abs(actual_x1 - orig_x1) < tolerance, f"x1 mismatch: {actual_x1} != {orig_x1}"
            assert abs(actual_y1 - orig_y1) < tolerance, f"y1 mismatch: {actual_y1} != {orig_y1}"
            assert abs(actual_x2 - orig_x2) < tolerance, f"x2 mismatch: {actual_x2} != {orig_x2}"
            assert abs(actual_y2 - orig_y2) < tolerance, f"y2 mismatch: {actual_y2} != {orig_y2}"

            print(f"✓ Image {i}: size={image.size}, bbox=[{actual_x1:.1f}, {actual_y1:.1f}, {actual_x2:.1f}, {actual_y2:.1f}]")

        print("\n✓ Dataset without cache resize works correctly!")


def test_bbox_area_ratio():
    """Test that bbox area ratio is preserved after transformation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        images_dir, ann_file, images_meta, annotations_meta = create_test_coco_dataset(tmpdir)

        target_size = (320, 320)

        cache = ImageCache(
            mode='ram',
            cache_dir=tmpdir,
            target_size=target_size,
            refresh=True,
        )

        dataset = CocoDetectionWrapper(
            root=str(images_dir),
            annFile=str(ann_file),
            transforms=None,
            image_size=target_size,
            image_cache=cache,
        )

        for i in range(len(dataset)):
            image, target = dataset[i]
            boxes = target["boxes"]

            img_meta = images_meta[i]
            orig_w, orig_h = img_meta["width"], img_meta["height"]
            orig_ann = annotations_meta[i]
            orig_bbox_xywh = orig_ann["bbox"]

            # Calculate original bbox relative area
            orig_bbox_area = orig_bbox_xywh[2] * orig_bbox_xywh[3]
            orig_img_area = orig_w * orig_h
            orig_area_ratio = orig_bbox_area / orig_img_area

            # Calculate transformed bbox area relative to content area (excluding padding)
            x1, y1, x2, y2 = boxes[0].tolist()
            transformed_area = (x2 - x1) * (y2 - y1)

            # Content area in letterboxed image
            scale = min(target_size[0] / orig_w, target_size[1] / orig_h)
            content_w = int(orig_w * scale)
            content_h = int(orig_h * scale)
            content_area = content_w * content_h

            # The ratio should be approximately the same
            transformed_area_ratio = transformed_area / content_area

            # Allow for some tolerance due to rounding
            tolerance = 0.05
            assert abs(orig_area_ratio - transformed_area_ratio) < tolerance, \
                f"Area ratio mismatch: {orig_area_ratio:.4f} != {transformed_area_ratio:.4f}"

            print(f"✓ Image {i}: orig_ratio={orig_area_ratio:.4f}, transformed_ratio={transformed_area_ratio:.4f}")

        print("\n✓ Bbox area ratios are preserved!")


def test_different_image_sizes():
    """Test with various image sizes including edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        images_dir = tmpdir / "images"
        images_dir.mkdir()

        # Test different aspect ratios
        test_cases = [
            (640, 480),   # 4:3 landscape
            (480, 640),   # 3:4 portrait
            (320, 320),   # Square
            (1920, 1080), # 16:9 landscape
            (1080, 1920), # 9:16 portrait
        ]

        categories = [{"id": 1, "name": "object", "supercategory": "object"}]
        images = []
        annotations = []

        for i, (w, h) in enumerate(test_cases):
            img = Image.new('RGB', (w, h), color=(100, 150, 200))
            img_filename = f"test_{i:03d}.jpg"
            img.save(images_dir / img_filename)

            images.append({"id": i + 1, "file_name": img_filename, "width": w, "height": h})

            # Bbox at 25%-75% of image
            bbox_x1, bbox_y1 = w * 0.25, h * 0.25
            bbox_w, bbox_h = w * 0.5, h * 0.5
            annotations.append({
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [bbox_x1, bbox_y1, bbox_w, bbox_h],
                "area": bbox_w * bbox_h,
                "iscrowd": 0,
            })

        ann_file = tmpdir / "annotations.json"
        with open(ann_file, 'w') as f:
            json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

        target_size = (320, 320)
        cache = ImageCache(mode='ram', cache_dir=tmpdir, target_size=target_size, refresh=True)
        dataset = CocoDetectionWrapper(
            root=str(images_dir),
            annFile=str(ann_file),
            image_size=target_size,
            image_cache=cache,
        )

        for i, (orig_w, orig_h) in enumerate(test_cases):
            image, target = dataset[i]
            boxes = target["boxes"]

            # All images should be resized to target_size
            assert image.size == target_size, f"Image {i} size mismatch"

            # All bboxes should be within bounds
            x1, y1, x2, y2 = boxes[0].tolist()
            assert 0 <= x1 < x2 <= target_size[0], f"Image {i}: x bounds invalid [{x1}, {x2}]"
            assert 0 <= y1 < y2 <= target_size[1], f"Image {i}: y bounds invalid [{y1}, {y2}]"

            print(f"✓ {orig_w}x{orig_h} → {target_size}: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

        print("\n✓ All image sizes handled correctly!")


if __name__ == "__main__":
    print("=" * 60)
    print("Integration tests for cache resize with bbox transformation")
    print("=" * 60)
    print()

    print("Test 1: Dataset with cache resize")
    print("-" * 40)
    test_dataset_with_cache_resize()
    print()

    print("Test 2: Dataset without cache resize")
    print("-" * 40)
    test_dataset_without_cache_resize()
    print()

    print("Test 3: Bbox area ratio preservation")
    print("-" * 40)
    test_bbox_area_ratio()
    print()

    print("Test 4: Different image sizes")
    print("-" * 40)
    test_different_image_sizes()
    print()

    print("=" * 60)
    print("All integration tests passed!")
    print("=" * 60)
