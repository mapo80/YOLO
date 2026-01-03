"""
YOLO Transforms - Using torchvision.transforms.v2 for object detection.

These transforms handle both images and bounding boxes correctly.
"""

from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes


class YOLOTargetTransform:
    """
    Transform COCO annotations to YOLO format.

    Converts from COCO format (list of annotation dicts) to:
        {
            "boxes": Tensor[N, 4] in xyxy format,
            "labels": Tensor[N] class indices
        }
    """

    def __call__(
        self,
        coco_annotations: List[Dict],
        image_size: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        """
        Args:
            coco_annotations: List of COCO annotation dicts
            image_size: Original image size (width, height)

        Returns:
            Dict with 'boxes' and 'labels' tensors
        """
        boxes = []
        labels = []

        for ann in coco_annotations:
            # Skip crowd annotations
            if ann.get("iscrowd", 0):
                continue

            # Get bbox in COCO format [x, y, width, height]
            if "bbox" not in ann:
                continue

            x, y, w, h = ann["bbox"]

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue

            # Convert to xyxy format
            x1, y1, x2, y2 = x, y, x + w, y + h

            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])

        if len(boxes) == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
            }

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class LetterBox:
    """
    Resize and pad image to target size while maintaining aspect ratio.

    This is the standard YOLO preprocessing that:
    1. Resizes the image to fit within target size
    2. Pads with gray (114, 114, 114) to reach exact target size
    3. Updates bounding boxes accordingly
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        fill_value: int = 114,
        center: bool = True,
    ):
        """
        Args:
            target_size: Target (width, height)
            fill_value: Padding fill value (default: 114 gray)
            center: If True, center image in padded space
        """
        self.target_size = target_size
        self.fill_value = fill_value
        self.center = center

    def __call__(
        self,
        image: Image.Image,
        target: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Apply letterbox transform.

        Args:
            image: PIL Image
            target: Dict with 'boxes' and 'labels'

        Returns:
            Transformed image tensor and updated target
        """
        # Get original and target sizes
        orig_w, orig_h = image.size
        target_w, target_h = self.target_size

        # Calculate scale to fit within target
        scale = min(target_w / orig_w, target_h / orig_h)

        # New size after scaling
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h

        if self.center:
            pad_left = pad_w // 2
            pad_top = pad_h // 2
        else:
            pad_left = 0
            pad_top = 0

        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        # Resize image
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Convert to tensor
        image_tensor = v2.functional.to_image(image)
        image_tensor = v2.functional.to_dtype(image_tensor, torch.float32, scale=True)

        # Pad image
        image_tensor = v2.functional.pad(
            image_tensor,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill_value / 255.0,  # Normalize fill value
        )

        # Update boxes
        if len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            # Scale boxes
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale
            # Shift boxes by padding
            boxes[:, [0, 2]] += pad_left
            boxes[:, [1, 3]] += pad_top
            target["boxes"] = boxes

        return image_tensor, target


class RandomHSV:
    """Random HSV color augmentation."""

    def __init__(self, h_gain: float = 0.015, s_gain: float = 0.7, v_gain: float = 0.4):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(
        self,
        image: Tensor,
        target: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Apply random HSV augmentation."""
        if self.h_gain == 0 and self.s_gain == 0 and self.v_gain == 0:
            return image, target

        # Random gains
        r = torch.rand(3) * 2 - 1  # [-1, 1]
        h_gain = 1 + r[0] * self.h_gain
        s_gain = 1 + r[1] * self.s_gain
        v_gain = 1 + r[2] * self.v_gain

        # Convert to HSV (approximate using transforms)
        # Note: This is a simplified version. For exact HSV transform,
        # convert to numpy/cv2 which is more accurate
        image = v2.functional.adjust_hue(image, (h_gain - 1) * 0.5)
        image = v2.functional.adjust_saturation(image, s_gain)
        image = v2.functional.adjust_brightness(image, v_gain)

        return image, target


class RandomFlip:
    """Random horizontal and vertical flip."""

    def __init__(self, lr_prob: float = 0.5, ud_prob: float = 0.0):
        self.lr_prob = lr_prob
        self.ud_prob = ud_prob

    def __call__(
        self,
        image: Tensor,
        target: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Apply random flips."""
        _, h, w = image.shape

        # Horizontal flip
        if torch.rand(1).item() < self.lr_prob:
            image = v2.functional.horizontal_flip(image)
            if len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes

        # Vertical flip
        if torch.rand(1).item() < self.ud_prob:
            image = v2.functional.vertical_flip(image)
            if len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target["boxes"] = boxes

        return image, target


class Compose:
    """Compose multiple transforms that handle both image and target."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(
        self,
        image,
        target: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def create_train_transforms(
    image_size: Tuple[int, int] = (640, 640),
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.9,
    shear: float = 0.0,
    perspective: float = 0.0,
    flip_lr: float = 0.5,
    flip_ud: float = 0.0,
) -> Compose:
    """
    Create training transforms pipeline.

    Args:
        image_size: Target image size (width, height)
        hsv_h: HSV hue augmentation
        hsv_s: HSV saturation augmentation
        hsv_v: HSV value augmentation
        degrees: Rotation degrees (not implemented yet)
        translate: Translation fraction (not implemented yet)
        scale: Scale range (not implemented yet)
        shear: Shear degrees (not implemented yet)
        perspective: Perspective distortion (not implemented yet)
        flip_lr: Horizontal flip probability
        flip_ud: Vertical flip probability

    Returns:
        Composed transform pipeline
    """
    transforms = [
        LetterBox(target_size=image_size),
        RandomHSV(h_gain=hsv_h, s_gain=hsv_s, v_gain=hsv_v),
        RandomFlip(lr_prob=flip_lr, ud_prob=flip_ud),
    ]

    return Compose(transforms)


def create_val_transforms(
    image_size: Tuple[int, int] = (640, 640),
) -> Compose:
    """
    Create validation transforms pipeline (no augmentation).

    Args:
        image_size: Target image size (width, height)

    Returns:
        Composed transform pipeline
    """
    transforms = [
        LetterBox(target_size=image_size),
    ]

    return Compose(transforms)
