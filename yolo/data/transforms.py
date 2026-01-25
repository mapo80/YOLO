"""
YOLO Transforms - Using torchvision.transforms.v2 for object detection.

These transforms handle both images and bounding boxes correctly.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes


# =============================================================================
# Shared Transform Helpers (used by both RandomPerspective and bbox_mosaic)
# =============================================================================


def apply_hsv_numpy(
    img: np.ndarray,
    h_gain: float = 0.0,
    s_gain: float = 0.0,
    v_gain: float = 0.0,
) -> np.ndarray:
    """
    Apply HSV augmentation to a numpy image.

    This is a shared helper used by both RandomHSV (via tensor conversion)
    and bbox_mosaic (directly on numpy crops).

    Args:
        img: BGR or RGB numpy array (H, W, 3) in uint8
        h_gain: Hue gain factor (0.0 to disable)
        s_gain: Saturation gain factor (0.0 to disable)
        v_gain: Value/brightness gain factor (0.0 to disable)

    Returns:
        Augmented image as numpy array
    """
    if h_gain == 0 and s_gain == 0 and v_gain == 0:
        return img

    # Random multipliers in range [1-gain, 1+gain]
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1

    # Convert to HSV
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    dtype = img.dtype

    # Apply gains
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    im_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    )
    return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)


def build_affine_matrix(
    w: int,
    h: int,
    new_w: int,
    new_h: int,
    degrees: float = 0.0,
    translate: float = 0.0,
    scale: float = 0.0,
    shear: float = 0.0,
    perspective: float = 0.0,
) -> np.ndarray:
    """
    Build a 3x3 homography matrix for affine/perspective augmentation.

    This is a shared helper used by both RandomPerspective and bbox_mosaic.
    All transformations are centered on the image center.

    Args:
        w, h: Original image dimensions
        new_w, new_h: Output image dimensions
        degrees: Max rotation degrees (+/-), 0 to disable
        translate: Max translation as fraction of output size, 0 to disable
        scale: Scale range (1-scale to 1+scale), 0 to disable
        shear: Max shear degrees (+/-), 0 to disable
        perspective: Perspective distortion factor, 0 to disable

    Returns:
        3x3 transformation matrix (np.float32)
    """
    # Shift origin to image center (so rotations/scales are around the center)
    shift_to_center = np.eye(3, dtype=np.float32)
    shift_to_center[0, 2] = -w / 2
    shift_to_center[1, 2] = -h / 2

    # Perspective (optional)
    persp_x = 0.0
    persp_y = 0.0
    if perspective != 0:
        persp_x = random.uniform(-perspective, perspective)
        persp_y = random.uniform(-perspective, perspective)
    persp_mat = np.eye(3, dtype=np.float32)
    persp_mat[2, 0] = persp_x
    persp_mat[2, 1] = persp_y

    # Rotation + scale
    angle = random.uniform(-degrees, degrees) if degrees != 0 else 0.0
    scale_factor = random.uniform(1 - scale, 1 + scale) if scale != 0 else 1.0
    rot_scale = np.eye(3, dtype=np.float32)
    rot_scale[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale_factor)

    # Shear (optional)
    shear_mat = np.eye(3, dtype=np.float32)
    if shear != 0:
        shear_mat[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
        shear_mat[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # Translate back into output canvas coordinates
    translate_mat = np.eye(3, dtype=np.float32)
    translate_mat[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_w
    translate_mat[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_h

    return translate_mat @ shear_mat @ rot_scale @ persp_mat @ shift_to_center


def apply_affine_to_image(
    img: np.ndarray,
    M: np.ndarray,
    new_w: int,
    new_h: int,
    fill_value: int = 114,
    use_perspective: bool = False,
) -> np.ndarray:
    """
    Apply affine/perspective transformation to numpy image.

    Args:
        img: Input image (H, W, 3) numpy array
        M: 3x3 transformation matrix
        new_w, new_h: Output dimensions
        fill_value: Border fill value (default: 114 gray)
        use_perspective: If True, use warpPerspective; else warpAffine

    Returns:
        Transformed image as numpy array
    """
    border_value = (fill_value, fill_value, fill_value)

    if use_perspective:
        return cv2.warpPerspective(
            img, M, dsize=(new_w, new_h), borderValue=border_value
        )
    else:
        return cv2.warpAffine(
            img, M[:2], dsize=(new_w, new_h), borderValue=border_value
        )


class YOLOTargetTransform:
    """
    Transform COCO annotations to YOLO format.

    Converts from COCO format (list of annotation dicts) to:
        {
            "boxes": Tensor[N, 4] in xyxy format,
            "labels": Tensor[N] class indices (0-indexed)
        }

    Args:
        category_id_to_idx: Optional mapping from COCO category_id to 0-indexed class index.
                           If None, category_id values are used directly (assumes 0-indexed).
                           For standard COCO datasets (1-indexed), pass a mapping like {1:0, 2:1, ...}.
    """

    def __init__(self, category_id_to_idx: Optional[Dict[int, int]] = None):
        self.category_id_to_idx = category_id_to_idx

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

            # Map category_id to class index
            category_id = ann["category_id"]
            if self.category_id_to_idx is not None:
                if category_id not in self.category_id_to_idx:
                    continue  # Skip unknown categories
                class_idx = self.category_id_to_idx[category_id]
            else:
                class_idx = category_id

            boxes.append([x1, y1, x2, y2])
            labels.append(class_idx)

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


class RandomPerspective:
    """
    Random perspective transformation with rotation, translation, scale, shear.

    Applies affine and perspective transformations to image and updates bounding boxes.
    All parameters can be set to 0 to disable the corresponding transformation.

    Args:
        degrees: Max rotation degrees (+/-). Set to 0 to disable rotation.
        translate: Max translation as fraction of image size. Set to 0 to disable.
        scale: Scale range (1-scale to 1+scale). Set to 0 to disable scaling.
        shear: Max shear degrees (+/-). Set to 0 to disable shear.
        perspective: Perspective distortion factor. Set to 0 to disable.
        border: Border size for mosaic cropping (negative values crop).
        fill_value: Fill value for padding (default: 114 gray).
    """

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        border: Tuple[int, int] = (0, 0),
        fill_value: int = 114,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.fill_value = fill_value

    def __call__(
        self,
        image: Tensor,
        target: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Apply random perspective transform.

        Args:
            image: Image tensor [C, H, W] in range [0, 1]
            target: Dict with 'boxes' and 'labels'

        Returns:
            Transformed image and updated target
        """
        # Skip if all transforms are disabled
        if (
            self.degrees == 0
            and self.translate == 0
            and self.scale == 0
            and self.shear == 0
            and self.perspective == 0
        ):
            return image, target

        # Convert tensor to numpy for cv2
        _, h, w = image.shape
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Calculate new image size with border
        new_h = h + self.border[1] * 2
        new_w = w + self.border[0] * 2

        # Build transformation matrix using shared helper
        M = build_affine_matrix(
            w, h, new_w, new_h,
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
        )

        # Apply transformation using shared helper
        img_warped = apply_affine_to_image(
            img_np, M, new_w, new_h,
            fill_value=self.fill_value,
            use_perspective=(self.perspective != 0),
        )

        # Convert back to tensor
        image_out = torch.from_numpy(img_warped).permute(2, 0, 1).float() / 255.0

        # Transform bounding boxes
        if len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            labels = target["labels"].clone()
            boxes, labels = self._transform_boxes(boxes, labels, M, new_w, new_h)
            target = {"boxes": boxes, "labels": labels}

        return image_out, target

    def _transform_boxes(
        self,
        boxes: Tensor,
        labels: Tensor,
        M: np.ndarray,
        new_w: int,
        new_h: int,
        min_area: float = 4.0,
        min_visibility: float = 0.1,
    ) -> Tuple[Tensor, Tensor]:
        """
        Transform bounding boxes through the perspective matrix.

        Args:
            boxes: Boxes in xyxy format [N, 4]
            labels: Class labels [N]
            M: 3x3 transformation matrix
            new_w, new_h: New image dimensions
            min_area: Minimum box area to keep (pixels)
            min_visibility: Minimum fraction of original box that must be visible

        Returns:
            Transformed boxes and labels (filtered)
        """
        n = len(boxes)
        if n == 0:
            return boxes, labels

        # Get original box areas for visibility filtering
        orig_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Convert boxes to 4 corner points
        # corners: [N, 4, 2] - 4 corners per box, each corner is (x, y)
        xy = torch.zeros((n, 4, 2), dtype=torch.float32)
        xy[:, 0] = boxes[:, [0, 1]]  # top-left
        xy[:, 1] = boxes[:, [2, 1]]  # top-right
        xy[:, 2] = boxes[:, [2, 3]]  # bottom-right
        xy[:, 3] = boxes[:, [0, 3]]  # bottom-left

        # Reshape to [N*4, 2] for transformation
        xy = xy.reshape(-1, 2).numpy()

        # Add homogeneous coordinate
        ones = np.ones((xy.shape[0], 1), dtype=np.float32)
        xy_h = np.hstack([xy, ones])  # [N*4, 3]

        # Transform points
        xy_t = xy_h @ M.T  # [N*4, 3]

        # Perspective division
        xy_t = xy_t[:, :2] / (xy_t[:, 2:3] + 1e-8)  # [N*4, 2]

        # Reshape back to [N, 4, 2]
        xy_t = xy_t.reshape(n, 4, 2)

        # Get new bounding boxes (axis-aligned)
        x_coords = xy_t[:, :, 0]
        y_coords = xy_t[:, :, 1]

        new_boxes = torch.zeros((n, 4), dtype=torch.float32)
        new_boxes[:, 0] = torch.from_numpy(x_coords.min(axis=1))
        new_boxes[:, 1] = torch.from_numpy(y_coords.min(axis=1))
        new_boxes[:, 2] = torch.from_numpy(x_coords.max(axis=1))
        new_boxes[:, 3] = torch.from_numpy(y_coords.max(axis=1))

        # Clip boxes to image bounds
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clamp(0, new_w)
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clamp(0, new_h)

        # Filter boxes
        new_areas = (new_boxes[:, 2] - new_boxes[:, 0]) * (
            new_boxes[:, 3] - new_boxes[:, 1]
        )
        visibility = new_areas / (orig_areas + 1e-8)

        # Keep boxes with sufficient area and visibility
        valid = (new_areas >= min_area) & (visibility >= min_visibility)

        return new_boxes[valid], labels[valid]


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
        hsv_h: HSV hue augmentation (0 to disable)
        hsv_s: HSV saturation augmentation (0 to disable)
        hsv_v: HSV value augmentation (0 to disable)
        degrees: Max rotation degrees (+/-), 0 to disable
        translate: Max translation as fraction of image size, 0 to disable
        scale: Scale range (1-scale to 1+scale), 0 to disable
        shear: Max shear degrees (+/-), 0 to disable
        perspective: Perspective distortion factor, 0 to disable
        flip_lr: Horizontal flip probability (0 to disable)
        flip_ud: Vertical flip probability (0 to disable)

    Returns:
        Composed transform pipeline
    """
    transforms = [
        LetterBox(target_size=image_size),
        RandomPerspective(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
        ),
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
