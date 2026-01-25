"""
Multi-image augmentations: Mosaic, MixUp, CutMix.

These augmentations require access to multiple images from the dataset,
so they are implemented as a dataset wrapper rather than transforms.

All augmentations can be individually enabled/disabled via probability parameters.
Set probability to 0.0 to disable any augmentation.
"""

import random
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class MosaicMixupDataset(Dataset):
    """
    Dataset wrapper for multi-image augmentations.

    Supports Mosaic (4-way and 9-way), MixUp, and CutMix augmentations.
    All augmentations can be individually enabled/disabled via probability parameters.

    Args:
        dataset: Base dataset returning (image, target) tuples.
            Images should be PIL Images, targets should be dicts with 'boxes' and 'labels'.
        image_size: Target image size (width, height)
        mosaic_prob: Probability of applying mosaic (0.0 to disable, 1.0 always)
        mosaic_9_prob: Probability of 9-way vs 4-way when mosaic is applied
        mixup_prob: Probability of applying mixup after mosaic (0.0 to disable)
        mixup_alpha: Beta distribution parameter for mixup weight (default: 32.0)
        cutmix_prob: Probability of applying cutmix (0.0 to disable)
        cutmix_beta: Beta distribution parameter for cutmix area (default: 1.0)
        transforms: Post-augmentation transforms to apply
        fill_value: Fill value for padding (default: 114 gray)
        buffer_size: Size of LRU buffer for caching recently accessed images.
            Default is 0 (disabled) because the buffer is not shared between
            DataLoader workers - each worker gets its own copy, providing no
            benefit with num_workers > 0.
    """

    def __init__(
        self,
        dataset: Dataset,
        image_size: Tuple[int, int] = (640, 640),
        mosaic_prob: float = 1.0,
        mosaic_9_prob: float = 0.0,
        mixup_prob: float = 0.15,
        mixup_alpha: float = 32.0,
        cutmix_prob: float = 0.0,
        cutmix_beta: float = 1.0,
        transforms: Optional[Callable] = None,
        fill_value: int = 114,
        buffer_size: int = 0,  # Disabled by default - doesn't help with multiprocessing
        # bbox_mosaic: specialized mosaic that crops to bounding box
        # Each image is cropped to its bbox, transformed individually, then placed in grid
        # Pass as dict: {"prob": 1.0, "degrees": 5.0, "translate": 0.05, ...}
        bbox_mosaic: Optional[Dict[str, float]] = None,
    ):
        self.dataset = dataset
        self.image_size = image_size
        self.mosaic_prob = mosaic_prob
        self.mosaic_9_prob = mosaic_9_prob
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob
        self.cutmix_beta = cutmix_beta
        self.transforms = transforms
        self.fill_value = fill_value
        self._mosaic_enabled = True
        # Border for mosaic: allows center to be outside image bounds
        self.border = (-image_size[0] // 2, -image_size[1] // 2)

        # bbox_mosaic parameters (parsed from dict config)
        bbox_cfg = bbox_mosaic or {}
        self.bbox_mosaic_prob = bbox_cfg.get("prob", 0.0)
        self.bbox_mosaic_degrees = bbox_cfg.get("degrees", 0.0)
        self.bbox_mosaic_translate = bbox_cfg.get("translate", 0.0)
        self.bbox_mosaic_scale = bbox_cfg.get("scale", 0.0)
        self.bbox_mosaic_shear = bbox_cfg.get("shear", 0.0)
        self.bbox_mosaic_perspective = bbox_cfg.get("perspective", 0.0)
        self.bbox_mosaic_hsv_h = bbox_cfg.get("hsv_h", 0.0)
        self.bbox_mosaic_hsv_s = bbox_cfg.get("hsv_s", 0.0)
        self.bbox_mosaic_hsv_v = bbox_cfg.get("hsv_v", 0.0)
        self.bbox_mosaic_jitter = bbox_cfg.get("jitter", 0.0)

        # LRU buffer for caching recently accessed images
        # NOTE: This buffer is NOT shared between DataLoader workers (each worker
        # gets its own copy), so it provides minimal benefit with num_workers > 0.
        # It's disabled by default. Enable only for single-threaded loading.
        self._buffer_size = buffer_size
        self._buffer = None
        if buffer_size > 0:
            from yolo.data.cache import LRUImageBuffer

            self._buffer = LRUImageBuffer(buffer_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getstate__(self) -> dict:
        """Prepare state for pickling (spawn multiprocessing compatibility)."""
        state = self.__dict__.copy()
        # The LRU buffer doesn't need to be pickled - it will be recreated per worker
        state["_buffer"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)
        # Recreate the buffer if it was configured
        if self._buffer_size > 0:
            from yolo.data.cache import LRUImageBuffer
            self._buffer = LRUImageBuffer(self._buffer_size)

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Get item with optional mosaic/mixup/cutmix augmentation."""
        apply_bbox_mosaic = self._mosaic_enabled and random.random() < self.bbox_mosaic_prob
        apply_mosaic = self._mosaic_enabled and random.random() < self.mosaic_prob
        apply_mixup = self._mosaic_enabled and random.random() < self.mixup_prob
        apply_cutmix = self._mosaic_enabled and random.random() < self.cutmix_prob

        # bbox_mosaic has priority over standard mosaic (mutually exclusive)
        if apply_bbox_mosaic:
            # bbox_mosaic: crops to bounding box, transforms individually, places in grid
            # Returns target-size canvas (no 2x expansion)
            image, target = self._bbox_mosaic4(index)
            # Note: bbox_mosaic returns final size, so CenterCropWithBoxes will skip it
        elif apply_mosaic:
            # Standard mosaic: may crop images at quadrant boundaries
            if random.random() < self.mosaic_9_prob:
                image, target = self._mosaic9(index)
            else:
                image, target = self._mosaic4(index)
        else:
            # Regular single image
            image, target = self._load_image_target(index)

        # MixUp: blend with another image (can be applied with or without mosaic)
        # Skip MixUp for bbox_mosaic to preserve the clean grid layout
        if apply_mixup and not apply_bbox_mosaic:
            image, target = self._mixup(image, target, apply_mosaic)

        # CutMix: only when mosaic is not applied (as before)
        if not apply_mosaic and not apply_bbox_mosaic and apply_cutmix:
            image, target = self._cutmix(image, target)

        # Apply post-augmentation transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _load_image_target(
        self, index: int
    ) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """
        Load a single image and target from base dataset.

        Uses LRU buffer to cache recently accessed images for faster
        mosaic operations when the same images are reused.

        Note: Images are stored as numpy arrays in the buffer to avoid
        PIL file handle leaks. This is important for large batch sizes
        with many workers.
        """
        # Check buffer first
        if self._buffer is not None:
            cached = self._buffer.get(index)
            if cached is not None:
                # Cached as (numpy_array, target), convert back to PIL
                img_np, target = cached
                return Image.fromarray(img_np), {
                    "boxes": target["boxes"].clone(),
                    "labels": target["labels"].clone(),
                }

        # Load from dataset
        img, target = self.dataset[index]

        # Store in buffer as numpy to avoid file handle leaks
        # PIL images may hold file handles; numpy arrays don't
        if self._buffer is not None:
            img_np = np.asarray(img).copy()  # Ensure contiguous copy
            self._buffer.put(index, (img_np, target))

        return img, target

    def _load_and_resize(
        self, index: int, target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, Dict[str, Tensor], float]:
        """
        Load image and resize to target size maintaining aspect ratio.

        Returns:
            img_np: Resized image as numpy array
            target: Updated target with scaled boxes
            scale: Scale factor applied
        """
        img, target = self._load_image_target(index)

        # Convert to numpy
        img_np = np.array(img)
        h0, w0 = img_np.shape[:2]
        tw, th = target_size

        # Calculate scale to fit
        scale = min(tw / w0, th / h0)
        new_w = int(w0 * scale)
        new_h = int(h0 * scale)

        # Resize
        img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Scale boxes
        if len(target["boxes"]) > 0:
            target = {
                "boxes": target["boxes"].clone(),
                "labels": target["labels"].clone(),
            }
            target["boxes"][:, [0, 2]] *= scale
            target["boxes"][:, [1, 3]] *= scale

        return img_resized, target, scale

    @staticmethod
    def _compute_paste_slices(
        x0: int,
        y0: int,
        patch_w: int,
        patch_h: int,
        canvas_w: int,
        canvas_h: int,
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Compute destination (canvas) and source (patch) slices for pasting with clipping.

        Args:
            x0, y0: Desired top-left corner on the canvas
            patch_w, patch_h: Patch dimensions
            canvas_w, canvas_h: Canvas dimensions

        Returns:
            ((dx0, dy0, dx1, dy1), (sx0, sy0, sx1, sy1)) slices where:
              - canvas[dy0:dy1, dx0:dx1] = patch[sy0:sy1, sx0:sx1]
        """
        dx0 = max(x0, 0)
        dy0 = max(y0, 0)
        dx1 = min(x0 + patch_w, canvas_w)
        dy1 = min(y0 + patch_h, canvas_h)

        sx0 = dx0 - x0
        sy0 = dy0 - y0
        sx1 = sx0 + (dx1 - dx0)
        sy1 = sy0 + (dy1 - dy0)

        return (dx0, dy0, dx1, dy1), (sx0, sy0, sx1, sy1)

    def _mosaic4(self, index: int) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """
        Create 4-way mosaic from 4 random images.

        Uses a 2x image_size canvas with random center point placement.
        Each image is placed in its quadrant and CROPPED to fit (not overlapped).
        Images don't overlap and letterbox (gray) areas are visible
        where images don't fully cover their quadrant.

        Returns the full 2x canvas - cropping is delegated to transforms.
        """
        s = self.image_size[0]  # Assume square for simplicity

        # Sample 3 additional random indices
        indices = [index] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]

        # Random center point for mosaic
        # yc, xc in range [s/2, 3*s/2] for s=imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)

        # Create 2x canvas
        canvas_side = s * 2
        canvas = np.full((canvas_side, canvas_side, 3), self.fill_value, dtype=np.uint8)

        placed = []

        for i, idx in enumerate(indices):
            patch, target, _ = self._load_and_resize(idx, (s, s))
            h, w = patch.shape[:2]

            # Calculate destination (canvas) and source (patch) coordinates
            # Each image is CROPPED to fit its quadrant
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, canvas_side), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(canvas_side, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right (i == 3)
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, canvas_side), min(canvas_side, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # Paste the PORTION of the image that fits in the quadrant
            canvas[y1a:y2a, x1a:x2a] = patch[y1b:y2b, x1b:x2b]

            # Calculate padding offset for box coordinates
            padw = x1a - x1b
            padh = y1a - y1b

            if len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] += padw
                boxes[:, [1, 3]] += padh
                placed.append((boxes, target["labels"]))

        # Clip boxes to canvas bounds and filter zero-area boxes
        final_boxes = []
        final_labels = []
        for boxes, labels in placed:
            boxes = boxes.clone()
            # Clip to canvas bounds (NOT final image size)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, canvas_side)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, canvas_side)
            # Filter zero-area boxes
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid = areas > 0
            if valid.any():
                final_boxes.append(boxes[valid])
                final_labels.append(labels[valid])

        # Combine all boxes and labels
        if final_boxes:
            combined_boxes = torch.cat(final_boxes, dim=0)
            combined_labels = torch.cat(final_labels, dim=0)
        else:
            combined_boxes = torch.zeros((0, 4), dtype=torch.float32)
            combined_labels = torch.zeros((0,), dtype=torch.long)

        # Return full 2x canvas with mosaic center for cropping
        # The center (xc, yc) is passed to CenterCropWithBoxes so the crop
        # position varies based on the random mosaic center
        target = {"boxes": combined_boxes, "labels": combined_labels, "_mosaic_center": (xc, yc)}
        return Image.fromarray(canvas), target

    def _mosaic9(self, index: int) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """
        Create 9-way mosaic from 9 random images.

        Uses 3x3 canvas with images placed relative to the center image.
        Returns the full 3x canvas - cropping is delegated to transforms.
        """
        s = self.image_size[0]

        # Sample 8 additional random indices
        indices = [index] + [random.randint(0, len(self.dataset) - 1) for _ in range(8)]

        # Create 3x canvas
        canvas_side = s * 3
        canvas = np.full((canvas_side, canvas_side, 3), self.fill_value, dtype=np.uint8)

        placed = []
        prev_h, prev_w = -1, -1
        center_h, center_w = 0, 0

        for pos, idx in enumerate(indices):
            patch, target, _ = self._load_and_resize(idx, (s, s))
            patch_h, patch_w = patch.shape[:2]

            if pos == 0:
                center_h, center_w = patch_h, patch_w

            # Determine top-left corner for each tile.
            if pos == 0:  # center
                x0, y0 = s, s
            elif pos == 1:  # top
                x0, y0 = s, s - patch_h
            elif pos == 2:  # top-right
                x0, y0 = s + prev_w, s - patch_h
            elif pos == 3:  # right
                x0, y0 = s + center_w, s
            elif pos == 4:  # bottom-right
                x0, y0 = s + center_w, s + prev_h
            elif pos == 5:  # bottom
                x0, y0 = s + center_w - patch_w, s + center_h
            elif pos == 6:  # bottom-left
                x0, y0 = s + center_w - prev_w - patch_w, s + center_h
            elif pos == 7:  # left
                x0, y0 = s - patch_w, s + center_h - patch_h
            else:  # pos == 8, top-left
                x0, y0 = s - patch_w, s + center_h - prev_h - patch_h

            (dx0, dy0, dx1, dy1), (sx0, sy0, sx1, sy1) = self._compute_paste_slices(
                x0=x0,
                y0=y0,
                patch_w=patch_w,
                patch_h=patch_h,
                canvas_w=canvas_side,
                canvas_h=canvas_side,
            )

            canvas[dy0:dy1, dx0:dx1] = patch[sy0:sy1, sx0:sx1]
            prev_h, prev_w = patch_h, patch_w

            if len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0
                placed.append((boxes, target["labels"]))

        # Clip boxes to canvas bounds and filter zero-area boxes
        final_boxes = []
        final_labels = []
        for boxes, labels in placed:
            boxes = boxes.clone()
            # Clip to canvas bounds (NOT final image size)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, canvas_side)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, canvas_side)
            # Filter zero-area boxes
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid = areas > 0
            if valid.any():
                final_boxes.append(boxes[valid])
                final_labels.append(labels[valid])

        # Combine
        if final_boxes:
            combined_boxes = torch.cat(final_boxes, dim=0)
            combined_labels = torch.cat(final_labels, dim=0)
        else:
            combined_boxes = torch.zeros((0, 4), dtype=torch.float32)
            combined_labels = torch.zeros((0,), dtype=torch.long)

        # Return full 3x canvas - cropping delegated to transforms
        return Image.fromarray(canvas), {"boxes": combined_boxes, "labels": combined_labels}

    def _bbox_mosaic4(self, index: int) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """
        Create 4-way mosaic from bbox-cropped images.

        Unlike standard mosaic which can crop images at quadrant boundaries,
        bbox_mosaic ensures each object is fully visible:

        1. Load 4 images and crop each to its bounding box (extract document only)
        2. Apply individual transforms to each crop (degrees, scale, hsv, etc.)
        3. Scale crops to fit in quadrants without exceeding 50% of canvas
        4. Place in 2x2 grid centered in each quadrant (no overlapping)
        5. Global transforms (from config root) are applied later by post_transforms

        This is ideal for documents, logos, products, or any objects that should
        always appear complete (never partially cropped).

        Returns:
            PIL Image at image_size (NOT 2x canvas like standard mosaic)
            Target dict with combined boxes and labels
        """
        from yolo.data.transforms import apply_hsv_numpy, build_affine_matrix, apply_affine_to_image

        s = self.image_size[0]  # Assume square
        half_s = s // 2

        # Sample 4 indices
        indices = [index] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]

        # Create canvas at target size (NOT 2x like standard mosaic)
        canvas = np.full((s, s, 3), self.fill_value, dtype=np.uint8)

        # Quadrant positions: (x_start, y_start, x_end, y_end)
        quadrants = [
            (0, 0, half_s, half_s),           # top-left
            (half_s, 0, s, half_s),           # top-right
            (0, half_s, half_s, s),           # bottom-left
            (half_s, half_s, s, s),           # bottom-right
        ]

        placed_boxes = []
        placed_labels = []

        for i, idx in enumerate(indices):
            img, target = self._load_image_target(idx)
            img_np = np.array(img)

            # Get first bbox (assuming one main object per image for document use case)
            if len(target["boxes"]) == 0:
                continue

            box = target["boxes"][0]  # [x1, y1, x2, y2]
            label = target["labels"][0]

            # Crop to bbox with padding for transforms
            x1, y1, x2, y2 = box.int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            cropped = img_np[y1:y2, x1:x2].copy()
            crop_h, crop_w = cropped.shape[:2]

            if crop_h < 4 or crop_w < 4:
                continue

            # Apply individual transforms using shared helpers
            # HSV augmentation
            if self.bbox_mosaic_hsv_h > 0 or self.bbox_mosaic_hsv_s > 0 or self.bbox_mosaic_hsv_v > 0:
                cropped = apply_hsv_numpy(
                    cropped,
                    h_gain=self.bbox_mosaic_hsv_h,
                    s_gain=self.bbox_mosaic_hsv_s,
                    v_gain=self.bbox_mosaic_hsv_v,
                )

            # Affine transforms (degrees, translate, scale, shear, perspective)
            has_affine = (
                self.bbox_mosaic_degrees != 0
                or self.bbox_mosaic_translate != 0
                or self.bbox_mosaic_scale != 0
                or self.bbox_mosaic_shear != 0
                or self.bbox_mosaic_perspective != 0
            )

            # Track the bbox through transforms
            # Start with corners of the original crop (before any transform)
            # Corners: top-left, top-right, bottom-right, bottom-left
            orig_corners = np.array([
                [0, 0, 1],
                [crop_w, 0, 1],
                [crop_w, crop_h, 1],
                [0, crop_h, 1],
            ], dtype=np.float32).T  # Shape: (3, 4)

            if has_affine:
                # Calculate expanded canvas size to fit rotated/transformed image
                # For rotation θ: new_W = W*|cos(θ)| + H*|sin(θ)|
                # This ensures the rotated image fits completely without clipping
                max_angle_rad = abs(self.bbox_mosaic_degrees) * np.pi / 180
                cos_a = abs(np.cos(max_angle_rad))
                sin_a = abs(np.sin(max_angle_rad))

                # Add margin for scale variation and shear
                scale_margin = 1 + self.bbox_mosaic_scale + self.bbox_mosaic_shear * 0.02
                expanded_w = int((crop_w * cos_a + crop_h * sin_a) * scale_margin) + 4
                expanded_h = int((crop_w * sin_a + crop_h * cos_a) * scale_margin) + 4

                M = build_affine_matrix(
                    crop_w, crop_h, expanded_w, expanded_h,
                    degrees=self.bbox_mosaic_degrees,
                    translate=self.bbox_mosaic_translate,
                    scale=self.bbox_mosaic_scale,
                    shear=self.bbox_mosaic_shear,
                    perspective=self.bbox_mosaic_perspective,
                )
                cropped = apply_affine_to_image(
                    cropped, M, expanded_w, expanded_h,
                    fill_value=self.fill_value,
                    use_perspective=(self.bbox_mosaic_perspective != 0),
                )

                # Transform the corners through M to get new bbox
                transformed_corners = M @ orig_corners  # Shape: (3, 4)
                # For perspective, normalize by w coordinate
                if self.bbox_mosaic_perspective != 0:
                    transformed_corners = transformed_corners[:2] / transformed_corners[2:3]
                else:
                    transformed_corners = transformed_corners[:2]

                # Get bounding box of transformed corners (in expanded canvas coords)
                bbox_x1 = transformed_corners[0].min()
                bbox_y1 = transformed_corners[1].min()
                bbox_x2 = transformed_corners[0].max()
                bbox_y2 = transformed_corners[1].max()

                # Update dimensions after expansion
                crop_h, crop_w = cropped.shape[:2]
            else:
                # No transform: bbox is the entire crop
                bbox_x1, bbox_y1 = 0, 0
                bbox_x2, bbox_y2 = crop_w, crop_h

            # Scale to fit in quadrant (max half_s x half_s with margin)
            qx1, qy1, qx2, qy2 = quadrants[i]
            quad_w = qx2 - qx1
            quad_h = qy2 - qy1

            # Scale to fit with 15% margin for spacing
            fit_scale = min(quad_w / crop_w, quad_h / crop_h) * 0.85
            new_w = max(4, int(crop_w * fit_scale))
            new_h = max(4, int(crop_h * fit_scale))

            cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Position in quadrant with optional jitter
            margin_x = quad_w - new_w
            margin_y = quad_h - new_h

            if self.bbox_mosaic_jitter > 0 and margin_x > 0 and margin_y > 0:
                # Random offset within jitter range
                # jitter=0.5 means up to 50% of margin in each direction from center
                jitter_x = random.uniform(-self.bbox_mosaic_jitter, self.bbox_mosaic_jitter)
                jitter_y = random.uniform(-self.bbox_mosaic_jitter, self.bbox_mosaic_jitter)
                offset_x = qx1 + int(margin_x * (0.5 + jitter_x * 0.5))
                offset_y = qy1 + int(margin_y * (0.5 + jitter_y * 0.5))
            else:
                # Centered (default)
                offset_x = qx1 + margin_x // 2
                offset_y = qy1 + margin_y // 2

            # Paste on canvas
            canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = cropped

            # Scale the transformed bbox to match the resize and offset
            # bbox coords are in expanded canvas space, scale them to final placement
            scale_ratio = new_w / crop_w  # Same as new_h / crop_h
            final_bbox_x1 = offset_x + bbox_x1 * scale_ratio
            final_bbox_y1 = offset_y + bbox_y1 * scale_ratio
            final_bbox_x2 = offset_x + bbox_x2 * scale_ratio
            final_bbox_y2 = offset_y + bbox_y2 * scale_ratio

            # Clip to canvas bounds
            final_bbox_x1 = max(0, final_bbox_x1)
            final_bbox_y1 = max(0, final_bbox_y1)
            final_bbox_x2 = min(s, final_bbox_x2)
            final_bbox_y2 = min(s, final_bbox_y2)

            new_box = torch.tensor([
                final_bbox_x1, final_bbox_y1,
                final_bbox_x2, final_bbox_y2
            ], dtype=torch.float32)

            placed_boxes.append(new_box)
            placed_labels.append(label)

        # Combine results
        if placed_boxes:
            combined_boxes = torch.stack(placed_boxes)
            combined_labels = torch.stack(placed_labels)
        else:
            combined_boxes = torch.zeros((0, 4), dtype=torch.float32)
            combined_labels = torch.zeros((0,), dtype=torch.long)

        # Return target-size canvas (no 2x expansion needed)
        # Global transforms will be applied by post_transforms pipeline
        return Image.fromarray(canvas), {"boxes": combined_boxes, "labels": combined_labels}

    def _mixup(
        self,
        image1: Image.Image,
        target1: Dict[str, Tensor],
        first_is_mosaic: bool = False,
    ) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """
        Apply MixUp augmentation - blend two images.

        - Uses Beta(32.0, 32.0) distribution for blending weight (produces ~0.5)
        - If first image is mosaic, second image is also mosaic
        - If first image is single, second image is also single
        - Both images are blended at pixel level
        - Boxes and labels from both images are concatenated

        Args:
            image1: First image (PIL Image)
            target1: First target dict with 'boxes' and 'labels'
            first_is_mosaic: Whether image1 came from mosaic (determines image2 source)

        Returns:
            Mixed image and combined target
        """
        # Get second image - same type as first (mosaic or single)
        idx2 = random.randint(0, len(self.dataset) - 1)

        if first_is_mosaic:
            # Second image is also mosaic
            if random.random() < self.mosaic_9_prob:
                image2, target2 = self._mosaic9(idx2)
            else:
                image2, target2 = self._mosaic4(idx2)
        else:
            # Second image is single image (loaded and transformed same as first)
            image2, target2 = self._load_image_target(idx2)

        # Convert to numpy for blending
        img1_np = np.array(image1)
        img2_np = np.array(image2)

        # Ensure same size - resize image2 to match image1
        h1, w1 = img1_np.shape[:2]
        h2, w2 = img2_np.shape[:2]

        if (w1, h1) != (w2, h2):
            # Resize image2 to match image1
            img2_np = cv2.resize(img2_np, (w1, h1), interpolation=cv2.INTER_LINEAR)

            # Scale boxes proportionally
            if len(target2["boxes"]) > 0:
                scale_x = w1 / w2
                scale_y = h1 / h2
                target2 = {
                    "boxes": target2["boxes"].clone(),
                    "labels": target2["labels"].clone(),
                }
                target2["boxes"][:, [0, 2]] *= scale_x
                target2["boxes"][:, [1, 3]] *= scale_y

        # Sample mixing weight from Beta(32.0, 32.0)
        # This produces values concentrated around 0.5
        r = np.random.beta(32.0, 32.0)

        # Blend images: mixed = r * img1 + (1 - r) * img2
        mixed_np = (img1_np.astype(np.float32) * r + img2_np.astype(np.float32) * (1 - r)).astype(np.uint8)
        mixed_img = Image.fromarray(mixed_np)

        # Concatenate boxes and labels from both images
        boxes1 = target1["boxes"]
        labels1 = target1["labels"]
        boxes2 = target2["boxes"]
        labels2 = target2["labels"]

        if len(boxes1) > 0 and len(boxes2) > 0:
            combined_boxes = torch.cat([boxes1, boxes2], dim=0)
            combined_labels = torch.cat([labels1, labels2], dim=0)
        elif len(boxes1) > 0:
            combined_boxes = boxes1
            combined_labels = labels1
        elif len(boxes2) > 0:
            combined_boxes = boxes2
            combined_labels = labels2
        else:
            combined_boxes = torch.zeros((0, 4), dtype=torch.float32)
            combined_labels = torch.zeros((0,), dtype=torch.long)

        # Preserve _mosaic_center if present (for proper cropping later)
        result_target = {"boxes": combined_boxes, "labels": combined_labels}
        if "_mosaic_center" in target1:
            result_target["_mosaic_center"] = target1["_mosaic_center"]

        return mixed_img, result_target

    def _rand_bbox(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Generate random bounding box coordinates for CutMix.

        Uses center-based sampling with clipping to image bounds.

        Returns:
            (x1, y1, x2, y2) coordinates of the bounding box.
        """
        # Sample mixing ratio from Beta distribution
        lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)

        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        # Random center
        cx = np.random.randint(width)
        cy = np.random.randint(height)

        # Bounding box coordinates (clip to image bounds)
        x1 = int(np.clip(cx - cut_w // 2, 0, width))
        y1 = int(np.clip(cy - cut_h // 2, 0, height))
        x2 = int(np.clip(cx + cut_w // 2, 0, width))
        y2 = int(np.clip(cy + cut_h // 2, 0, height))

        return x1, y1, x2, y2

    def _cutmix(
        self,
        image: Image.Image,
        target: Dict[str, Tensor],
    ) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """
        Apply CutMix augmentation - paste rectangular region from another image.

        Uses Beta(beta, beta) distribution for determining cut area.
        Center-based random bbox generation with IoA filtering.
        """
        w, h = image.size

        # Sample another random image
        idx2 = random.randint(0, len(self.dataset) - 1)
        image2, target2 = self._load_image_target(idx2)

        # Resize image2 to match
        if image2.size != image.size:
            orig_w2, orig_h2 = image2.size
            image2 = image2.resize(image.size, Image.BILINEAR)
            # Scale boxes
            if len(target2["boxes"]) > 0:
                scale_x = w / orig_w2
                scale_y = h / orig_h2
                target2 = {
                    "boxes": target2["boxes"].clone(),
                    "labels": target2["labels"].clone(),
                }
                target2["boxes"][:, [0, 2]] *= scale_x
                target2["boxes"][:, [1, 3]] *= scale_y

        # Generate random cut region (center-based with clipping)
        x1, y1, x2, y2 = self._rand_bbox(w, h)

        # Skip if cut region is too small
        if x2 - x1 < 2 or y2 - y1 < 2:
            return image, target

        # Paste region from image2 to image
        image_np = np.array(image)
        image2_np = np.array(image2)
        image_np[y1:y2, x1:x2] = image2_np[y1:y2, x1:x2]
        mixed_img = Image.fromarray(image_np)

        # Update boxes
        boxes1 = target["boxes"]
        labels1 = target["labels"]
        boxes2 = target2["boxes"]
        labels2 = target2["labels"]

        final_boxes = []
        final_labels = []

        # Keep all boxes from image1 (CutMix keeps original labels)
        if len(boxes1) > 0:
            final_boxes.append(boxes1)
            final_labels.append(labels1)

        # Add boxes from image2 that have sufficient overlap with cut region
        # Using IoA (Intersection over Area) threshold of 0.1
        if len(boxes2) > 0:
            cut_area = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            for i in range(len(boxes2)):
                box = boxes2[i]
                bx1, by1, bx2, by2 = box.tolist()
                box_area = (bx2 - bx1) * (by2 - by1)
                if box_area <= 0:
                    continue

                # Calculate intersection with cut region
                ix1 = max(bx1, x1)
                iy1 = max(by1, y1)
                ix2 = min(bx2, x2)
                iy2 = min(by2, y2)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                # IoA = intersection / box_area
                ioa = inter_area / box_area
                if ioa >= 0.1:
                    # Clip box to cut region
                    new_x1 = max(bx1, x1)
                    new_y1 = max(by1, y1)
                    new_x2 = min(bx2, x2)
                    new_y2 = min(by2, y2)
                    new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
                    if new_area >= 4.0:
                        final_boxes.append(
                            torch.tensor(
                                [[new_x1, new_y1, new_x2, new_y2]], dtype=torch.float32
                            )
                        )
                        final_labels.append(labels2[i].unsqueeze(0))

        # Combine boxes
        if final_boxes:
            combined_boxes = torch.cat(final_boxes, dim=0)
            combined_labels = torch.cat(final_labels, dim=0)
        else:
            combined_boxes = torch.zeros((0, 4), dtype=torch.float32)
            combined_labels = torch.zeros((0,), dtype=torch.long)

        return mixed_img, {"boxes": combined_boxes, "labels": combined_labels}

    def disable_mosaic(self) -> None:
        """Disable mosaic, mixup, and cutmix augmentations (for close_mosaic)."""
        self._mosaic_enabled = False

    def enable_mosaic(self) -> None:
        """Re-enable mosaic, mixup, and cutmix augmentations."""
        self._mosaic_enabled = True

    @property
    def mosaic_enabled(self) -> bool:
        """Check if mosaic augmentation is currently enabled."""
        return self._mosaic_enabled


class CenterCropWithBoxes:
    """
    Crop image to target size from center and adjust bounding boxes accordingly.

    This transform is designed to work with mosaic augmentation, which returns
    a larger canvas (2x or 3x image_size). The crop extracts the center region
    and filters out boxes that fall outside or have insufficient area.

    Args:
        size: Target size as (width, height) tuple.
        min_box_area: Minimum box area in pixels to keep (default: 4.0).
    """

    def __init__(self, size: Tuple[int, int], min_box_area: float = 4.0):
        self.size = size  # (width, height)
        self.min_box_area = min_box_area

    def __call__(
        self, image: Image.Image, target: Dict[str, Tensor]
    ) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """
        Apply center crop to image and adjust boxes.

        Args:
            image: PIL Image to crop.
            target: Dict with 'boxes' (N, 4) in xyxy format and 'labels' (N,).

        Returns:
            Cropped image and updated target with filtered boxes.
        """
        w, h = image.size
        tw, th = self.size

        # Only crop if image is a mosaic canvas (exactly 2x or 3x target size)
        # Mosaic4 creates 2x canvas, Mosaic9 creates 3x canvas.
        # Raw images (when mosaic is disabled) should pass through to LetterBox unchanged.
        is_mosaic_canvas = (w == tw * 2 and h == th * 2) or (w == tw * 3 and h == th * 3)
        if not is_mosaic_canvas:
            return image, target

        # Calculate crop offset based on mosaic center if available
        # For mosaic4: use the random center (xc, yc) passed from _mosaic4
        # For mosaic9: use fixed center crop (no _mosaic_center passed)
        # Mosaic4 has random center variation, mosaic9 uses fixed center
        if "_mosaic_center" in target:
            xc, yc = target.pop("_mosaic_center")
            # Crop is centered on the mosaic junction point (xc, yc)
            # Clamp to ensure the crop stays within canvas bounds
            x0 = max(0, min(xc - tw // 2, w - tw))
            y0 = max(0, min(yc - th // 2, h - th))
        else:
            # Mosaic9 or fallback: fixed center crop
            x0 = (w - tw) // 2
            y0 = (h - th) // 2

        # Crop image
        image = image.crop((x0, y0, x0 + tw, y0 + th))

        # Adjust boxes
        boxes = target["boxes"]
        labels = target["labels"]

        if len(boxes) > 0:
            boxes = boxes.clone()

            # Translate boxes by crop offset
            boxes[:, [0, 2]] -= x0
            boxes[:, [1, 3]] -= y0

            # Clip to new image bounds
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, tw)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, th)

            # Filter boxes with insufficient area
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid = areas >= self.min_box_area

            boxes = boxes[valid]
            labels = labels[valid]

        target = {"boxes": boxes, "labels": labels}

        return image, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, min_box_area={self.min_box_area})"
