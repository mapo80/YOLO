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

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Get item with optional mosaic/mixup/cutmix augmentation."""
        if self._mosaic_enabled and random.random() < self.mosaic_prob:
            # Apply mosaic
            if random.random() < self.mosaic_9_prob:
                image, target = self._mosaic9(index)
            else:
                image, target = self._mosaic4(index)

            # Optionally apply mixup on top of mosaic
            if random.random() < self.mixup_prob:
                image, target = self._mixup(image, target)
        else:
            # Regular single image
            image, target = self._load_image_target(index)

            # Optionally apply cutmix (only when mosaic is not applied)
            if self._mosaic_enabled and random.random() < self.cutmix_prob:
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
        The final image is cropped back to image_size using the border.
        """
        s = self.image_size[0]  # Assume square for simplicity

        # Sample 3 additional random indices
        indices = [index] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]

        # Random center point for mosaic (can be outside image bounds)
        # yc, xc in range [-border, 2*s + border] which is [s/2, 3*s/2]
        yc = int(random.uniform(-self.border[1], 2 * s + self.border[1]))
        xc = int(random.uniform(-self.border[0], 2 * s + self.border[0]))

        # Create 2x canvas
        canvas_side = s * 2
        canvas = np.full((canvas_side, canvas_side, 3), self.fill_value, dtype=np.uint8)

        placed = []
        placements = (
            (False, False),  # top-left
            (True, False),   # top-right
            (False, True),   # bottom-left
            (True, True),    # bottom-right
        )

        for (place_right, place_down), idx in zip(placements, indices):
            patch, target, _ = self._load_and_resize(idx, (s, s))
            patch_h, patch_w = patch.shape[:2]

            x0 = xc if place_right else xc - patch_w
            y0 = yc if place_down else yc - patch_h

            (dx0, dy0, dx1, dy1), (sx0, sy0, sx1, sy1) = self._compute_paste_slices(
                x0=x0,
                y0=y0,
                patch_w=patch_w,
                patch_h=patch_h,
                canvas_w=canvas_side,
                canvas_h=canvas_side,
            )

            canvas[dy0:dy1, dx0:dx1] = patch[sy0:sy1, sx0:sx1]

            if len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0
                placed.append((boxes, target["labels"]))

        # Crop from 2x canvas to final size using border
        # The border defines the crop region: from -border to s+(-border) = from s/2 to 3s/2
        # which gives us the center s x s region
        crop_x1 = -self.border[0]  # s/2
        crop_y1 = -self.border[1]  # s/2
        final_img = canvas[crop_y1 : crop_y1 + s, crop_x1 : crop_x1 + s]

        # Adjust boxes for crop and clip
        final_boxes = []
        final_labels = []
        for boxes, labels in placed:
            boxes = boxes.clone()
            boxes[:, [0, 2]] -= crop_x1
            boxes[:, [1, 3]] -= crop_y1
            # Clip to image bounds
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, s)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, s)
            # Filter small/invalid boxes (min area 4 pixels)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid = areas >= 4.0
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

        return Image.fromarray(final_img), {"boxes": combined_boxes, "labels": combined_labels}

    def _mosaic9(self, index: int) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """
        Create 9-way mosaic from 9 random images.

        Uses 3x3 canvas with images placed relative to the center image.
        Final output is cropped to image_size.
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
                boxes[:, [0, 2]] += x0 + self.border[0]
                boxes[:, [1, 3]] += y0 + self.border[1]
                placed.append((boxes, target["labels"]))

        # Crop using border (from -border to s-border on the 3x canvas)
        # border = (-s/2, -s/2), so we crop from s/2 to s/2+s
        crop_x1 = -self.border[0]  # s/2
        crop_y1 = -self.border[1]  # s/2
        final_img = canvas[crop_y1 : crop_y1 + s, crop_x1 : crop_x1 + s]

        # Adjust boxes - they already have border offset applied
        final_boxes = []
        final_labels = []
        for boxes, labels in placed:
            boxes = boxes.clone()
            # Clip to image bounds
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, s)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, s)
            # Filter small boxes
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid = areas >= 4.0
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

        return Image.fromarray(final_img), {"boxes": combined_boxes, "labels": combined_labels}

    def _mixup(
        self,
        image1: Image.Image,
        target1: Dict[str, Tensor],
    ) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """
        Apply MixUp augmentation - blend two images.

        Uses Beta(alpha, alpha) distribution for blending weight.
        Default alpha=32.0 produces values concentrated around 0.5.
        """
        # Get another mosaic image for mixing
        idx2 = random.randint(0, len(self.dataset) - 1)

        # Apply mosaic to second image too for proper mixing
        if random.random() < self.mosaic_9_prob:
            image2, target2 = self._mosaic9(idx2)
        else:
            image2, target2 = self._mosaic4(idx2)

        # Ensure same size
        if image2.size != image1.size:
            image2 = image2.resize(image1.size, Image.BILINEAR)

        # Sample mixing weight from Beta distribution
        r = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        # Blend images: img = r * img1 + (1 - r) * img2
        img1_np = np.array(image1).astype(np.float32)
        img2_np = np.array(image2).astype(np.float32)
        mixed_np = (img1_np * r + img2_np * (1 - r)).astype(np.uint8)
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

        return mixed_img, {"boxes": combined_boxes, "labels": combined_labels}

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
