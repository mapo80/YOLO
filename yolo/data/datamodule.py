"""
YOLODataModule - PyTorch Lightning data module for COCO format datasets.

Uses torchvision.datasets.CocoDetection for standard COCO format.
"""

import os
from pathlib import Path
from typing import Callable, List, Optional

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2

from yolo.data.loaders import DefaultImageLoader, ImageLoader
from yolo.data.mosaic import MosaicMixupDataset
from yolo.data.transforms import (
    LetterBox,
    YOLOTargetTransform,
    create_train_transforms,
    create_val_transforms,
)
from yolo.utils.logger import logger


class YOLODataModule(L.LightningDataModule):
    """
    Lightning DataModule for YOLO training with COCO format datasets.

    Uses torchvision.datasets.CocoDetection - the standard COCO dataset class.

    Expected directory structure:
        data/coco/
        ├── train2017/
        │   └── *.jpg
        ├── val2017/
        │   └── *.jpg
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json

    Args:
        root: Root directory containing images and annotations
        train_images: Subdirectory for training images
        val_images: Subdirectory for validation images
        train_ann: Path to training annotations JSON (relative to root)
        val_ann: Path to validation annotations JSON (relative to root)
        batch_size: Batch size for training and validation
        num_workers: Number of data loading workers
        image_size: Target image size [width, height]
        pin_memory: Whether to pin memory for faster GPU transfer
        image_loader: Custom image loader (e.g., for encrypted images).
            Can be configured via YAML class_path or CLI.
        # Augmentation parameters
        mosaic_prob: Probability of applying mosaic augmentation
        mixup_prob: Probability of applying mixup augmentation
        hsv_h: HSV hue augmentation range
        hsv_s: HSV saturation augmentation range
        hsv_v: HSV value augmentation range
        degrees: Random rotation degrees
        translate: Random translation fraction
        scale: Random scale range
        shear: Random shear degrees
        perspective: Random perspective distortion
        flip_lr: Horizontal flip probability
        flip_ud: Vertical flip probability
        close_mosaic_epochs: Disable mosaic for last N epochs
    """

    def __init__(
        self,
        # Dataset paths
        root: str = "data/coco",
        train_images: str = "train2017",
        val_images: str = "val2017",
        train_ann: str = "annotations/instances_train2017.json",
        val_ann: str = "annotations/instances_val2017.json",
        # DataLoader settings
        batch_size: int = 16,
        num_workers: int = 8,
        image_size: List[int] = [640, 640],
        pin_memory: bool = True,
        # Custom image loader (e.g., for encrypted images)
        image_loader: Optional[ImageLoader] = None,
        # Multi-image augmentation parameters (set prob to 0.0 to disable)
        mosaic_prob: float = 1.0,
        mosaic_9_prob: float = 0.0,
        mixup_prob: float = 0.15,
        mixup_alpha: float = 32.0,
        cutmix_prob: float = 0.0,
        # Single-image augmentation parameters
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
        # Training schedule
        close_mosaic_epochs: int = 15,
    ):
        super().__init__()
        # Exclude image_loader from hyperparameters (not serializable)
        self.save_hyperparameters(ignore=["image_loader"])

        self.train_dataset = None
        self.val_dataset: Optional[CocoDetection] = None
        self._mosaic_enabled = True
        self._image_loader = image_loader

        # Log if using custom loader
        if image_loader is not None:
            logger.info(f"Using custom image loader: {type(image_loader).__name__}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training and validation."""
        root = Path(self.hparams.root)
        image_size = tuple(self.hparams.image_size)

        if stage == "fit" or stage is None:
            # Create base COCO dataset (without transforms - applied after mosaic)
            base_train_dataset = CocoDetectionWrapper(
                root=str(root / self.hparams.train_images),
                annFile=str(root / self.hparams.train_ann),
                transforms=None,  # Transforms applied after mosaic
                image_size=image_size,
                image_loader=self._image_loader,
            )

            # Create post-mosaic transforms (applied after multi-image augmentation)
            post_transforms = create_train_transforms(
                image_size=image_size,
                hsv_h=self.hparams.hsv_h,
                hsv_s=self.hparams.hsv_s,
                hsv_v=self.hparams.hsv_v,
                degrees=self.hparams.degrees,
                translate=self.hparams.translate,
                scale=self.hparams.scale,
                shear=self.hparams.shear,
                perspective=self.hparams.perspective,
                flip_lr=self.hparams.flip_lr,
                flip_ud=self.hparams.flip_ud,
            )

            # Wrap with MosaicMixupDataset for multi-image augmentation
            self.train_dataset = MosaicMixupDataset(
                dataset=base_train_dataset,
                image_size=image_size,
                mosaic_prob=self.hparams.mosaic_prob,
                mosaic_9_prob=self.hparams.mosaic_9_prob,
                mixup_prob=self.hparams.mixup_prob,
                mixup_alpha=self.hparams.mixup_alpha,
                cutmix_prob=self.hparams.cutmix_prob,
                transforms=post_transforms,
            )

        if stage == "fit" or stage == "validate" or stage is None:
            # Validation transforms (no augmentation)
            val_transforms = create_val_transforms(image_size=image_size)

            self.val_dataset = CocoDetectionWrapper(
                root=str(root / self.hparams.val_images),
                annFile=str(root / self.hparams.val_ann),
                transforms=val_transforms,
                image_size=image_size,
                image_loader=self._image_loader,
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self.hparams.pin_memory,
        )

    def on_train_epoch_start(self) -> None:
        """Disable mosaic augmentation for the final N epochs (close_mosaic)."""
        if self.hparams.close_mosaic_epochs <= 0:
            return

        epochs_remaining = self.trainer.max_epochs - self.trainer.current_epoch
        if epochs_remaining <= self.hparams.close_mosaic_epochs:
            if self._mosaic_enabled:
                self._mosaic_enabled = False
                if hasattr(self.train_dataset, "disable_mosaic"):
                    self.train_dataset.disable_mosaic()
                logger.info(
                    f"Disabling mosaic/mixup augmentation for final "
                    f"{self.hparams.close_mosaic_epochs} epochs"
                )

    @staticmethod
    def _collate_fn(batch):
        """
        Custom collate function for batching images with variable number of boxes.

        Args:
            batch: List of (image, target) tuples

        Returns:
            images: Batched images tensor [B, C, H, W]
            targets: List of target dicts with 'boxes' and 'labels'
        """
        images = []
        targets = []

        for image, target in batch:
            images.append(image)
            targets.append(target)

        images = torch.stack(images, dim=0)
        return images, targets


class CocoDetectionWrapper(CocoDetection):
    """
    Wrapper around CocoDetection that formats output for YOLO training.

    Converts COCO annotations to the format expected by YOLOModule:
        - boxes: [N, 4] in xyxy format
        - labels: [N] class indices

    Supports custom image loaders for special use cases (e.g., encrypted images).

    Args:
        root: Root directory containing images
        annFile: Path to COCO annotations JSON
        transforms: Optional transform to apply to images and targets
        image_size: Target image size (width, height)
        image_loader: Optional custom image loader (e.g., for encrypted images)
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transforms: Optional[Callable] = None,
        image_size: tuple = (640, 640),
        image_loader: Optional[ImageLoader] = None,
    ):
        super().__init__(root, annFile)
        self._transforms = transforms
        self.image_size = image_size
        self.target_transform = YOLOTargetTransform()
        # Use provided loader or default PIL loader
        self._image_loader = image_loader or DefaultImageLoader()

    def _load_image(self, id: int):
        """Override parent's _load_image to use custom loader."""
        path = self.coco.loadImgs(id)[0]["file_name"]
        full_path = os.path.join(self.root, path)
        return self._image_loader(full_path)

    def __getitem__(self, index: int):
        """Get image and target at index."""
        # Get image id
        id = self.ids[index]

        # Load image using custom loader
        image = self._load_image(id)

        # Get annotations
        target = self._load_target(id)

        # Convert COCO annotations to YOLO format
        target = self.target_transform(target, image.size)

        # Apply transforms
        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target

    def disable_mosaic(self):
        """Disable mosaic augmentation (called near end of training)."""
        # Update transforms to remove mosaic if present
        pass  # Mosaic is handled separately, this is a placeholder
