"""
YOLODataModule - PyTorch Lightning data module for COCO and YOLO format datasets.

Supports:
- COCO format: Uses torchvision.datasets.CocoDetection
- YOLO format: Uses custom YOLOFormatDataset for .txt label files

The format can be configured via YAML or CLI:
    data:
      format: coco  # or yolo
      ...
"""

import os
import platform
import random
from pathlib import Path

try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    # Windows doesn't have resource module
    HAS_RESOURCE = False
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from PIL import Image

from yolo.data.loaders import DefaultImageLoader, ImageLoader


def _get_safe_num_workers(requested: int) -> int:
    """
    Get a safe number of workers based on system file descriptor limits.

    Each worker uses ~10-15 file descriptors for shared memory, pipes, etc.
    We reserve some headroom for the main process and other system usage.

    Args:
        requested: Requested number of workers

    Returns:
        Safe number of workers that won't exhaust file descriptors
    """
    from yolo.utils.logger import logger

    if requested == 0:
        return 0

    if not HAS_RESOURCE:
        # Windows: can't check limits, use conservative default if high
        if requested > 16:
            logger.warning(
                f"⚠️ num_workers={requested} is high. "
                f"Reducing to 16 to avoid potential resource issues."
            )
            return 16
        return requested

    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):
        # Can't get limits, use conservative default
        if requested > 16:
            logger.warning(
                f"⚠️ Cannot determine file descriptor limit. "
                f"Reducing num_workers from {requested} to 16."
            )
            return 16
        return requested

    # Reserve ~500 FDs for main process, system, dataset files, etc.
    # Each worker needs ~15 FDs for shared memory, pipes, queues
    available = soft_limit - 500
    fd_per_worker = 15
    max_safe_workers = max(1, available // fd_per_worker)

    if requested > max_safe_workers:
        logger.warning(
            f"⚠️ Reducing num_workers: {requested} → {max_safe_workers} "
            f"(ulimit -n = {soft_limit}). "
            f"To use {requested} workers, run: ulimit -n {requested * fd_per_worker + 1000}"
        )
        return max_safe_workers

    return requested


def _worker_init_fn(worker_id: int) -> None:
    """
    Initialize each DataLoader worker with unique random seed.

    This ensures workers don't all generate the same random augmentations,
    which can cause synchronization issues and reduce augmentation diversity.
    """
    # Get the base seed from PyTorch
    worker_seed = torch.initial_seed() % 2**32

    # Set different seed for each worker
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


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
    Unified Lightning DataModule for YOLO training with COCO or YOLO format datasets.

    Supports both formats via the `format` parameter:
    - 'coco': Uses torchvision.datasets.CocoDetection (JSON annotations)
    - 'yolo': Uses YOLOFormatDataset (.txt label files)

    COCO format directory structure:
        data/coco/
        ├── train2017/
        │   └── *.jpg
        ├── val2017/
        │   └── *.jpg
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json

    YOLO format directory structure:
        dataset/
        ├── train/
        │   ├── images/
        │   │   └── *.jpg
        │   └── labels/
        │       └── *.txt
        └── valid/
            ├── images/
            │   └── *.jpg
            └── labels/
                └── *.txt

    Args:
        format: Dataset format - 'coco' or 'yolo' (default: 'coco')
        root: Root directory containing images and annotations/labels
        train_images: Subdirectory for training images
        val_images: Subdirectory for validation images
        train_ann: Path to training annotations JSON (COCO format, relative to root)
        val_ann: Path to validation annotations JSON (COCO format, relative to root)
        train_labels: Path to training labels directory (YOLO format, relative to root)
        val_labels: Path to validation labels directory (YOLO format, relative to root)
        batch_size: Batch size for training and validation
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker (default: 4)
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
        data_fraction: Fraction of data to use (default: 1.0). Uses stratified sampling
            to maintain class distribution. Useful for quick testing.
        cache_labels: Enable label caching (default: True)
        cache_images: Image caching mode - 'none', 'ram', or 'disk' (default: 'none')
        cache_resize_images: Resize images to image_size when caching (default: True, saves RAM)
        cache_max_memory_gb: Maximum RAM for image caching in GB (default: 8.0)
        cache_refresh: Force cache regeneration (default: False)
    """

    def __init__(
        self,
        # Dataset format
        format: Literal["coco", "yolo"] = "coco",
        # Dataset paths
        root: str = "data/coco",
        train_images: str = "train2017",
        val_images: str = "val2017",
        # COCO format paths
        train_ann: str = "annotations/instances_train2017.json",
        val_ann: str = "annotations/instances_val2017.json",
        # YOLO format paths
        train_labels: str = "train/labels",
        val_labels: str = "valid/labels",
        # DataLoader settings
        batch_size: int = 16,
        num_workers: int = 8,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
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
        # Data sampling
        data_fraction: float = 1.0,
        # Caching parameters
        cache_labels: bool = True,
        cache_images: Literal["none", "ram", "disk"] = "none",
        cache_resize_images: bool = True,
        cache_max_memory_gb: float = 8.0,
        cache_refresh: bool = False,
    ):
        super().__init__()
        # Exclude image_loader from hyperparameters (not serializable)
        self.save_hyperparameters(ignore=["image_loader"])

        self.train_dataset = None
        self.val_dataset = None
        self._mosaic_enabled = True
        self._image_loader = image_loader
        # Image size is set via CLI link from model.image_size
        self._image_size: Tuple[int, int] = (640, 640)

        # Validate format
        if format not in ("coco", "yolo"):
            raise ValueError(f"Invalid format '{format}'. Must be 'coco' or 'yolo'.")

        logger.info(f"Using dataset format: {format}")

        # Log if using custom loader
        if image_loader is not None:
            logger.info(f"Using custom image loader: {type(image_loader).__name__}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training and validation."""
        root = Path(self.hparams.root)
        image_size = self._image_size
        is_yolo_format = self.hparams.format == "yolo"

        # Setup image cache if enabled
        image_cache = None
        if self.hparams.cache_images != "none":
            from yolo.data.cache import ImageCache
            # Determine target size for caching (None = original size)
            target_size = image_size if self.hparams.cache_resize_images else None
            image_cache = ImageCache(
                mode=self.hparams.cache_images,
                cache_dir=root,
                max_memory_gb=self.hparams.cache_max_memory_gb,
                target_size=target_size,
            )

        # Get data fraction for sampling
        data_fraction = self.hparams.data_fraction
        if data_fraction < 1.0:
            logger.info(f"Using {data_fraction*100:.1f}% of data (stratified by class)")

        if stage == "fit" or stage is None:
            # Create base dataset based on format
            if is_yolo_format:
                base_train_dataset = YOLOFormatDataset(
                    images_dir=str(root / self.hparams.train_images),
                    labels_dir=str(root / self.hparams.train_labels),
                    transforms=None,  # Transforms applied after mosaic
                    image_size=image_size,
                    image_loader=self._image_loader,
                    cache_labels=self.hparams.cache_labels,
                    cache_refresh=self.hparams.cache_refresh,
                    image_cache=image_cache,
                    data_fraction=data_fraction,
                )
            else:
                base_train_dataset = CocoDetectionWrapper(
                    root=str(root / self.hparams.train_images),
                    annFile=str(root / self.hparams.train_ann),
                    transforms=None,  # Transforms applied after mosaic
                    image_size=image_size,
                    image_loader=self._image_loader,
                    image_cache=image_cache,
                    data_fraction=data_fraction,
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

            if is_yolo_format:
                self.val_dataset = YOLOFormatDataset(
                    images_dir=str(root / self.hparams.val_images),
                    labels_dir=str(root / self.hparams.val_labels),
                    transforms=val_transforms,
                    image_size=image_size,
                    image_loader=self._image_loader,
                    cache_labels=self.hparams.cache_labels,
                    cache_refresh=self.hparams.cache_refresh,
                    data_fraction=data_fraction,
                )
            else:
                self.val_dataset = CocoDetectionWrapper(
                    root=str(root / self.hparams.val_images),
                    annFile=str(root / self.hparams.val_ann),
                    transforms=val_transforms,
                    image_size=image_size,
                    image_loader=self._image_loader,
                    data_fraction=data_fraction,
                )

            # Extract class names from dataset
            self._extract_class_names(is_yolo_format)

    def _extract_class_names(self, is_yolo_format: bool) -> None:
        """Extract class names from dataset for metrics display."""
        from yolo.data.class_names import load_class_names

        data_format = "yolo" if is_yolo_format else "coco"
        ann_file = None if is_yolo_format else self.hparams.val_ann

        self.class_names = load_class_names(
            data_root=self.hparams.root,
            data_format=data_format,
            ann_file=ann_file,
        )

    def _get_dataloader_kwargs(self) -> Dict[str, Any]:
        """Get common DataLoader kwargs with optimized settings."""
        num_workers = _get_safe_num_workers(self.hparams.num_workers)
        kwargs = {
            "num_workers": num_workers,
            "collate_fn": self._collate_fn,
            "pin_memory": self.hparams.pin_memory,
            "persistent_workers": num_workers > 0,
            "prefetch_factor": self.hparams.prefetch_factor if num_workers > 0 else None,
            "worker_init_fn": _worker_init_fn if num_workers > 0 else None,
        }

        # Use 'spawn' multiprocessing context to avoid fork issues
        # Fork can cause file descriptor leaks, deadlocks with CUDA, and issues
        # with certain libraries (OpenCV, some crypto libs). Spawn is safer but
        # slightly slower at worker startup.
        if num_workers > 0:
            kwargs["multiprocessing_context"] = "spawn"

        return kwargs

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            **self._get_dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            **self._get_dataloader_kwargs(),
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
        image_cache: Optional ImageCache for caching decoded images in RAM/disk
        data_fraction: Fraction of data to use (default: 1.0). Uses stratified
            sampling by primary class to maintain class distribution.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transforms: Optional[Callable] = None,
        image_size: tuple = (640, 640),
        image_loader: Optional[ImageLoader] = None,
        image_cache: Optional[Any] = None,
        data_fraction: float = 1.0,
    ):
        super().__init__(root, annFile)
        self._transforms = transforms
        self.image_size = image_size
        self._image_cache = image_cache

        # Build category_id to 0-indexed class mapping from COCO categories
        # This handles both standard COCO (1-indexed) and custom datasets
        categories = self.coco.loadCats(self.coco.getCatIds())
        sorted_cats = sorted(categories, key=lambda x: x["id"])
        self._category_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(sorted_cats)}

        self.target_transform = YOLOTargetTransform(self._category_id_to_idx)
        # Use provided loader or default PIL loader
        self._image_loader = image_loader or DefaultImageLoader()

        # Apply stratified sampling if data_fraction < 1.0
        if data_fraction < 1.0:
            self._apply_stratified_sampling(data_fraction)

        # Pre-cache images if using RAM cache
        if self._image_cache is not None and self._image_cache.mode == "ram":
            self._precache_images()

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for pickling (spawn multiprocessing compatibility)."""
        state = self.__dict__.copy()
        # Store loader class and its pickle-safe state
        loader = state.pop("_image_loader", None)
        if loader is not None:
            state["_image_loader_class"] = type(loader)
            # Use loader's __getstate__ if available, otherwise use __dict__
            if hasattr(loader, "__getstate__"):
                state["_image_loader_state"] = loader.__getstate__()
            else:
                state["_image_loader_state"] = loader.__dict__.copy() if hasattr(loader, "__dict__") else {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after unpickling."""
        # Extract loader info before updating __dict__
        loader_class = state.pop("_image_loader_class", DefaultImageLoader)
        loader_state = state.pop("_image_loader_state", {})

        self.__dict__.update(state)

        # Recreate the image loader
        try:
            self._image_loader = loader_class.__new__(loader_class)
            # Use loader's __setstate__ if available
            if hasattr(self._image_loader, "__setstate__"):
                self._image_loader.__setstate__(loader_state)
            elif loader_state:
                self._image_loader.__dict__.update(loader_state)
        except Exception:
            # Fallback to default loader
            self._image_loader = DefaultImageLoader()

    def _apply_stratified_sampling(self, fraction: float) -> None:
        """
        Apply stratified sampling to reduce dataset size while preserving class distribution.

        For each image, the primary class is determined from annotations.
        Then, a fraction of images is sampled from each class proportionally.

        Args:
            fraction: Fraction of data to keep (0.0 to 1.0)
        """
        from collections import defaultdict

        if fraction >= 1.0:
            return

        # Group image ids by primary class
        class_to_ids: Dict[int, List[int]] = defaultdict(list)

        for img_id in self.ids:
            # Get annotations for this image
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            if anns:
                # Primary class: first annotation's category (mapped to 0-indexed)
                cat_id = anns[0]["category_id"]
                primary_class = self._category_id_to_idx.get(cat_id, -1)
            else:
                primary_class = -1  # Background/empty

            class_to_ids[primary_class].append(img_id)

        # Sample from each class
        sampled_ids = []
        for class_id, ids in sorted(class_to_ids.items()):
            n_samples = max(1, int(len(ids) * fraction))
            sampled = random.sample(ids, min(n_samples, len(ids)))
            sampled_ids.extend(sampled)

        # Sort to maintain some order consistency
        sampled_ids.sort()

        # Update ids list
        original_count = len(self.ids)
        self.ids = sampled_ids

        logger.info(
            f"Stratified sampling: {original_count} → {len(self.ids)} images "
            f"({len(class_to_ids)} classes, {fraction*100:.1f}%)"
        )

    def _precache_images(self) -> None:
        """Pre-load all images into RAM cache (optionally resized)."""
        from tqdm import tqdm

        logger.info(f"Pre-caching {len(self.ids)} images to RAM...")

        # Get all image paths for memory estimation
        image_paths = []
        for id in self.ids:
            path = self.coco.loadImgs(id)[0]["file_name"]
            image_paths.append(Path(os.path.join(self.root, path)))

        # Estimate memory (pass image_loader for encrypted images)
        estimated_gb = self._image_cache.estimate_memory(
            image_paths, image_loader=self._image_loader
        )
        if not self._image_cache.can_cache_in_ram(estimated_gb):
            logger.warning(
                f"⚠️ Estimated {estimated_gb:.1f}GB needed for image cache, "
                f"but max allowed is {self._image_cache.max_memory_gb:.1f}GB. "
                f"Disabling image caching."
            )
            self._image_cache = None
            return

        target_size = self._image_cache.target_size
        if target_size:
            logger.info(f"Estimated memory: {estimated_gb:.1f}GB (resized to {target_size[0]}x{target_size[1]})")
        else:
            logger.info(f"Estimated memory: {estimated_gb:.1f}GB (original size)")

        # Load all images
        for idx, id in enumerate(tqdm(self.ids, desc="Caching images")):
            path = self.coco.loadImgs(id)[0]["file_name"]
            full_path = Path(os.path.join(self.root, path))

            try:
                img = self._image_loader(str(full_path))
                # Resize if target_size is set (letterbox to preserve aspect ratio)
                if target_size is not None:
                    img = self._resize_for_cache(img, target_size)
                img_np = np.asarray(img).copy()
                self._image_cache.put(idx, full_path, img_np)
            except Exception as e:
                logger.warning(f"Failed to cache image {full_path}: {e}")

        logger.info(f"✅ Cached {self._image_cache.size} images to RAM")

    @staticmethod
    def _resize_for_cache(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image for caching using letterbox (preserves aspect ratio)."""
        target_w, target_h = target_size
        orig_w, orig_h = img.size

        # Calculate scale to fit within target while preserving aspect ratio
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize image
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Create letterbox (padded) image
        padded = Image.new("RGB", target_size, (114, 114, 114))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded.paste(img_resized, (paste_x, paste_y))

        return padded

    def _load_image(self, id: int, index: int):
        """Load image, using cache if available."""
        path = self.coco.loadImgs(id)[0]["file_name"]
        full_path = Path(os.path.join(self.root, path))

        # Try cache first
        if self._image_cache is not None:
            cached = self._image_cache.get(index, full_path)
            if cached is not None:
                return Image.fromarray(cached)

        # Load from disk
        img = self._image_loader(str(full_path))

        # Store in cache for next time
        if self._image_cache is not None:
            img_np = np.asarray(img).copy()
            self._image_cache.put(index, full_path, img_np)

        return img

    def __getitem__(self, index: int):
        """Get image and target at index."""
        # Get image id
        id = self.ids[index]

        # Load image using custom loader (with cache)
        image = self._load_image(id, index)

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


class YOLOFormatDataset(Dataset):
    """
    Dataset for YOLO format annotations (.txt files with normalized xywh coordinates).

    YOLO format: Each image has a corresponding .txt file with lines:
        class_id x_center y_center width height
    All coordinates are normalized (0-1).

    Expected directory structure:
        dataset/
        ├── images/
        │   └── *.jpg
        └── labels/
            └── *.txt

    Supports label caching to accelerate data loading on subsequent runs.
    Labels are cached in a .cache file with hash validation.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label .txt files
        transforms: Optional transform to apply
        image_size: Target image size (width, height)
        image_loader: Optional custom image loader
        cache_labels: Enable label caching (default: True)
        cache_refresh: Force cache regeneration (default: False)
        image_cache: Optional ImageCache for caching decoded images in RAM/disk
        data_fraction: Fraction of data to use (default: 1.0). Uses stratified
            sampling by primary class to maintain class distribution.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transforms: Optional[Callable] = None,
        image_size: tuple = (640, 640),
        image_loader: Optional[ImageLoader] = None,
        cache_labels: bool = True,
        cache_refresh: bool = False,
        image_cache: Optional[Any] = None,
        data_fraction: float = 1.0,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self._transforms = transforms
        self.image_size = image_size
        self._image_loader = image_loader or DefaultImageLoader()
        self._labels_cache: Optional[List[Dict[str, Any]]] = None
        self._image_cache = image_cache

        # Find all images
        self.image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]:
            self.image_files.extend(self.images_dir.glob(ext))
        self.image_files = sorted(self.image_files)

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")

        logger.info(f"Found {len(self.image_files)} images in {images_dir}")

        # Setup label caching (needed before stratified sampling)
        if cache_labels:
            self._setup_cache(refresh=cache_refresh)

        # Apply stratified sampling if data_fraction < 1.0
        if data_fraction < 1.0:
            self._apply_stratified_sampling(data_fraction)

        # Pre-cache images if using RAM cache
        if self._image_cache is not None and self._image_cache.mode == "ram":
            self._precache_images()

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for pickling (spawn multiprocessing compatibility)."""
        state = self.__dict__.copy()
        # Store loader class and its pickle-safe state
        loader = state.pop("_image_loader", None)
        if loader is not None:
            state["_image_loader_class"] = type(loader)
            # Use loader's __getstate__ if available, otherwise use __dict__
            if hasattr(loader, "__getstate__"):
                state["_image_loader_state"] = loader.__getstate__()
            else:
                state["_image_loader_state"] = loader.__dict__.copy() if hasattr(loader, "__dict__") else {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after unpickling."""
        # Extract loader info before updating __dict__
        loader_class = state.pop("_image_loader_class", DefaultImageLoader)
        loader_state = state.pop("_image_loader_state", {})

        self.__dict__.update(state)

        # Recreate the image loader
        try:
            self._image_loader = loader_class.__new__(loader_class)
            # Use loader's __setstate__ if available
            if hasattr(self._image_loader, "__setstate__"):
                self._image_loader.__setstate__(loader_state)
            elif loader_state:
                self._image_loader.__dict__.update(loader_state)
        except Exception:
            # Fallback to default loader
            self._image_loader = DefaultImageLoader()

    def _precache_images(self) -> None:
        """Pre-load all images into RAM cache (optionally resized)."""
        from tqdm import tqdm

        logger.info(f"Pre-caching {len(self.image_files)} images to RAM...")

        # Estimate memory (pass image_loader for encrypted images)
        estimated_gb = self._image_cache.estimate_memory(
            self.image_files, image_loader=self._image_loader
        )
        if not self._image_cache.can_cache_in_ram(estimated_gb):
            logger.warning(
                f"⚠️ Estimated {estimated_gb:.1f}GB needed for image cache, "
                f"but max allowed is {self._image_cache.max_memory_gb:.1f}GB. "
                f"Disabling image caching."
            )
            self._image_cache = None
            return

        target_size = self._image_cache.target_size
        if target_size:
            logger.info(f"Estimated memory: {estimated_gb:.1f}GB (resized to {target_size[0]}x{target_size[1]})")
        else:
            logger.info(f"Estimated memory: {estimated_gb:.1f}GB (original size)")

        # Load all images
        for idx, image_path in enumerate(tqdm(self.image_files, desc="Caching images")):
            try:
                img = self._image_loader(str(image_path))
                # Resize if target_size is set (letterbox to preserve aspect ratio)
                if target_size is not None:
                    img = self._resize_for_cache(img, target_size)
                img_np = np.asarray(img).copy()
                self._image_cache.put(idx, image_path, img_np)
            except Exception as e:
                logger.warning(f"Failed to cache image {image_path}: {e}")

        logger.info(f"✅ Cached {self._image_cache.size} images to RAM")

    @staticmethod
    def _resize_for_cache(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image for caching using letterbox (preserves aspect ratio)."""
        target_w, target_h = target_size
        orig_w, orig_h = img.size

        # Calculate scale to fit within target while preserving aspect ratio
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize image
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Create letterbox (padded) image
        padded = Image.new("RGB", target_size, (114, 114, 114))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded.paste(img_resized, (paste_x, paste_y))

        return padded

    def __len__(self) -> int:
        return len(self.image_files)

    def _apply_stratified_sampling(self, fraction: float) -> None:
        """
        Apply stratified sampling to reduce dataset size while preserving class distribution.

        For each image, the primary class is determined (first label or most frequent).
        Then, a fraction of images is sampled from each class proportionally.

        Args:
            fraction: Fraction of data to keep (0.0 to 1.0)
        """
        from collections import defaultdict

        if fraction >= 1.0:
            return

        # Group images by primary class
        class_to_indices: Dict[int, List[int]] = defaultdict(list)

        for idx, image_path in enumerate(self.image_files):
            # Get labels for this image
            if self._labels_cache is not None:
                labels = self._labels_cache[idx].get("labels", [])
            else:
                parsed = self._parse_label_file(image_path)
                labels = parsed.get("labels", [])

            # Primary class: first label, or -1 for background/empty
            primary_class = labels[0] if labels else -1
            class_to_indices[primary_class].append(idx)

        # Sample from each class
        sampled_indices = []
        for class_id, indices in sorted(class_to_indices.items()):
            n_samples = max(1, int(len(indices) * fraction))
            # Random sample without replacement
            sampled = random.sample(indices, min(n_samples, len(indices)))
            sampled_indices.extend(sampled)

        # Sort to maintain some order consistency
        sampled_indices.sort()

        # Update image_files and labels_cache
        original_count = len(self.image_files)
        self.image_files = [self.image_files[i] for i in sampled_indices]

        if self._labels_cache is not None:
            self._labels_cache = [self._labels_cache[i] for i in sampled_indices]

        logger.info(
            f"Stratified sampling: {original_count} → {len(self.image_files)} images "
            f"({len(class_to_indices)} classes, {fraction*100:.1f}%)"
        )

    def _setup_cache(self, refresh: bool = False) -> None:
        """
        Setup label caching.

        Loads labels from cache if valid, otherwise parses all label files
        and saves to cache.

        Args:
            refresh: Force cache regeneration even if valid cache exists.
        """
        from yolo.data.cache import DatasetCache

        cache = DatasetCache(self.labels_dir.parent, self.labels_dir.name)

        # Get all label files for hash computation
        label_files = list(self.labels_dir.glob("*.txt"))

        # Force refresh: delete existing cache
        if refresh:
            cache.delete()

        if cache.is_valid(label_files):
            logger.info(f"Loading cached labels from {cache.cache_path}")
            self._labels_cache = cache.load()["labels"]
            logger.info(f"Loaded {len(self._labels_cache)} cached labels")
        else:
            logger.info(f"Parsing {len(self.image_files)} labels (first run or files changed)...")
            self._labels_cache = []

            for image_path in self.image_files:
                label_data = self._parse_label_file(image_path)
                self._labels_cache.append(label_data)

            # Save cache
            stats = {
                "count": len(self._labels_cache),
                "total_boxes": sum(len(l["boxes_norm"]) for l in self._labels_cache),
            }
            cache.save(self._labels_cache, label_files, stats)

    def _parse_label_file(self, image_path: Path) -> Dict[str, Any]:
        """
        Parse a single YOLO format label file.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary with normalized box coordinates and class labels.
        """
        label_path = self.labels_dir / (image_path.stem + ".txt")

        boxes_norm = []  # Normalized xywh format
        labels = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Validate normalized coordinates
                        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and width > 0 and height > 0:
                            boxes_norm.append([x_center, y_center, width, height])
                            labels.append(class_id)

        return {"boxes_norm": boxes_norm, "labels": labels}

    def _load_image(self, index: int) -> Image.Image:
        """Load image, using cache if available."""
        image_path = self.image_files[index]

        # Try cache first
        if self._image_cache is not None:
            cached = self._image_cache.get(index, image_path)
            if cached is not None:
                return Image.fromarray(cached)

        # Load from disk
        img = self._image_loader(str(image_path))

        # Store in cache for next time
        if self._image_cache is not None:
            img_np = np.asarray(img).copy()
            self._image_cache.put(index, image_path, img_np)

        return img

    def __getitem__(self, index: int):
        """Get image and target at index."""
        # Load image (with cache if enabled)
        image = self._load_image(index)
        img_w, img_h = image.size

        # Get labels (from cache or parse on-the-fly)
        if self._labels_cache is not None:
            cached = self._labels_cache[index]
            boxes_norm = cached["boxes_norm"]
            label_ids = cached["labels"]
        else:
            # Parse on-the-fly (no caching)
            parsed = self._parse_label_file(image_path)
            boxes_norm = parsed["boxes_norm"]
            label_ids = parsed["labels"]

        # Convert normalized xywh to xyxy pixel coordinates
        boxes = []
        labels = []

        for (x_center, y_center, width, height), class_id in zip(boxes_norm, label_ids):
            # Convert from normalized xywh to xyxy (pixel coordinates)
            x1 = (x_center - width / 2) * img_w
            y1 = (y_center - height / 2) * img_h
            x2 = (x_center + width / 2) * img_w
            y2 = (y_center + height / 2) * img_h

            # Clip to image bounds
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            # Only add valid boxes
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

        # Create target dict
        if boxes:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
            }

        # Apply transforms
        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target
