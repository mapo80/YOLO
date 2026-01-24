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

    IMPORTANT: image_size is automatically linked from model.image_size via the CLI.
    Do NOT specify data.image_size manually - it will be set from model.image_size.
    If you need to change the image size, modify model.image_size instead.

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
        cache_workers: Number of parallel workers for caching (default: None = all CPU threads)
        cache_refresh: Force cache regeneration (default: False)
        cache_only: Load images only from cache, without requiring original images on disk.
            Useful for training on remote machines where only the encrypted cache is available.
            Requires a complete cache created by cache-create command.
        image_size: DO NOT SPECIFY - automatically linked from model.image_size via CLI.
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
        cache_dir: Optional[str] = None,  # None = same as images directory
        cache_resize_images: bool = True,
        cache_max_memory_gb: float = 8.0,
        cache_workers: Optional[int] = None,  # None = auto (all CPU threads)
        cache_refresh: bool = False,
        cache_encrypt: bool = False,
        cache_only: bool = False,
        cache_sync: bool = False,  # Enable LMDB fsync for crash safety (disable for external volumes on macOS)
        # Encryption key for encrypted images (.enc) and/or encrypted cache
        # Can also be set via YOLO_ENCRYPTION_KEY environment variable
        encryption_key: Optional[str] = None,
        # Image size - automatically linked from model.image_size via CLI.
        # DO NOT specify this manually - it is set automatically from model.image_size.
        image_size: Tuple[int, int] = (640, 640),
    ):
        super().__init__()
        # Exclude image_loader and encryption_key from hyperparameters (not serializable)
        self.save_hyperparameters(ignore=["image_loader", "encryption_key"])

        self.train_dataset = None
        self.val_dataset = None
        self._mosaic_enabled = True
        self._image_loader = image_loader
        # Encryption key: prefer parameter, fallback to environment variable
        self._encryption_key = encryption_key or os.environ.get("YOLO_ENCRYPTION_KEY")
        # Image size: linked from model.image_size via CLI (apply_on="instantiate")
        self._image_size: Tuple[int, int] = tuple(image_size)

        # Validate format
        if format not in ("coco", "yolo"):
            raise ValueError(f"Invalid format '{format}'. Must be 'coco' or 'yolo'.")

        logger.info(f"Using dataset format: {format}")

        # Log if using custom loader
        if image_loader is not None:
            logger.info(f"Using custom image loader: {type(image_loader).__name__}")

        # Log encryption status
        if encryption_key is not None:
            logger.info("🔐 Encryption key configured")

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training and validation."""
        root = Path(self.hparams.root)
        image_size = self._image_size
        is_yolo_format = self.hparams.format == "yolo"

        # Setup image cache if enabled
        image_cache = None
        if self.hparams.cache_images != "none":
            from yolo.data.cache import ImageCache

            # Determine encryption key for cache (only if cache_encrypt is True)
            cache_encryption_key = None
            if self.hparams.cache_encrypt:
                if self._encryption_key is None:
                    raise ValueError(
                        "\n"
                        "╔══════════════════════════════════════════════════════════════════════╗\n"
                        "║  ENCRYPTION KEY REQUIRED                                             ║\n"
                        "╠══════════════════════════════════════════════════════════════════════╣\n"
                        "║  cache_encrypt=True but no encryption key found.                     ║\n"
                        "║                                                                      ║\n"
                        "║  Set the key using ONE of these methods:                             ║\n"
                        "║                                                                      ║\n"
                        "║  1. Environment variable (recommended):                              ║\n"
                        "║     export YOLO_ENCRYPTION_KEY='your-64-char-hex-key'                ║\n"
                        "║                                                                      ║\n"
                        "║  2. YAML configuration:                                              ║\n"
                        "║     data:                                                            ║\n"
                        "║       encryption_key: 'your-64-char-hex-key'                         ║\n"
                        "║                                                                      ║\n"
                        "║  Generate a new key with:                                            ║\n"
                        "║     python -c \"import os; print(os.urandom(32).hex())\"               ║\n"
                        "║                                                                      ║\n"
                        "║  Or disable encryption:                                              ║\n"
                        "║     data:                                                            ║\n"
                        "║       cache_encrypt: false                                           ║\n"
                        "╚══════════════════════════════════════════════════════════════════════╝"
                    )
                cache_encryption_key = self._encryption_key
                logger.info(f"🔒 {self.hparams.cache_images.upper()} cache encryption enabled")

            # Determine target size for caching (None = original size)
            target_size = image_size if self.hparams.cache_resize_images else None

            # Determine cache directory (custom or default to dataset root)
            cache_directory = Path(self.hparams.cache_dir) if self.hparams.cache_dir else root

            # Build cache suffix to differentiate caches with different settings
            # Format: "{width}x{height}_f{fraction}" e.g., "640x640_f1.0" or "640x640_f0.1"
            data_fraction = self.hparams.data_fraction
            if target_size:
                size_str = f"{target_size[0]}x{target_size[1]}"
            else:
                size_str = "orig"
            cache_suffix = f"{size_str}_f{data_fraction}"

            image_cache = ImageCache(
                mode=self.hparams.cache_images,
                cache_dir=cache_directory,
                max_memory_gb=self.hparams.cache_max_memory_gb,
                target_size=target_size,
                encryption_key=cache_encryption_key,
                cache_suffix=cache_suffix,
                refresh=self.hparams.cache_refresh,
                sync=self.hparams.cache_sync,
            )

        # Get data fraction for sampling
        data_fraction = self.hparams.data_fraction
        if data_fraction < 1.0:
            logger.info(f"Using {data_fraction*100:.1f}% of data (stratified by class)")

        # Cache-only mode check
        cache_only = self.hparams.cache_only
        if cache_only and image_cache is None:
            raise ValueError(
                "cache_only=True requires cache_images to be 'ram' or 'disk'. "
                "Set --data.cache_images=disk --data.cache_only=true"
            )

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
                    cache_workers=self.hparams.cache_workers,
                    cache_only=cache_only,
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
                    cache_workers=self.hparams.cache_workers,
                    cache_only=cache_only,
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

        # Pre-warm DataLoader workers (only for fit stage with num_workers > 0)
        if stage == "fit" or stage is None:
            self._prewarm_dataloader_workers()

    def _prewarm_dataloader_workers(self) -> None:
        """Pre-warm DataLoader workers to avoid delay at first training step.

        With 'spawn' multiprocessing, worker creation is slow. By fetching one batch
        here, we force worker initialization during setup() instead of during training.
        """
        from yolo.utils.progress import spinner, console

        num_workers = _get_safe_num_workers(self.hparams.num_workers)
        if num_workers == 0:
            return

        with spinner(f"Initializing {num_workers * 2} DataLoader workers (train + val)..."):
            # Pre-warm TRAIN dataloader
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=False,  # No shuffle for faster first batch
                drop_last=True,
                **self._get_dataloader_kwargs(),
            )
            # Fetch first batch to initialize workers
            _ = next(iter(train_loader))
            # Store for reuse
            self._train_dataloader = train_loader

            # Pre-warm VALIDATION dataloader (needed for sanity check)
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                **self._get_dataloader_kwargs(),
            )
            # Fetch first batch to initialize workers
            _ = next(iter(val_loader))
            # Store for reuse
            self._val_dataloader = val_loader

        console.print(f"[green]✓[/green] DataLoader workers initialized ({num_workers} train + {num_workers} val)")

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
        # Reuse pre-warmed loader if available (workers already initialized)
        if hasattr(self, "_train_dataloader") and self._train_dataloader is not None:
            loader = self._train_dataloader
            self._train_dataloader = None  # Only reuse once, then create new
            return loader

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            **self._get_dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        # Reuse pre-warmed loader if available (workers already initialized)
        if hasattr(self, "_val_dataloader") and self._val_dataloader is not None:
            loader = self._val_dataloader
            self._val_dataloader = None  # Only reuse once, then create new
            return loader

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
        cache_workers: Number of parallel workers for caching (None = all CPU threads)
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
        cache_workers: Optional[int] = None,
        cache_only: bool = False,
    ):
        self._cache_only = cache_only
        self._image_cache = image_cache
        self._cache_workers = cache_workers
        self._transforms = transforms
        self.image_size = image_size
        self.root = root  # Store root for cache-only mode

        # In cache-only mode, load image paths and annotations from cache metadata
        if cache_only:
            if image_cache is None:
                raise ValueError("cache_only=True requires image_cache to be provided")
            self._setup_from_cache()
            return

        # Normal mode: initialize from COCO annotations
        super().__init__(root, annFile)

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

        # Pre-cache images if using RAM or disk cache
        if self._image_cache is not None and self._image_cache.mode in ("ram", "disk"):
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
        """Pre-load all images into cache (RAM or disk, parallelized)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from yolo.utils.progress import progress_bar, spinner, console

        cache_mode = self._image_cache.mode

        # Get all image paths (fast - just string lookups)
        image_paths = [
            Path(os.path.join(self.root, self.coco.loadImgs(id)[0]["file_name"]))
            for id in self.ids
        ]

        # Initialize unified LMDB cache with spinner
        dataset_root = Path(self.root)
        with spinner(f"Validating {cache_mode.upper()} cache ({len(image_paths):,} images)..."):
            cache_exists = self._image_cache.initialize(
                num_images=len(image_paths),
                cache_dir=dataset_root,
                paths=image_paths,
            )
        cache_location = self._image_cache.cache_path or "alongside images"

        if not self._image_cache._enabled:
            self._image_cache = None
            return

        if cache_exists:
            console.print(f"[green]✓[/green] {cache_mode.upper()} cache found: {len(image_paths):,} images ready ({cache_location})")
            return

        # Show why cache is being rebuilt
        if getattr(self._image_cache, '_refresh_requested', False):
            console.print(f"[yellow]🔄[/yellow] Cache refresh requested - rebuilding {cache_mode.upper()} cache")
        elif getattr(self._image_cache, '_invalidation_reason', None):
            reason = self._image_cache._invalidation_reason
            console.print(f"[yellow]⚠[/yellow] Cache invalidated: {reason} - rebuilding")

        # All images need to be cached
        work_items = [(idx, path) for idx, path in enumerate(image_paths)]
        cache_desc = f"Creating {cache_mode} cache ({len(image_paths)} images)"

        # Determine size info for display
        target_size = self._image_cache.target_size
        if target_size:
            w, h = target_size
            size_info = f"resized to {w}x{h}"
        else:
            first_img = self._image_loader(str(image_paths[0]))
            w, h = first_img.size
            size_info = f"original size {w}x{h}"

        # Determine number of workers for parallel loading
        num_workers = self._cache_workers if self._cache_workers is not None else (os.cpu_count() or 4)

        # Display cache info
        enc_info = ", encrypted" if self._image_cache._encrypt_cache else ""
        console.print(f"\n[bold]💾 Pre-caching {len(work_items):,} images[/bold] ({size_info}{enc_info}, {num_workers} workers)")
        console.print(f"   Cache location: {cache_location}")

        # Use ThreadPoolExecutor for parallel I/O
        failed_count = 0
        executor = ThreadPoolExecutor(max_workers=num_workers)

        # Submit all tasks with spinner (can take a moment for large datasets)
        with spinner(f"Queueing {len(work_items):,} images..."):
            futures = {
                executor.submit(self._load_and_cache_image, idx, path, target_size): idx
                for idx, path in work_items
            }

        # Process results with progress bar
        with progress_bar(len(work_items), cache_desc) as update:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 3:
                        logger.warning(f"Failed to cache image: {e}")
                update(1)

        executor.shutdown(wait=False)

        # Finalize cache
        self._image_cache.finalize()
        cached_count = self._image_cache.size
        console.print(f"[green]✓[/green] Cached {cached_count} images to {cache_mode.upper()} (LMDB)")

        if failed_count > 0:
            console.print(f"[yellow]⚠[/yellow] {failed_count} images failed to cache")

    def _load_and_cache_image(self, idx: int, path: Path, target_size: Optional[Tuple[int, int]]) -> None:
        """Load a single image and store in cache (called by thread workers)."""
        img = self._image_loader(str(path))
        orig_size = img.size if target_size is not None else None  # Save original size only if resizing
        if target_size is not None:
            img = self._resize_for_cache(img, target_size)
        img_np = np.asarray(img).copy()
        self._image_cache.put(idx, img_np, orig_size=orig_size)

    def _setup_from_cache(self) -> None:
        """
        Setup dataset from cache metadata (cache-only mode).

        In cache-only mode, image paths and annotations are loaded from the cache
        metadata instead of scanning the filesystem. This enables training
        on machines where only the cache is available, not the original images.

        Raises:
            ValueError: If cache metadata doesn't contain required information.
        """
        from yolo.utils.progress import spinner, console

        # Initialize cache in read-only mode
        dataset_root = Path(self.root)

        with spinner("Loading cache metadata (COCO format)..."):
            # Try to find existing cache
            cache_dir = self._image_cache._build_cache_dir(dataset_root)
            if not cache_dir.exists():
                raise ValueError(
                    f"Cache directory not found: {cache_dir}. "
                    f"Create the cache first with: yolo cache-create"
                )

            # Open cache in read-only mode
            self._image_cache._cache_dir_path = cache_dir
            self._image_cache._db_path = cache_dir / "cache.lmdb"
            self._image_cache._open_db(readonly=True)
            self._image_cache._load_cached_indices()

        # Get metadata
        metadata = self._image_cache.get_metadata()
        if metadata is None:
            raise ValueError(
                "\n"
                "╔══════════════════════════════════════════════════════════════════════╗\n"
                "║  CACHE METADATA NOT FOUND                                            ║\n"
                "╠══════════════════════════════════════════════════════════════════════╣\n"
                "║  Could not read cache metadata. Possible causes:                     ║\n"
                "║                                                                      ║\n"
                "║  1. ENCRYPTED CACHE with cache_encrypt: false                        ║\n"
                "║     If the cache was created with encryption, you must set:          ║\n"
                "║       data:                                                          ║\n"
                "║         cache_encrypt: true                                          ║\n"
                "║     AND provide the encryption key via:                              ║\n"
                "║       export YOLO_ENCRYPTION_KEY='your-64-char-hex-key'              ║\n"
                "║                                                                      ║\n"
                "║  2. CORRUPTED CACHE                                                  ║\n"
                "║     Delete the cache directory and recreate it                       ║\n"
                "║                                                                      ║\n"
                "║  3. OLD CACHE FORMAT (pre cache-only support)                        ║\n"
                "║     Recreate the cache with: yolo cache-create                       ║\n"
                "╚══════════════════════════════════════════════════════════════════════╝"
            )

        # Verify format matches
        cache_format = metadata.get("format")
        if cache_format is not None and cache_format != "coco":
            raise ValueError(
                "\n"
                "╔══════════════════════════════════════════════════════════════════════╗\n"
                "║  FORMAT MISMATCH                                                     ║\n"
                "╠══════════════════════════════════════════════════════════════════════╣\n"
                f"║  YAML config:  format: coco                                          ║\n"
                f"║  Cache format: {cache_format:<54}║\n"
                "║                                                                      ║\n"
                "║  The cache was created with a different format than your config.     ║\n"
                "║                                                                      ║\n"
                "║  Solutions:                                                          ║\n"
                f"║  1. Change YAML to: format: {cache_format:<42}║\n"
                "║  2. Recreate cache with: yolo cache-create --data.format coco        ║\n"
                "╚══════════════════════════════════════════════════════════════════════╝"
            )

        # Get image paths from metadata
        image_paths = metadata.get("image_paths")
        if image_paths is None:
            raise ValueError(
                "Cache does not contain image paths. "
                "This cache was created before cache-only mode was supported. "
                "Recreate the cache with: yolo cache-create"
            )

        # In cache-only mode for COCO format, we need annotations from metadata
        # Create a minimal COCO-like structure from cached data
        coco_annotations = metadata.get("coco_annotations")
        if coco_annotations is None:
            raise ValueError(
                "\n"
                "╔══════════════════════════════════════════════════════════════════════╗\n"
                "║  CACHE MISSING COCO ANNOTATIONS                                      ║\n"
                "╠══════════════════════════════════════════════════════════════════════╣\n"
                "║  The cache does not contain COCO annotations required for            ║\n"
                "║  cache-only mode with format: coco                                   ║\n"
                "║                                                                      ║\n"
                "║  Possible causes:                                                    ║\n"
                "║  1. Cache created with format: yolo (change YAML to format: yolo)    ║\n"
                "║  2. Cache created before cache-only mode was supported               ║\n"
                "║                                                                      ║\n"
                "║  Solution: Recreate cache with:                                      ║\n"
                "║    yolo cache-create --data.format coco ...                          ║\n"
                "╚══════════════════════════════════════════════════════════════════════╝"
            )

        # Build the ids list (image indices)
        self.ids = list(range(len(image_paths)))
        self._image_paths = [Path(p) for p in image_paths]
        self._cached_annotations = coco_annotations.get("annotations", {})
        self._cached_categories = coco_annotations.get("categories", [])

        # Build category_id to 0-indexed class mapping
        sorted_cats = sorted(self._cached_categories, key=lambda x: x["id"])
        self._category_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(sorted_cats)}
        self.target_transform = YOLOTargetTransform(self._category_id_to_idx)

        # Use default image loader (not used in cache-only mode, but needed for interface)
        self._image_loader = DefaultImageLoader()

        num_images = len(self.ids)
        cached_images = len(self._image_cache._cached_indices)
        encrypted = metadata.get("encrypted", False)

        # Verify encryption setting matches
        if encrypted and not self._image_cache._encrypt_cache:
            raise ValueError(
                "\n"
                "╔══════════════════════════════════════════════════════════════════════╗\n"
                "║  ENCRYPTION MISMATCH                                                 ║\n"
                "╠══════════════════════════════════════════════════════════════════════╣\n"
                "║  The cache is ENCRYPTED but cache_encrypt: false in your config.    ║\n"
                "║                                                                      ║\n"
                "║  To use this cache, you must:                                        ║\n"
                "║                                                                      ║\n"
                "║  1. Set in YAML config:                                              ║\n"
                "║       data:                                                          ║\n"
                "║         cache_encrypt: true                                          ║\n"
                "║                                                                      ║\n"
                "║  2. Provide the encryption key used during cache creation:           ║\n"
                "║       export YOLO_ENCRYPTION_KEY='your-64-char-hex-key'              ║\n"
                "║                                                                      ║\n"
                "║  Without the correct key, encrypted images cannot be read.          ║\n"
                "╚══════════════════════════════════════════════════════════════════════╝"
            )

        console.print(f"[green]✓[/green] Cache-only mode (COCO): {num_images:,} images from cache")
        if encrypted:
            console.print(f"   [dim]Encrypted cache - decryption in memory only[/dim]")

        # Verify cache is complete
        if cached_images < num_images:
            logger.warning(
                f"Cache is incomplete: {cached_images:,}/{num_images:,} images cached. "
                f"Some images may fail to load."
            )

        self._image_cache._initialized = True

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.ids)

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

    def _load_image(self, id: int, index: int) -> Tuple[Image.Image, Optional[Tuple[int, int]]]:
        """
        Load image, using cache if available.

        Returns:
            Tuple of (image, orig_size) where orig_size is the original image size
            before cache resize (width, height), or None if not resized.

        Raises:
            RuntimeError: If cache_only=True and image is not in cache.
        """
        # Get image path - handle cache-only mode
        if self._cache_only:
            image_path = self._image_paths[index]
        else:
            path = self.coco.loadImgs(id)[0]["file_name"]
            image_path = Path(os.path.join(self.root, path))

        # Try cache first
        if self._image_cache is not None:
            cached = self._image_cache.get(index)
            if cached is not None:
                img_arr, orig_size = cached
                return Image.fromarray(img_arr), orig_size

        # In cache-only mode, raise error if image not in cache
        if self._cache_only:
            raise RuntimeError(
                f"Cache-only mode: Image not found in cache (index={index}, path={image_path}). "
                f"The cache may be incomplete. Recreate with: yolo cache-create"
            )

        # Load from disk
        img = self._image_loader(str(image_path))

        # Store in cache for next time (no resize when loading from disk directly)
        if self._image_cache is not None:
            img_np = np.asarray(img).copy()
            self._image_cache.put(index, img_np, orig_size=None)

        return img, None  # No resize info when loaded from disk

    def __getitem__(self, index: int):
        """Get image and target at index."""
        # Get image id
        id = self.ids[index]

        # Load image using custom loader (with cache)
        image, orig_size = self._load_image(id, index)

        # Get annotations - handle cache-only mode
        if self._cache_only:
            # In cache-only mode, load from cached annotations
            target = self._cached_annotations.get(str(index), [])
        else:
            target = self._load_target(id)

        # Convert COCO annotations to YOLO format
        # Use orig_size for bbox transform if image was resized in cache
        bbox_reference_size = orig_size if orig_size is not None else image.size
        target = self.target_transform(target, bbox_reference_size)

        # If image was resized in cache, transform bboxes to match the resized image
        if orig_size is not None:
            target = self._transform_bboxes_for_letterbox(target, orig_size, image.size)

        # Apply transforms
        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target

    def _transform_bboxes_for_letterbox(
        self, target: Dict[str, Any], orig_size: Tuple[int, int], target_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Transform bboxes from original image coordinates to letterboxed image coordinates.

        Args:
            target: Target dict with 'boxes' in xyxy pixel format (from YOLOTargetTransform)
            orig_size: Original image size (width, height)
            target_size: Letterboxed image size (width, height)

        Returns:
            Target dict with transformed bboxes
        """
        if "boxes" not in target or len(target["boxes"]) == 0:
            return target

        orig_w, orig_h = orig_size
        target_w, target_h = target_size

        # Calculate letterbox parameters (same as _resize_for_cache)
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        # Transform bboxes: xyxy pixel coords in original -> xyxy pixel coords in letterboxed
        boxes = target["boxes"].clone()

        # Scale all coordinates
        boxes[:, 0] = boxes[:, 0] * scale + pad_x  # x1
        boxes[:, 1] = boxes[:, 1] * scale + pad_y  # y1
        boxes[:, 2] = boxes[:, 2] * scale + pad_x  # x2
        boxes[:, 3] = boxes[:, 3] * scale + pad_y  # y2

        target["boxes"] = boxes
        return target

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
        cache_workers: Number of parallel workers for caching (None = all CPU threads)
        cache_only: Load images only from cache, without requiring original files.
            Requires a complete cache with stored image paths.
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
        cache_workers: Optional[int] = None,
        cache_only: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self._transforms = transforms
        self.image_size = image_size
        self._image_loader = image_loader or DefaultImageLoader()
        self._labels_cache: Optional[List[Dict[str, Any]]] = None
        self._image_cache = image_cache
        self._cache_workers = cache_workers
        self._cache_only = cache_only

        # In cache-only mode, load image paths from cache metadata
        if cache_only:
            if image_cache is None:
                raise ValueError("cache_only=True requires image_cache to be provided")
            self._setup_from_cache()
            return

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

        # Pre-cache images if using RAM or disk cache
        if self._image_cache is not None and self._image_cache.mode in ("ram", "disk"):
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
        """Pre-load all images into cache (RAM or disk, parallelized)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from yolo.utils.progress import progress_bar, spinner, console

        cache_mode = self._image_cache.mode

        # Initialize unified LMDB cache with spinner
        dataset_root = self.images_dir.parent
        with spinner(f"Validating {cache_mode.upper()} cache ({len(self.image_files):,} images)..."):
            cache_exists = self._image_cache.initialize(
                num_images=len(self.image_files),
                cache_dir=dataset_root,
                paths=self.image_files,
            )
        cache_location = self._image_cache.cache_path or "alongside images"

        if not self._image_cache._enabled:
            self._image_cache = None
            return

        if cache_exists:
            console.print(f"[green]✓[/green] {cache_mode.upper()} cache found: {len(self.image_files):,} images ready ({cache_location})")
            return

        # Show why cache is being rebuilt
        if getattr(self._image_cache, '_refresh_requested', False):
            console.print(f"[yellow]🔄[/yellow] Cache refresh requested - rebuilding {cache_mode.upper()} cache")
        elif getattr(self._image_cache, '_invalidation_reason', None):
            reason = self._image_cache._invalidation_reason
            console.print(f"[yellow]⚠[/yellow] Cache invalidated: {reason} - rebuilding")

        # All images need to be cached
        work_items = [(idx, path) for idx, path in enumerate(self.image_files)]
        cache_desc = f"Creating {cache_mode} cache ({len(self.image_files)} images)"

        # Determine size info for display
        target_size = self._image_cache.target_size
        if target_size:
            w, h = target_size
            size_info = f"resized to {w}x{h}"
        else:
            first_img = self._image_loader(str(self.image_files[0]))
            w, h = first_img.size
            size_info = f"original size {w}x{h}"

        # Determine number of workers for parallel loading
        num_workers = self._cache_workers if self._cache_workers is not None else (os.cpu_count() or 4)

        # Display cache info
        enc_info = ", encrypted" if self._image_cache._encrypt_cache else ""
        console.print(f"\n[bold]💾 Pre-caching {len(work_items):,} images[/bold] ({size_info}{enc_info}, {num_workers} workers)")
        console.print(f"   Cache location: {cache_location}")

        # Use ThreadPoolExecutor for parallel I/O
        failed_count = 0
        executor = ThreadPoolExecutor(max_workers=num_workers)

        # Submit all tasks with spinner (can take a moment for large datasets)
        with spinner(f"Queueing {len(work_items):,} images..."):
            futures = {
                executor.submit(self._load_and_cache_image, idx, path, target_size): idx
                for idx, path in work_items
            }

        # Process results with progress bar
        with progress_bar(len(work_items), cache_desc) as update:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 3:
                        logger.warning(f"Failed to cache image: {e}")
                update(1)

        executor.shutdown(wait=False)

        # Finalize cache
        self._image_cache.finalize()
        cached_count = self._image_cache.size
        console.print(f"[green]✓[/green] Cached {cached_count} images to {cache_mode.upper()} (LMDB)")

        if failed_count > 0:
            console.print(f"[yellow]⚠[/yellow] {failed_count} images failed to cache")

    def _load_and_cache_image(self, idx: int, path: Path, target_size: Optional[Tuple[int, int]]) -> None:
        """Load a single image and store in cache (called by thread workers)."""
        img = self._image_loader(str(path))
        orig_size = img.size if target_size is not None else None  # Save original size only if resizing
        if target_size is not None:
            img = self._resize_for_cache(img, target_size)
        img_np = np.asarray(img).copy()
        self._image_cache.put(idx, img_np, orig_size=orig_size)

    def _setup_from_cache(self) -> None:
        """
        Setup dataset from cache metadata (cache-only mode).

        In cache-only mode, image paths and labels are loaded from the cache
        metadata instead of scanning the filesystem. This enables training
        on machines where only the cache is available, not the original images.

        Raises:
            ValueError: If cache metadata doesn't contain required information.
        """
        from yolo.utils.progress import spinner, console

        # Initialize cache in read-only mode
        dataset_root = self.images_dir.parent
        cache_suffix = f"{self.image_size[0]}x{self.image_size[1]}_f1.0"  # Default suffix

        with spinner("Loading cache metadata (YOLO format)..."):
            # Try to find existing cache
            cache_dir = self._image_cache._build_cache_dir(dataset_root)
            if not cache_dir.exists():
                raise ValueError(
                    f"Cache directory not found: {cache_dir}. "
                    f"Create the cache first with: yolo cache-create"
                )

            # Open cache in read-only mode
            self._image_cache._cache_dir_path = cache_dir
            self._image_cache._db_path = cache_dir / "cache.lmdb"
            self._image_cache._open_db(readonly=True)
            self._image_cache._load_cached_indices()

        # Get metadata
        metadata = self._image_cache.get_metadata()
        if metadata is None:
            raise ValueError(
                "\n"
                "╔══════════════════════════════════════════════════════════════════════╗\n"
                "║  CACHE METADATA NOT FOUND                                            ║\n"
                "╠══════════════════════════════════════════════════════════════════════╣\n"
                "║  Could not read cache metadata. Possible causes:                     ║\n"
                "║                                                                      ║\n"
                "║  1. ENCRYPTED CACHE with cache_encrypt: false                        ║\n"
                "║     If the cache was created with encryption, you must set:          ║\n"
                "║       data:                                                          ║\n"
                "║         cache_encrypt: true                                          ║\n"
                "║     AND provide the encryption key via:                              ║\n"
                "║       export YOLO_ENCRYPTION_KEY='your-64-char-hex-key'              ║\n"
                "║                                                                      ║\n"
                "║  2. CORRUPTED CACHE                                                  ║\n"
                "║     Delete the cache directory and recreate it                       ║\n"
                "║                                                                      ║\n"
                "║  3. OLD CACHE FORMAT (pre cache-only support)                        ║\n"
                "║     Recreate the cache with: yolo cache-create                       ║\n"
                "╚══════════════════════════════════════════════════════════════════════╝"
            )

        # Verify format matches
        cache_format = metadata.get("format")
        if cache_format is not None and cache_format != "yolo":
            raise ValueError(
                "\n"
                "╔══════════════════════════════════════════════════════════════════════╗\n"
                "║  FORMAT MISMATCH                                                     ║\n"
                "╠══════════════════════════════════════════════════════════════════════╣\n"
                f"║  YAML config:  format: yolo                                          ║\n"
                f"║  Cache format: {cache_format:<54}║\n"
                "║                                                                      ║\n"
                "║  The cache was created with a different format than your config.     ║\n"
                "║                                                                      ║\n"
                "║  Solutions:                                                          ║\n"
                f"║  1. Change YAML to: format: {cache_format:<42}║\n"
                "║  2. Recreate cache with: yolo cache-create --data.format yolo        ║\n"
                "╚══════════════════════════════════════════════════════════════════════╝"
            )

        # Get image paths from metadata
        image_paths = metadata.get("image_paths")
        if image_paths is None:
            raise ValueError(
                "Cache does not contain image paths. "
                "This cache was created before cache-only mode was supported. "
                "Recreate the cache with: yolo cache-create"
            )

        # Convert paths back to Path objects
        self.image_files = [Path(p) for p in image_paths]

        # Get labels from metadata (if available)
        labels = metadata.get("labels")
        if labels is not None:
            self._labels_cache = labels

        num_images = len(self.image_files)
        cached_images = len(self._image_cache._cached_indices)
        encrypted = metadata.get("encrypted", False)

        # Verify encryption setting matches
        if encrypted and not self._image_cache._encrypt_cache:
            raise ValueError(
                "\n"
                "╔══════════════════════════════════════════════════════════════════════╗\n"
                "║  ENCRYPTION MISMATCH                                                 ║\n"
                "╠══════════════════════════════════════════════════════════════════════╣\n"
                "║  The cache is ENCRYPTED but cache_encrypt: false in your config.    ║\n"
                "║                                                                      ║\n"
                "║  To use this cache, you must:                                        ║\n"
                "║                                                                      ║\n"
                "║  1. Set in YAML config:                                              ║\n"
                "║       data:                                                          ║\n"
                "║         cache_encrypt: true                                          ║\n"
                "║                                                                      ║\n"
                "║  2. Provide the encryption key used during cache creation:           ║\n"
                "║       export YOLO_ENCRYPTION_KEY='your-64-char-hex-key'              ║\n"
                "║                                                                      ║\n"
                "║  Without the correct key, encrypted images cannot be read.          ║\n"
                "╚══════════════════════════════════════════════════════════════════════╝"
            )

        console.print(f"[green]✓[/green] Cache-only mode (YOLO): {num_images:,} images from cache")
        if encrypted:
            console.print(f"   [dim]Encrypted cache - decryption in memory only[/dim]")

        # Verify cache is complete
        if cached_images < num_images:
            logger.warning(
                f"Cache is incomplete: {cached_images:,}/{num_images:,} images cached. "
                f"Some images may fail to load."
            )

        self._image_cache._initialized = True

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

        from yolo.utils.progress import console

        if cache.is_valid(label_files):
            self._labels_cache = cache.load()["labels"]
            console.print(f"[green]✓[/green] Labels cache: {len(self._labels_cache)} labels loaded ({cache.cache_path})")
        else:
            console.print(f"[bold]📝 Parsing {len(self.image_files)} labels[/bold] (first run or files changed)")
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
            console.print(f"[green]✓[/green] Labels cached to {cache.cache_path}")

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

    def _load_image(self, index: int) -> Tuple[Image.Image, Optional[Tuple[int, int]]]:
        """
        Load image, using cache if available.

        Returns:
            Tuple of (image, orig_size) where orig_size is the original image size
            before cache resize (width, height), or None if not resized.

        Raises:
            RuntimeError: If cache_only=True and image is not in cache.
        """
        image_path = self.image_files[index]

        # Try cache first
        if self._image_cache is not None:
            cached = self._image_cache.get(index)
            if cached is not None:
                img_arr, orig_size = cached
                return Image.fromarray(img_arr), orig_size

        # In cache-only mode, raise error if image not in cache
        if self._cache_only:
            raise RuntimeError(
                f"Cache-only mode: Image not found in cache (index={index}, path={image_path}). "
                f"The cache may be incomplete. Recreate with: yolo cache-create"
            )

        # Load from disk
        img = self._image_loader(str(image_path))

        # Store in cache for next time (no resize when loading from disk directly)
        if self._image_cache is not None:
            img_np = np.asarray(img).copy()
            self._image_cache.put(index, img_np, orig_size=None)

        return img, None  # No resize info when loaded from disk

    def __getitem__(self, index: int):
        """Get image and target at index."""
        # Load image (with cache if enabled)
        image, orig_size = self._load_image(index)

        # Use original size for bbox calculation if image was resized in cache
        if orig_size is not None:
            img_w, img_h = orig_size
        else:
            img_w, img_h = image.size

        # Get labels (from cache or parse on-the-fly)
        if self._labels_cache is not None:
            cached = self._labels_cache[index]
            boxes_norm = cached["boxes_norm"]
            label_ids = cached["labels"]
        else:
            # Parse on-the-fly (no caching)
            image_path = self.image_files[index]
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

        # If image was resized in cache, transform bboxes to match letterboxed image
        if orig_size is not None:
            target = self._transform_bboxes_for_letterbox(target, orig_size, image.size)

        # Apply transforms
        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target

    def _transform_bboxes_for_letterbox(
        self, target: Dict[str, Any], orig_size: Tuple[int, int], target_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Transform bboxes from original image coordinates to letterboxed image coordinates.

        Args:
            target: Target dict with 'boxes' in xyxy pixel format
            orig_size: Original image size (width, height)
            target_size: Letterboxed image size (width, height)

        Returns:
            Target dict with transformed bboxes
        """
        if "boxes" not in target or len(target["boxes"]) == 0:
            return target

        orig_w, orig_h = orig_size
        target_w, target_h = target_size

        # Calculate letterbox parameters (same as _resize_for_cache)
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        # Transform bboxes: pixel coords in original -> pixel coords in letterboxed
        boxes = target["boxes"].clone()

        # Scale coordinates
        boxes[:, 0] = boxes[:, 0] * scale + pad_x  # x1
        boxes[:, 1] = boxes[:, 1] * scale + pad_y  # y1
        boxes[:, 2] = boxes[:, 2] * scale + pad_x  # x2
        boxes[:, 3] = boxes[:, 3] * scale + pad_y  # y2

        target["boxes"] = boxes
        return target
