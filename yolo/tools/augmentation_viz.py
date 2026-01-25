"""
Augmentation visualization tool for debugging.

Instantiates YOLODataModule using the same parameters from config YAML,
ensuring consistent behavior with training.
"""

import random
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from yolo.tools.drawer import draw_bboxes
from yolo.utils.logger import logger


def _get_bbox_mosaic_config(data_cfg) -> Optional[dict]:
    """
    Extract bbox_mosaic config as dict from data config.

    Handles both OmegaConf DictConfig and regular dict/Namespace.
    Returns None if bbox_mosaic is not configured.
    """
    bbox_mosaic_cfg = getattr(data_cfg, "bbox_mosaic", None)
    if bbox_mosaic_cfg is None:
        return None

    # Convert OmegaConf/Namespace to plain dict
    if hasattr(bbox_mosaic_cfg, "items"):
        # Dict-like object
        return dict(bbox_mosaic_cfg)
    elif hasattr(bbox_mosaic_cfg, "__dict__"):
        # Namespace-like object
        return vars(bbox_mosaic_cfg)
    else:
        return None


def _create_datamodule_from_config(config_path: str):
    """
    Instantiate YOLODataModule from config YAML.

    Uses the same parameter extraction as qat-finetune and other CLI commands,
    ensuring consistent behavior with training.
    """
    from omegaconf import ListConfig, OmegaConf

    from yolo.data.datamodule import YOLODataModule

    # Load config
    config = OmegaConf.load(config_path)

    # Extract image_size from model section (like LightningCLI's link_arguments)
    model_cfg = config.get("model", {})
    image_size = model_cfg.get("image_size", [640, 640])
    # OmegaConf returns ListConfig, not list/tuple, so check explicitly
    if isinstance(image_size, (list, tuple, ListConfig)):
        image_size = tuple(int(x) for x in image_size)
    else:
        image_size = (int(image_size), int(image_size))

    # Extract data configuration
    if not hasattr(config, "data"):
        raise ValueError("Config file must have 'data' section")

    data_cfg = config.data

    # Create datamodule with all parameters from config
    # This is the same pattern used by qat-finetune and other CLI commands
    datamodule = YOLODataModule(
        # Dataset format and paths
        format=getattr(data_cfg, "format", "coco"),
        root=data_cfg.root,
        train_images=getattr(data_cfg, "train_images", "train2017"),
        val_images=getattr(data_cfg, "val_images", "val2017"),
        train_labels=getattr(data_cfg, "train_labels", "train/labels"),
        val_labels=getattr(data_cfg, "val_labels", "valid/labels"),
        train_ann=getattr(data_cfg, "train_ann", "annotations/instances_train2017.json"),
        val_ann=getattr(data_cfg, "val_ann", "annotations/instances_val2017.json"),
        train_split=getattr(data_cfg, "train_split", None),
        val_split=getattr(data_cfg, "val_split", None),
        # Override for visualization: single-threaded, batch=1
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
        # Image size from model config
        image_size=image_size,
        # Augmentation parameters from config
        mosaic_prob=getattr(data_cfg, "mosaic_prob", 1.0),
        mosaic_9_prob=getattr(data_cfg, "mosaic_9_prob", 0.0),
        mixup_prob=getattr(data_cfg, "mixup_prob", 0.15),
        mixup_alpha=getattr(data_cfg, "mixup_alpha", 32.0),
        cutmix_prob=getattr(data_cfg, "cutmix_prob", 0.0),
        # bbox_mosaic: pass nested dict config (or None if not present)
        bbox_mosaic=_get_bbox_mosaic_config(data_cfg),
        hsv_h=getattr(data_cfg, "hsv_h", 0.015),
        hsv_s=getattr(data_cfg, "hsv_s", 0.7),
        hsv_v=getattr(data_cfg, "hsv_v", 0.4),
        degrees=getattr(data_cfg, "degrees", 0.0),
        translate=getattr(data_cfg, "translate", 0.1),
        scale=getattr(data_cfg, "scale", 0.9),
        shear=getattr(data_cfg, "shear", 0.0),
        perspective=getattr(data_cfg, "perspective", 0.0),
        flip_lr=getattr(data_cfg, "flip_lr", 0.5),
        flip_ud=getattr(data_cfg, "flip_ud", 0.0),
        close_mosaic_epochs=getattr(data_cfg, "close_mosaic_epochs", 15),
        # Data sampling
        data_fraction=getattr(data_cfg, "data_fraction", 1.0),
        # Caching parameters
        cache_labels=getattr(data_cfg, "cache_labels", True),
        cache_images=getattr(data_cfg, "cache_images", "none"),
        cache_dir=getattr(data_cfg, "cache_dir", None),
        cache_resize_images=getattr(data_cfg, "cache_resize_images", True),
        cache_max_memory_gb=getattr(data_cfg, "cache_max_memory_gb", 8.0),
        cache_workers=getattr(data_cfg, "cache_workers", None),
        cache_refresh=getattr(data_cfg, "cache_refresh", False),
        cache_encrypt=getattr(data_cfg, "cache_encrypt", False),
        cache_format=getattr(data_cfg, "cache_format", "jpeg"),
        jpeg_quality=getattr(data_cfg, "jpeg_quality", 95),
        cache_only=getattr(data_cfg, "cache_only", False),
        cache_sync=getattr(data_cfg, "cache_sync", False),
    )

    # Return dataset root for default output directory
    dataset_root = str(data_cfg.root)

    return datamodule, image_size, dataset_root


def visualize_augmentations(
    config_path: str,
    num_samples: int = 10,
    indices: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed: Optional[int] = None,
    show_original: bool = False,
    create_grid: bool = False,
) -> int:
    """
    Visualize augmented samples with bounding boxes.

    Uses the same datamodule parameters as training.

    Args:
        config_path: Path to training config YAML
        num_samples: Number of random samples to visualize
        indices: Comma-separated specific indices (overrides num_samples)
        output_dir: Output directory for visualizations (default: dataset_root/aug_viz)
        seed: Random seed for reproducibility
        show_original: Also save original image without augmentation
        create_grid: Create grid image instead of separate files

    Returns:
        Exit code (0 for success)
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    logger.info(f"Loading config from {config_path}")

    # Create datamodule using same parameters as training
    datamodule, image_size, dataset_root = _create_datamodule_from_config(config_path)

    # Default output directory is inside dataset root
    if output_dir is None:
        output_dir = str(Path(dataset_root) / "aug_viz")

    # Clean and recreate output directory
    output_path = Path(output_dir)
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
        logger.info(f"Cleaned existing output directory: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Log augmentation settings from datamodule hparams
    logger.info("Augmentation settings (from config):")
    logger.info(f"  mosaic_prob: {datamodule.hparams.mosaic_prob}")
    logger.info(f"  mosaic_9_prob: {datamodule.hparams.mosaic_9_prob}")
    logger.info(f"  mixup_prob: {datamodule.hparams.mixup_prob}")
    logger.info(f"  cutmix_prob: {datamodule.hparams.cutmix_prob}")
    logger.info(f"  flip_lr: {datamodule.hparams.flip_lr}")
    logger.info(f"  flip_ud: {datamodule.hparams.flip_ud}")
    logger.info(f"  degrees: {datamodule.hparams.degrees}")
    logger.info(f"  scale: {datamodule.hparams.scale}")
    logger.info(f"  image_size: {image_size}")

    # Setup training data (applies augmentations)
    datamodule.setup(stage="fit")
    train_dataset = datamodule.train_dataset
    dataset_len = len(train_dataset)

    logger.info(f"Dataset size: {dataset_len}")

    # Get class names for labeling
    class_names = datamodule.class_names
    if class_names:
        logger.info(f"Class names: {class_names}")
    else:
        logger.warning("No class names found, will display class indices")

    # Determine indices to visualize
    if indices:
        sample_indices = [int(i.strip()) for i in indices.split(",")]
    else:
        sample_indices = random.sample(range(dataset_len), min(num_samples, dataset_len))

    logger.info(f"Visualizing indices: {sample_indices}")

    images_for_grid = []

    for idx in sample_indices:
        # Get augmented sample
        image_tensor, target = train_dataset[idx]

        # Convert tensor to PIL
        if isinstance(image_tensor, torch.Tensor):
            image_pil = to_pil_image(image_tensor)
        else:
            image_pil = image_tensor

        # Prepare bboxes for drawer: [class_id, x1, y1, x2, y2]
        boxes = target["boxes"]
        labels = target["labels"]

        if len(boxes) > 0:
            bboxes_for_draw = []
            for box, label in zip(boxes, labels):
                bboxes_for_draw.append([
                    label.item(),
                    box[0].item(), box[1].item(),
                    box[2].item(), box[3].item()
                ])
        else:
            bboxes_for_draw = []

        # Draw bboxes with class names
        # Note: draw_bboxes expects bboxes as [batch, boxes, coords] or list wrapped in outer list
        # because it does bboxes = bboxes[0] for list inputs (line 35-36 in drawer.py)
        image_with_boxes = draw_bboxes(image_pil, [bboxes_for_draw], idx2label=class_names)

        if create_grid:
            images_for_grid.append(image_with_boxes)
        else:
            # Save individual image
            output_file = output_path / f"aug_{idx:04d}.jpg"
            image_with_boxes.save(output_file, quality=95)
            logger.info(f"Saved: {output_file} ({len(boxes)} boxes)")

        # Optionally save original (without augmentation)
        if show_original:
            base_dataset = datamodule.train_dataset.dataset
            orig_img, orig_target = base_dataset[idx]
            if isinstance(orig_img, torch.Tensor):
                orig_pil = to_pil_image(orig_img)
            else:
                orig_pil = orig_img

            orig_boxes = orig_target["boxes"]
            orig_labels = orig_target["labels"]
            orig_bboxes = []
            for box, label in zip(orig_boxes, orig_labels):
                orig_bboxes.append([label.item(), box[0].item(), box[1].item(), box[2].item(), box[3].item()])

            orig_with_boxes = draw_bboxes(orig_pil, [orig_bboxes], idx2label=class_names)
            orig_file = output_path / f"orig_{idx:04d}.jpg"
            orig_with_boxes.save(orig_file, quality=95)

    # Create grid if requested
    if create_grid and images_for_grid:
        grid_img = _create_image_grid(images_for_grid)
        grid_file = output_path / "grid.jpg"
        grid_img.save(grid_file, quality=95)
        logger.info(f"Saved grid: {grid_file}")

    logger.info(f"Visualization complete. Output: {output_path}")
    return 0


def _create_image_grid(images: List[Image.Image], cols: int = 4) -> Image.Image:
    """Create a grid of images."""
    if not images:
        return Image.new("RGB", (100, 100), color=(128, 128, 128))

    n = len(images)
    rows = (n + cols - 1) // cols

    w = max(img.width for img in images)
    h = max(img.height for img in images)

    grid = Image.new("RGB", (cols * w, rows * h), color=(114, 114, 114))

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        img_resized = img.resize((w, h), Image.BILINEAR)
        grid.paste(img_resized, (col * w, row * h))

    return grid
