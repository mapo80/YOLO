"""
Cache Archive Tools for YOLO Dataset Caching.

Provides functionality to create, export, import, and inspect dataset caches.
This enables secure transfer of preprocessed datasets to remote machines
without exposing original images.

Usage:
    # Create cache
    from yolo.tools.cache_archive import create_cache
    create_cache(data_root, "yolo", (640, 640), encrypt=True)

    # Export cache
    from yolo.tools.cache_archive import export_cache
    export_cache(cache_dir, output_path, compression="gzip")

    # Import cache
    from yolo.tools.cache_archive import import_cache
    import_cache(archive_path, output_dir)

    # Get cache info
    from yolo.tools.cache_archive import get_cache_info
    info = get_cache_info(cache_dir)
"""

import os
import pickle
import shutil
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from yolo.utils.logger import logger


def create_cache(
    data_root: Path,
    data_format: Literal["coco", "yolo"],
    image_size: Tuple[int, int],
    train_images: Optional[str] = None,
    val_images: Optional[str] = None,
    train_labels: Optional[str] = None,
    val_labels: Optional[str] = None,
    train_ann: Optional[str] = None,
    val_ann: Optional[str] = None,
    train_split: Optional[str] = None,
    val_split: Optional[str] = None,
    encrypt: bool = False,
    workers: Optional[int] = None,
    split: Literal["train", "val", "both"] = "both",
    data_fraction: float = 1.0,
    output_dir: Optional[Path] = None,
    sync: bool = False,
    cache_format: Literal["jpeg", "raw"] = "jpeg",
    jpeg_quality: int = 95,
    compress: bool = False,  # Deprecated: LZ4 is now automatic for raw format
) -> Path:
    """
    Create LMDB cache from dataset without running training.

    This function creates a cache of preprocessed images that can be
    transferred to remote machines for training without the original images.

    Args:
        data_root: Root directory of the dataset.
        data_format: Dataset format - 'coco' or 'yolo'.
        image_size: Target image size (width, height) for caching.
        train_images: Path to training images (relative to data_root).
        val_images: Path to validation images (relative to data_root).
        train_labels: Path to training labels for YOLO format.
        val_labels: Path to validation labels for YOLO format.
        train_ann: Path to training annotations for COCO format.
        val_ann: Path to validation annotations for COCO format.
        encrypt: Whether to encrypt the cache (requires YOLO_ENCRYPTION_KEY).
        workers: Number of parallel workers for caching (None = auto).
        split: Which split to cache - 'train', 'val', or 'both'.
        data_fraction: Fraction of data to cache (1.0 = all).
        output_dir: Directory where cache will be created (default: data_root).
            Useful when data_root is on a slow/external volume.
        sync: Enable LMDB fsync for crash safety (default: False for compatibility).
        cache_format: Cache format - 'jpeg' (compressed, ~10x smaller) or 'raw' (lossless).
            JPEG uses TurboJPEG for fast encoding/decoding.
            RAW stores uncompressed numpy arrays with automatic LZ4 compression.
        jpeg_quality: JPEG quality (1-100). Only used when cache_format='jpeg'. Default 95.
        compress: Deprecated. LZ4 compression is now automatic for 'raw' format only.

    Returns:
        Path to the created cache directory.

    Raises:
        ValueError: If required paths are missing or encryption key not set.
        FileNotFoundError: If dataset directories don't exist.
    """
    from yolo.data.cache import ImageCache
    from yolo.data.datamodule import CocoDetectionWrapper, YOLOFormatDataset

    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    # Validate encryption key if encryption is requested
    encryption_key = None
    if encrypt:
        encryption_key = os.environ.get("YOLO_ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError(
                "Encryption requested but YOLO_ENCRYPTION_KEY environment variable not set. "
                "Generate a key with: python -c \"import os; print(os.urandom(32).hex())\""
            )

    # Build cache suffix (must match datamodule.py naming)
    if image_size is not None:
        size_str = f"{image_size[0]}x{image_size[1]}"
    else:
        size_str = "orig"  # Must match datamodule.py
    cache_suffix = f"{size_str}_f{data_fraction}"

    # Use output_dir if specified, otherwise use data_root
    cache_base_dir = Path(output_dir) if output_dir else data_root
    if output_dir:
        cache_base_dir.mkdir(parents=True, exist_ok=True)

    # Pre-count images for COCO format to allocate correct map size
    expected_total_images = None
    if data_format == "coco":
        total_coco_images = 0
        if split in ("train", "both") and train_ann:
            ann_file = data_root / train_ann
            if ann_file.exists():
                total_coco_images += _count_coco_images(ann_file)
        if split in ("val", "both") and val_ann:
            ann_file = data_root / val_ann
            if ann_file.exists():
                total_coco_images += _count_coco_images(ann_file)
        if total_coco_images > 0:
            expected_total_images = total_coco_images
            logger.info(f"Pre-counted COCO images: {expected_total_images:,}")

    # Create image cache
    image_cache = ImageCache(
        mode="disk",
        cache_dir=cache_base_dir,
        max_memory_gb=float("inf"),  # No limit for creation
        target_size=image_size,
        encryption_key=encryption_key,
        cache_suffix=cache_suffix,
        refresh=True,  # Always create fresh cache
        sync=sync,
        expected_total_images=expected_total_images,
        cache_format=cache_format,
        jpeg_quality=jpeg_quality,
    )

    logger.info(f"Creating cache for {data_format} dataset at {data_root}")
    if image_size is not None:
        logger.info(f"  Image size: {image_size[0]}x{image_size[1]}")
    else:
        logger.info(f"  Image size: original (no resize)")
    logger.info(f"  Cache format: {cache_format}")
    if cache_format == "jpeg":
        logger.info(f"  JPEG quality: {jpeg_quality}")
    else:
        logger.info(f"  LZ4 compression: enabled (automatic for raw)")
    logger.info(f"  Encryption: {'enabled' if encrypt else 'disabled'}")
    logger.info(f"  Split: {split}")

    datasets_to_cache = []

    # Variables for YOLO cache-only mode (labels + split indices)
    yolo_labels_to_save = {}
    yolo_split_indices = {"train": [], "val": []}
    current_index = 0

    # Setup datasets based on format and split
    if data_format == "yolo":
        # Check if train and val use the same directories (common YOLO setup with split files)
        same_dirs = (train_images == val_images and train_labels == val_labels)

        if same_dirs and split == "both" and train_images and train_labels:
            # Single directory with split files - create ONE dataset with all images
            images_dir = data_root / train_images
            labels_dir = data_root / train_labels
            if not images_dir.exists():
                raise FileNotFoundError(f"Images directory not found: {images_dir}")
            if not labels_dir.exists():
                raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

            logger.info(f"Detected shared train/val directories - caching all images once")

            dataset = YOLOFormatDataset(
                images_dir=str(images_dir),
                labels_dir=str(labels_dir),
                transforms=None,
                image_size=image_size,
                image_loader=None,
                cache_labels=True,
                cache_refresh=True,
                image_cache=image_cache,
                data_fraction=data_fraction,
                cache_workers=workers,
                skip_finalize=True,
                creating_cache=True,
                cache_index_offset=0,
            )
            datasets_to_cache.append(("all", dataset))

            # Collect YOLO labels for cache-only mode (all images)
            yolo_labels_to_save["all"] = _extract_yolo_labels(dataset)

            # Build filename to index mapping for split file processing
            all_count = len(dataset)
            image_files = dataset.image_files
            filename_to_idx = {f.name: idx for idx, f in enumerate(image_files)}

            # Read split files to determine train/val indices
            if train_split and val_split:
                train_split_path = data_root / train_split
                val_split_path = data_root / val_split

                if not train_split_path.exists():
                    raise FileNotFoundError(f"Train split file not found: {train_split_path}")
                if not val_split_path.exists():
                    raise FileNotFoundError(f"Val split file not found: {val_split_path}")

                # Read train split
                with open(train_split_path, "r") as f:
                    train_files = {Path(line.strip()).name for line in f if line.strip()}
                yolo_split_indices["train"] = [
                    idx for idx, img_file in enumerate(image_files)
                    if img_file.name in train_files
                ]

                # Read val split
                with open(val_split_path, "r") as f:
                    val_files = {Path(line.strip()).name for line in f if line.strip()}
                yolo_split_indices["val"] = [
                    idx for idx, img_file in enumerate(image_files)
                    if img_file.name in val_files
                ]

                logger.info(f"Split indices from files: train={len(yolo_split_indices['train'])}, val={len(yolo_split_indices['val'])}")
            else:
                # No split files provided - all indices available for both
                logger.warning("No split files provided - all images will be available for both train and val")
                yolo_split_indices["train"] = list(range(all_count))
                yolo_split_indices["val"] = list(range(all_count))

        else:
            # Separate directories for train/val - original logic
            if split in ("train", "both") and train_images and train_labels:
                images_dir = data_root / train_images
                labels_dir = data_root / train_labels
                if not images_dir.exists():
                    raise FileNotFoundError(f"Training images directory not found: {images_dir}")
                if not labels_dir.exists():
                    raise FileNotFoundError(f"Training labels directory not found: {labels_dir}")

                dataset = YOLOFormatDataset(
                    images_dir=str(images_dir),
                    labels_dir=str(labels_dir),
                    transforms=None,
                    image_size=image_size,
                    image_loader=None,
                    cache_labels=True,
                    cache_refresh=True,
                    image_cache=image_cache,
                    data_fraction=data_fraction,
                    cache_workers=workers,
                    skip_finalize=True,
                    creating_cache=True,
                    cache_index_offset=current_index,
                )
                datasets_to_cache.append(("train", dataset))

                # Collect YOLO labels for cache-only mode
                yolo_labels_to_save["train"] = _extract_yolo_labels(dataset)

                # Track split indices
                train_count = len(dataset)
                yolo_split_indices["train"] = list(range(current_index, current_index + train_count))
                current_index += train_count

            if split in ("val", "both") and val_images and val_labels:
                images_dir = data_root / val_images
                labels_dir = data_root / val_labels
                if not images_dir.exists():
                    raise FileNotFoundError(f"Validation images directory not found: {images_dir}")
                if not labels_dir.exists():
                    raise FileNotFoundError(f"Validation labels directory not found: {labels_dir}")

                dataset = YOLOFormatDataset(
                    images_dir=str(images_dir),
                    labels_dir=str(labels_dir),
                    transforms=None,
                    image_size=image_size,
                    image_loader=None,
                    cache_labels=True,
                    cache_refresh=False,  # Don't refresh - reuse cache from train
                    image_cache=image_cache,
                    data_fraction=data_fraction,
                    cache_workers=workers,
                    skip_finalize=True,
                    creating_cache=True,
                    cache_index_offset=current_index,
                )
                datasets_to_cache.append(("val", dataset))

                # Collect YOLO labels for cache-only mode
                yolo_labels_to_save["val"] = _extract_yolo_labels(dataset)

                # Track split indices
                val_count = len(dataset)
                yolo_split_indices["val"] = list(range(current_index, current_index + val_count))
                current_index += val_count

    else:  # COCO format
        coco_annotations_to_save = {}

        if split in ("train", "both") and train_images and train_ann:
            images_dir = data_root / train_images
            ann_file = data_root / train_ann
            if not images_dir.exists():
                raise FileNotFoundError(f"Training images directory not found: {images_dir}")
            if not ann_file.exists():
                raise FileNotFoundError(f"Training annotations not found: {ann_file}")

            train_offset = current_index
            dataset = CocoDetectionWrapper(
                root=str(images_dir),
                annFile=str(ann_file),
                transforms=None,
                image_size=image_size,
                image_loader=None,
                image_cache=image_cache,
                data_fraction=data_fraction,
                cache_workers=workers,
                skip_finalize=True,
                creating_cache=True,
                cache_index_offset=train_offset,
            )
            datasets_to_cache.append(("train", dataset))
            current_index += len(dataset)

            # Collect COCO annotations for cache-only mode
            coco_annotations_to_save["train"] = _extract_coco_annotations(dataset)

        if split in ("val", "both") and val_images and val_ann:
            images_dir = data_root / val_images
            ann_file = data_root / val_ann
            if not images_dir.exists():
                raise FileNotFoundError(f"Validation images directory not found: {images_dir}")
            if not ann_file.exists():
                raise FileNotFoundError(f"Validation annotations not found: {ann_file}")

            val_offset = current_index
            dataset = CocoDetectionWrapper(
                root=str(images_dir),
                annFile=str(ann_file),
                transforms=None,
                image_size=image_size,
                image_loader=None,
                image_cache=image_cache,
                data_fraction=data_fraction,
                cache_workers=workers,
                skip_finalize=True,
                creating_cache=True,
                cache_index_offset=val_offset,
            )
            datasets_to_cache.append(("val", dataset))
            current_index += len(dataset)

            # Collect COCO annotations for cache-only mode
            coco_annotations_to_save["val"] = _extract_coco_annotations(dataset)

    if not datasets_to_cache:
        raise ValueError(
            f"No datasets to cache. Check that paths are provided for split='{split}' "
            f"and format='{data_format}'."
        )

    # Collect all image paths from all datasets for proper LMDB initialization
    all_image_paths = []
    dataset_info = []  # (split_name, dataset, start_idx, num_images, image_paths)

    for split_name, dataset in datasets_to_cache:
        # Get image paths from dataset
        if hasattr(dataset, 'image_files'):
            # YOLO format
            image_paths = list(dataset.image_files)
        elif hasattr(dataset, 'coco') and hasattr(dataset, 'ids'):
            # COCO format
            image_paths = [
                Path(dataset.root) / dataset.coco.loadImgs(img_id)[0]["file_name"]
                for img_id in dataset.ids
            ]
        else:
            raise ValueError(f"Unknown dataset type: {type(dataset)}")

        start_idx = len(all_image_paths)
        dataset_info.append((split_name, dataset, start_idx, len(image_paths), image_paths))
        all_image_paths.extend(image_paths)

    # Initialize LMDB cache ONCE with ALL image paths
    logger.info(f"Initializing cache for {len(all_image_paths):,} total images...")
    image_cache.initialize(
        num_images=len(all_image_paths),
        cache_dir=cache_base_dir,
        paths=all_image_paths,
    )

    # Use _precache_images() from datasets (unified caching logic)
    # The datasets were created with skip_finalize=True, so they didn't cache in __init__
    # Now we call their _precache_images() method which uses the already-initialized cache
    total_images = 0
    for split_name, dataset, start_idx, num_images, image_paths in dataset_info:
        logger.info(f"Caching {split_name} split ({num_images:,} images)...")
        dataset._precache_images()
        total_images += num_images

    # Save dataset format to cache metadata (for format verification)
    logger.info(f"Saving dataset format '{data_format}' to cache metadata...")
    image_cache.save_format(data_format)

    # Save COCO annotations and split indices to cache (for cache-only mode)
    if data_format == "coco" and coco_annotations_to_save:
        logger.info("Saving COCO annotations to cache metadata...")
        # Merge annotations from all splits with global indices
        merged_annotations = {
            "annotations": {},
            "categories": [],
        }
        coco_split_indices = {"train": [], "val": []}

        for split_name, dataset, start_idx, num_images, image_paths in dataset_info:
            if split_name in coco_annotations_to_save:
                annots = coco_annotations_to_save[split_name]
                # Re-index annotations with global indices
                for local_idx_str, ann_list in annots["annotations"].items():
                    global_idx = start_idx + int(local_idx_str)
                    merged_annotations["annotations"][str(global_idx)] = ann_list
                if not merged_annotations["categories"]:
                    merged_annotations["categories"] = annots["categories"]

                # Track split indices
                coco_split_indices[split_name] = list(range(start_idx, start_idx + num_images))

        image_cache.save_coco_annotations(merged_annotations)

        # Save split indices for COCO
        logger.info("Saving COCO split indices to cache metadata...")
        image_cache.save_split_indices(
            train_indices=coco_split_indices.get("train", []),
            val_indices=coco_split_indices.get("val", []),
        )

    # Save YOLO labels and split indices to cache (for cache-only mode)
    if data_format == "yolo" and yolo_labels_to_save:
        logger.info("Saving YOLO labels to cache metadata...")
        # Merge labels from all splits (maintaining order: train first, then val, or all)
        merged_labels = []
        if "all" in yolo_labels_to_save:
            # Shared directory case - all labels together
            merged_labels = yolo_labels_to_save["all"]
        else:
            for split_name in ["train", "val"]:
                if split_name in yolo_labels_to_save:
                    merged_labels.extend(yolo_labels_to_save[split_name])
        image_cache.save_labels(merged_labels)

        # Save split indices
        logger.info("Saving split indices to cache metadata...")
        image_cache.save_split_indices(
            train_indices=yolo_split_indices.get("train", []),
            val_indices=yolo_split_indices.get("val", []),
        )

    # Finalize cache
    if image_cache._env is not None:
        image_cache.finalize()

    cache_path = image_cache.cache_path
    logger.info(f"Cache created successfully: {cache_path}")
    logger.info(f"  Total images cached: {total_images}")

    # Run sanity check
    logger.info("Running cache sanity check...")
    check_result = image_cache.sanity_check()

    if check_result["valid"]:
        logger.info("✓ Cache sanity check passed")
        stats = check_result["stats"]
        logger.info(f"  Format: {stats.get('format', 'unknown')}")
        logger.info(f"  Images: {stats.get('num_images_cached', 0)}")
        logger.info(f"  Size: {stats.get('target_size', 'original')}")
        logger.info(f"  Encrypted: {stats.get('encrypted', False)}")
        if stats.get('num_labels'):
            logger.info(f"  Labels: {stats.get('num_labels', 0)}")
        if stats.get('num_train_indices'):
            logger.info(f"  Train indices: {stats.get('num_train_indices', 0)}")
            logger.info(f"  Val indices: {stats.get('num_val_indices', 0)}")
    else:
        for error in check_result["errors"]:
            logger.error(f"✗ {error}")
        raise ValueError(
            f"Cache sanity check failed with {len(check_result['errors'])} error(s). "
            "Delete the cache and try again."
        )

    for warning in check_result.get("warnings", []):
        logger.warning(f"⚠ {warning}")

    return cache_path


def _extract_coco_annotations(dataset) -> Dict[str, Any]:
    """
    Extract COCO annotations from a CocoDetectionWrapper dataset.

    Args:
        dataset: CocoDetectionWrapper instance.

    Returns:
        Dictionary with 'annotations' (dict mapping index to annotation list)
        and 'categories' (list of category dicts).
    """
    annotations = {}
    for idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        anns = dataset.coco.loadAnns(ann_ids)
        annotations[str(idx)] = anns

    categories = dataset.coco.loadCats(dataset.coco.getCatIds())

    return {
        "annotations": annotations,
        "categories": categories,
    }


def _count_coco_images(ann_file: Path) -> int:
    """
    Count number of images in a COCO annotation file.

    Args:
        ann_file: Path to COCO annotation JSON file.

    Returns:
        Number of images in the annotation file.
    """
    import json

    with open(ann_file, "r") as f:
        data = json.load(f)
    return len(data.get("images", []))


def _resize_for_cache(img, target_size: Tuple[int, int]):
    """
    Resize image with letterboxing (maintain aspect ratio, pad with gray).

    Args:
        img: PIL Image to resize.
        target_size: Target size (width, height).

    Returns:
        Resized PIL Image.
    """
    from PIL import Image

    target_w, target_h = target_size
    orig_w, orig_h = img.size

    # Calculate scale to fit within target while maintaining aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize image
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Create gray background and paste resized image centered
    new_img = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))

    return new_img


def _extract_yolo_labels(dataset) -> List[Dict[str, Any]]:
    """
    Extract YOLO labels from a YOLOFormatDataset.

    This function extracts all labels from a YOLO dataset for storage
    in cache metadata, enabling cache-only mode without label files.

    Args:
        dataset: YOLOFormatDataset instance with cached labels.

    Returns:
        List of label dictionaries, each containing:
        - 'labels': numpy array of class indices
        - 'boxes_norm': numpy array of normalized bbox coordinates [x_center, y_center, w, h]
    """
    # Use the dataset's internal label cache if available
    if hasattr(dataset, "_labels_cache") and dataset._labels_cache is not None:
        return dataset._labels_cache

    # Fallback: parse labels from files
    labels = []
    for i in range(len(dataset)):
        label_data = dataset._parse_label_file(dataset.image_files[i])
        labels.append(label_data)

    return labels


def export_cache(
    cache_dir: Path,
    output_path: Optional[Path] = None,
    compression: Literal["gzip", "none"] = "gzip",
    progress: bool = True,
) -> Path:
    """
    Export cache directory to a compressed archive.

    Creates a tar archive of the LMDB cache that can be transferred
    to remote machines.

    Args:
        cache_dir: Path to the cache directory (.yolo_cache_*).
        output_path: Output archive path. If None, uses {cache_dir}.tar.gz.
        compression: Compression type - 'gzip' or 'none'.
        progress: Whether to show progress during export.

    Returns:
        Path to the created archive.

    Raises:
        FileNotFoundError: If cache directory doesn't exist.
        ValueError: If cache directory is invalid.
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    # Validate cache structure
    lmdb_dir = cache_dir / "cache.lmdb"
    if not lmdb_dir.exists():
        raise ValueError(
            f"Invalid cache directory: {cache_dir}. "
            f"Expected LMDB database at {lmdb_dir}"
        )

    # Determine output path
    if output_path is None:
        if compression == "gzip":
            output_path = cache_dir.parent / f"{cache_dir.name}.tar.gz"
        else:
            output_path = cache_dir.parent / f"{cache_dir.name}.tar"
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine tar mode
    if compression == "gzip":
        mode = "w:gz"
    else:
        mode = "w"

    logger.info(f"Exporting cache: {cache_dir}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Compression: {compression}")

    start_time = time.time()

    # Create archive
    with tarfile.open(output_path, mode) as tar:
        # Add the cache directory with its name preserved
        tar.add(cache_dir, arcname=cache_dir.name)

    elapsed = time.time() - start_time
    size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info(f"Export complete in {elapsed:.1f}s")
    logger.info(f"  Archive size: {size_mb:.1f} MB")

    return output_path


def import_cache(
    archive_path: Path,
    output_dir: Optional[Path] = None,
    progress: bool = True,
) -> Path:
    """
    Import cache from archive to target directory.

    Extracts a cache archive created by export_cache.

    Args:
        archive_path: Path to the cache archive (.tar.gz or .tar).
        output_dir: Target directory for extraction. If None, uses current dir.
        progress: Whether to show progress during import.

    Returns:
        Path to the extracted cache directory.

    Raises:
        FileNotFoundError: If archive doesn't exist.
        ValueError: If archive is invalid.
    """
    archive_path = Path(archive_path)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Importing cache: {archive_path}")
    logger.info(f"  Output directory: {output_dir}")

    start_time = time.time()

    # Determine tar mode based on file extension
    if archive_path.suffix == ".gz" or str(archive_path).endswith(".tar.gz"):
        mode = "r:gz"
    else:
        mode = "r"

    # Extract archive
    try:
        with tarfile.open(archive_path, mode) as tar:
            # Get the root directory name from the archive
            members = tar.getmembers()
            if not members:
                raise ValueError(f"Empty archive: {archive_path}")

            # Find the cache directory name
            root_name = members[0].name.split("/")[0]

            # Check if cache already exists
            cache_path = output_dir / root_name
            if cache_path.exists():
                logger.warning(f"Removing existing cache: {cache_path}")
                shutil.rmtree(cache_path)

            # Extract all files
            tar.extractall(output_dir)

    except tarfile.TarError as e:
        raise ValueError(f"Invalid or corrupted archive: {archive_path}. Error: {e}")

    elapsed = time.time() - start_time

    # Verify extraction
    cache_path = output_dir / root_name
    if not cache_path.exists():
        raise ValueError(f"Extraction failed: cache directory not found at {cache_path}")

    lmdb_dir = cache_path / "cache.lmdb"
    if not lmdb_dir.exists():
        raise ValueError(f"Invalid cache archive: no LMDB database found")

    logger.info(f"Import complete in {elapsed:.1f}s")
    logger.info(f"  Cache path: {cache_path}")

    return cache_path


def get_cache_info(cache_dir: Path) -> Dict[str, Any]:
    """
    Get information about a cache directory.

    Reads metadata from the LMDB cache and returns statistics.

    Args:
        cache_dir: Path to the cache directory (.yolo_cache_*).

    Returns:
        Dictionary containing cache information:
        - version: Cache format version
        - num_images: Number of cached images
        - target_size: Image size (width, height) or None
        - encrypted: Whether cache is encrypted
        - size_bytes: Total size in bytes
        - size_human: Human-readable size
        - path: Cache path
        - created: Creation timestamp (if available)

    Raises:
        FileNotFoundError: If cache directory doesn't exist.
        ValueError: If cache is invalid or corrupted.
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    lmdb_dir = cache_dir / "cache.lmdb"
    if not lmdb_dir.exists():
        raise ValueError(
            f"Invalid cache directory: {cache_dir}. "
            f"Expected LMDB database at {lmdb_dir}"
        )

    # Calculate directory size
    total_size = 0
    for f in cache_dir.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size

    # Read metadata from LMDB
    try:
        import lmdb

        env = lmdb.open(
            str(lmdb_dir),
            readonly=True,
            lock=False,
            readahead=False,
        )

        metadata = {}
        num_images = 0

        with env.begin() as txn:
            # Get metadata
            meta_data = txn.get(b"__metadata__")
            if meta_data is not None:
                metadata = pickle.loads(meta_data)

            # Count images (keys that are not metadata)
            cursor = txn.cursor()
            for key, _ in cursor:
                if key != b"__metadata__":
                    num_images += 1

        env.close()

    except Exception as e:
        raise ValueError(f"Failed to read cache metadata: {e}")

    # Build info dict
    info = {
        "path": str(cache_dir),
        "version": metadata.get("version", "unknown"),
        "num_images": metadata.get("num_images", num_images),
        "target_size": metadata.get("target_size"),
        "encrypted": metadata.get("encrypted", False),
        "cache_format": metadata.get("cache_format", "raw"),  # Default raw for old caches
        "jpeg_quality": metadata.get("jpeg_quality"),
        "size_bytes": total_size,
        "size_human": _format_size(total_size),
        "cache_suffix": metadata.get("cache_suffix"),
        "paths_hash": metadata.get("paths_hash"),
    }

    # Try to get creation time from file
    try:
        data_mdb = lmdb_dir / "data.mdb"
        if data_mdb.exists():
            mtime = data_mdb.stat().st_mtime
            info["created"] = datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        pass

    return info


def print_cache_info(cache_dir: Path) -> None:
    """
    Print formatted cache information to console.

    Args:
        cache_dir: Path to the cache directory.
    """
    info = get_cache_info(cache_dir)

    print("\nCache Information")
    print("─" * 45)
    print(f"  Path:        {info['path']}")
    print(f"  Version:     {info['version']}")
    print(f"  Images:      {info['num_images']:,}")
    print(f"  Size:        {info['size_human']}")
    print(f"  Encrypted:   {'Yes' if info['encrypted'] else 'No'}")

    if info.get("target_size"):
        w, h = info["target_size"]
        print(f"  Target Size: {w}x{h}")

    if info.get("created"):
        print(f"  Created:     {info['created']}")

    print("─" * 45)


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


__all__ = [
    "create_cache",
    "export_cache",
    "import_cache",
    "get_cache_info",
    "print_cache_info",
    "_extract_yolo_labels",
]
