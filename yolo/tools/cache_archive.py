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
    encrypt: bool = False,
    workers: Optional[int] = None,
    split: Literal["train", "val", "both"] = "both",
    data_fraction: float = 1.0,
    output_dir: Optional[Path] = None,
    sync: bool = False,
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

    # Build cache suffix
    size_str = f"{image_size[0]}x{image_size[1]}"
    cache_suffix = f"{size_str}_f{data_fraction}"

    # Use output_dir if specified, otherwise use data_root
    cache_base_dir = Path(output_dir) if output_dir else data_root
    if output_dir:
        cache_base_dir.mkdir(parents=True, exist_ok=True)

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
    )

    logger.info(f"Creating cache for {data_format} dataset at {data_root}")
    logger.info(f"  Image size: {image_size[0]}x{image_size[1]}")
    logger.info(f"  Encryption: {'enabled' if encrypt else 'disabled'}")
    logger.info(f"  Split: {split}")

    datasets_to_cache = []

    # Setup datasets based on format and split
    if data_format == "yolo":
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
            )
            datasets_to_cache.append(("train", dataset))

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
                cache_refresh=True,
                image_cache=image_cache,
                data_fraction=data_fraction,
                cache_workers=workers,
            )
            datasets_to_cache.append(("val", dataset))

    else:  # COCO format
        coco_annotations_to_save = {}

        if split in ("train", "both") and train_images and train_ann:
            images_dir = data_root / train_images
            ann_file = data_root / train_ann
            if not images_dir.exists():
                raise FileNotFoundError(f"Training images directory not found: {images_dir}")
            if not ann_file.exists():
                raise FileNotFoundError(f"Training annotations not found: {ann_file}")

            dataset = CocoDetectionWrapper(
                root=str(images_dir),
                annFile=str(ann_file),
                transforms=None,
                image_size=image_size,
                image_loader=None,
                image_cache=image_cache,
                data_fraction=data_fraction,
                cache_workers=workers,
            )
            datasets_to_cache.append(("train", dataset))

            # Collect COCO annotations for cache-only mode
            coco_annotations_to_save["train"] = _extract_coco_annotations(dataset)

        if split in ("val", "both") and val_images and val_ann:
            images_dir = data_root / val_images
            ann_file = data_root / val_ann
            if not images_dir.exists():
                raise FileNotFoundError(f"Validation images directory not found: {images_dir}")
            if not ann_file.exists():
                raise FileNotFoundError(f"Validation annotations not found: {ann_file}")

            dataset = CocoDetectionWrapper(
                root=str(images_dir),
                annFile=str(ann_file),
                transforms=None,
                image_size=image_size,
                image_loader=None,
                image_cache=image_cache,
                data_fraction=data_fraction,
                cache_workers=workers,
            )
            datasets_to_cache.append(("val", dataset))

            # Collect COCO annotations for cache-only mode
            coco_annotations_to_save["val"] = _extract_coco_annotations(dataset)

    if not datasets_to_cache:
        raise ValueError(
            f"No datasets to cache. Check that paths are provided for split='{split}' "
            f"and format='{data_format}'."
        )

    # Iterate through datasets to populate cache
    total_images = 0
    for split_name, dataset in datasets_to_cache:
        logger.info(f"Caching {split_name} split ({len(dataset)} images)...")
        for i in range(len(dataset)):
            # Access each item to trigger caching
            try:
                _ = dataset[i]
            except Exception as e:
                logger.warning(f"Failed to cache image {i}: {e}")
            if (i + 1) % 1000 == 0:
                logger.info(f"  Cached {i + 1}/{len(dataset)} images...")
        total_images += len(dataset)

    # Save dataset format to cache metadata (for format verification)
    logger.info(f"Saving dataset format '{data_format}' to cache metadata...")
    image_cache.save_format(data_format)

    # Save COCO annotations to cache (for cache-only mode)
    if data_format == "coco" and coco_annotations_to_save:
        logger.info("Saving COCO annotations to cache metadata...")
        # Merge annotations from all splits
        merged_annotations = {
            "annotations": {},
            "categories": [],
        }
        for split_name, annots in coco_annotations_to_save.items():
            merged_annotations["annotations"].update(annots["annotations"])
            if not merged_annotations["categories"]:
                merged_annotations["categories"] = annots["categories"]
        image_cache.save_coco_annotations(merged_annotations)

    # Finalize cache
    if image_cache._env is not None:
        image_cache.finalize()

    cache_path = image_cache.cache_path
    logger.info(f"Cache created successfully: {cache_path}")
    logger.info(f"  Total images cached: {total_images}")

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
]
