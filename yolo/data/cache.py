"""
Dataset Caching System for YOLO Training.

Provides label caching with hash validation and optional image caching
to accelerate data loading during training.
"""

import hashlib
import pickle
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from yolo.utils.logger import logger


class DatasetCache:
    """
    Cache for dataset labels with hash validation.

    Labels are parsed once and stored in a pickle file. On subsequent runs,
    the cache is validated using a hash of file paths and modification times.
    If files change, the cache is automatically regenerated.

    Attributes:
        VERSION: Cache format version for compatibility checking.
        CACHE_SUFFIX: File extension for cache files.
    """

    VERSION = "1.0.0"
    CACHE_SUFFIX = ".cache"

    def __init__(self, cache_dir: Path, name: str = "labels"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache file.
            name: Base name for cache file (without extension).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_path = self.cache_dir / f"{name}{self.CACHE_SUFFIX}"
        self._data: Optional[Dict] = None

    @staticmethod
    def compute_hash(file_paths: List[Path]) -> str:
        """
        Compute hash of file paths and their metadata.

        Creates a unique hash based on file paths, modification times,
        and file sizes to detect any changes to source files.

        Args:
            file_paths: List of paths to include in hash.

        Returns:
            MD5 hash string.
        """
        hasher = hashlib.md5()
        for path in sorted(file_paths):
            try:
                stat = path.stat()
                hasher.update(f"{path}:{stat.st_mtime}:{stat.st_size}".encode())
            except OSError:
                # File doesn't exist or can't be accessed
                hasher.update(f"{path}:missing".encode())
        return hasher.hexdigest()

    def is_valid(self, file_paths: List[Path]) -> bool:
        """
        Check if existing cache is valid.

        Validates cache by checking:
        1. Cache file exists
        2. Cache version matches current version
        3. Hash of source files matches stored hash

        Args:
            file_paths: List of source file paths to validate against.

        Returns:
            True if cache is valid and can be used.
        """
        if not self.cache_path.exists():
            return False
        try:
            data = self.load()
            if data.get("version") != self.VERSION:
                logger.debug(f"Cache version mismatch: {data.get('version')} != {self.VERSION}")
                return False
            stored_hash = data.get("hash")
            current_hash = self.compute_hash(file_paths)
            if stored_hash != current_hash:
                logger.debug("Cache hash mismatch - source files changed")
                return False
            return True
        except Exception as e:
            logger.debug(f"Cache validation failed: {e}")
            return False

    def load(self) -> Dict:
        """
        Load cache data from disk.

        Returns:
            Dictionary containing cached data.

        Raises:
            FileNotFoundError: If cache file doesn't exist.
            pickle.UnpicklingError: If cache file is corrupted.
        """
        with open(self.cache_path, "rb") as f:
            return pickle.load(f)

    def save(self, labels: List[Dict], file_paths: List[Path], stats: Optional[Dict] = None) -> None:
        """
        Save labels to cache file.

        Args:
            labels: List of label dictionaries to cache.
            file_paths: Source file paths used for hash computation.
            stats: Optional statistics about the cached data.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.VERSION,
            "hash": self.compute_hash(file_paths),
            "labels": labels,
            "stats": stats or {},
            "timestamp": time.time(),
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved label cache: {self.cache_path}")

    def delete(self) -> bool:
        """
        Delete cache file if exists.

        Returns:
            True if cache was deleted, False if it didn't exist.
        """
        if self.cache_path.exists():
            self.cache_path.unlink()
            logger.info(f"Deleted cache: {self.cache_path}")
            return True
        return False

    @property
    def labels(self) -> List[Dict]:
        """Get cached labels, loading from disk if needed."""
        if self._data is None:
            self._data = self.load()
        return self._data.get("labels", [])

    @property
    def stats(self) -> Dict:
        """Get cached statistics."""
        if self._data is None:
            self._data = self.load()
        return self._data.get("stats", {})


class ImageCache:
    """
    Optional image caching for faster data loading.

    Supports two caching modes:
    - RAM: Keep decoded images in memory (fastest, high memory usage)
    - Disk: Save decoded images as .npy files (moderate speedup, persistent)

    For encrypted images, disk cache can also be encrypted using AES-256.

    Attributes:
        mode: Caching mode ('none', 'ram', or 'disk').
        target_size: Target image size (width, height) for resizing, or None for original.
        encrypt_disk_cache: Whether to encrypt disk cache files.
    """

    def __init__(
        self,
        mode: Literal["none", "ram", "disk"] = "none",
        cache_dir: Optional[Path] = None,
        max_memory_gb: float = 8.0,
        target_size: Optional[Tuple[int, int]] = None,
        encryption_key: Optional[str] = None,
    ):
        """
        Initialize image cache.

        Args:
            mode: Caching mode - 'none', 'ram', or 'disk'.
            cache_dir: Directory for disk cache (only used in disk mode).
            max_memory_gb: Maximum RAM to use for caching (only used in ram mode).
            target_size: Target image size (width, height) for resizing. None = keep original size.
            encryption_key: Hex-encoded AES-256 key for encrypting disk cache.
                If provided, disk cache files will be saved as .npy.enc (encrypted).
                If None, disk cache files will be saved as plain .npy files.
        """
        self.mode = mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_gb = max_memory_gb
        self.target_size = target_size
        self._ram_cache: Dict[int, np.ndarray] = {}
        self._enabled = mode != "none"

        # Setup encryption for disk cache
        self._crypto = None
        self._encrypt_disk_cache = False
        if encryption_key is not None and mode == "disk":
            from yolo.data.crypto import CryptoManager
            self._crypto = CryptoManager(key_hex=encryption_key)
            self._encrypt_disk_cache = True
            logger.info("🔒 Disk cache encryption enabled")

    def estimate_memory(
        self,
        paths: List[Path],
        sample_size: int = 50,
        image_loader: Optional[Any] = None,
    ) -> float:
        """
        Estimate memory required to cache all images.

        Samples a subset of images to estimate total memory requirements.
        If target_size is set, estimates based on resized dimensions.

        Args:
            paths: List of image paths.
            sample_size: Number of images to sample for estimation.
            image_loader: Optional custom image loader (e.g., for encrypted images).

        Returns:
            Estimated memory in GB.
        """
        import random
        from PIL import Image

        if not paths:
            return 0.0

        # If target_size is set, we know the exact memory per image
        if self.target_size is not None:
            w, h = self.target_size
            bytes_per_image = w * h * 3  # RGB
            estimated_gb = (bytes_per_image * len(paths) * 1.2) / (1024**3)
            logger.debug(
                f"Memory estimate (resized to {w}x{h}): "
                f"{bytes_per_image/1024/1024:.1f}MB/img, "
                f"total {estimated_gb:.1f}GB for {len(paths)} images"
            )
            return estimated_gb

        # Sample from paths to estimate original image sizes
        sample_count = min(sample_size, len(paths))
        samples = random.sample(list(paths), sample_count)
        total_bytes = 0
        valid_samples = 0

        for path in samples:
            try:
                path_str = str(path)
                # Use custom loader if provided (for encrypted images)
                if image_loader is not None:
                    img = image_loader(path_str)
                    w, h = img.size
                    img.close()
                else:
                    with Image.open(path_str) as img:
                        w, h = img.size
                # Estimate bytes: width * height * channels (assume 3 for RGB)
                total_bytes += w * h * 3
                valid_samples += 1
            except Exception as e:
                logger.debug(f"Failed to sample image {path}: {e}")
                continue

        if valid_samples == 0:
            logger.warning("Could not sample any images for memory estimation")
            return 0.0

        # Average bytes per image * total images * safety margin (1.2x)
        avg_bytes = total_bytes / valid_samples
        estimated_gb = (avg_bytes * len(paths) * 1.2) / (1024**3)

        logger.debug(
            f"Memory estimate: {valid_samples} samples, "
            f"avg {avg_bytes/1024/1024:.1f}MB/img, "
            f"total {estimated_gb:.1f}GB for {len(paths)} images"
        )

        return estimated_gb

    def can_cache_in_ram(self, estimated_gb: float) -> bool:
        """
        Check if there's enough RAM for caching.

        Args:
            estimated_gb: Estimated memory requirement in GB.

        Returns:
            True if caching is feasible.
        """
        try:
            import psutil

            available_gb = psutil.virtual_memory().available / (1024**3)
            # Use at most 80% of available RAM or max_memory_gb, whichever is smaller
            max_allowed = min(self.max_memory_gb, available_gb * 0.8)
            return estimated_gb < max_allowed
        except ImportError:
            logger.warning("psutil not available, assuming RAM caching is feasible")
            return estimated_gb < self.max_memory_gb

    def _get_cache_path(self, path: Path) -> Path:
        """Get cache file path for an image."""
        if self._encrypt_disk_cache:
            return path.with_suffix(".npy.enc")
        return path.with_suffix(".npy")

    def get(self, idx: int, path: Path) -> Optional[np.ndarray]:
        """
        Get cached image if available.

        Args:
            idx: Image index.
            path: Original image path (used for disk cache lookup).

        Returns:
            Cached image array or None if not cached.
        """
        if not self._enabled:
            return None

        if self.mode == "ram":
            return self._ram_cache.get(idx)

        if self.mode == "disk":
            cache_path = self._get_cache_path(path)
            if cache_path.exists():
                try:
                    if self._encrypt_disk_cache:
                        # Load and decrypt
                        with open(cache_path, "rb") as f:
                            encrypted_data = f.read()
                        return self._crypto.decrypt_array(encrypted_data)
                    else:
                        return np.load(cache_path)
                except Exception:
                    return None

        return None

    def put(self, idx: int, path: Path, arr: np.ndarray) -> None:
        """
        Store image in cache.

        Args:
            idx: Image index.
            path: Original image path (used for disk cache path).
            arr: Image array to cache.
        """
        if not self._enabled:
            return

        if self.mode == "ram":
            self._ram_cache[idx] = arr

        elif self.mode == "disk":
            cache_path = self._get_cache_path(path)
            try:
                if self._encrypt_disk_cache:
                    # Encrypt and save
                    encrypted_data = self._crypto.encrypt_array(arr)
                    with open(cache_path, "wb") as f:
                        f.write(encrypted_data)
                else:
                    np.save(cache_path, arr, allow_pickle=False)
            except Exception as e:
                logger.debug(f"Failed to cache image to disk: {e}")

    def clear(self) -> None:
        """Clear RAM cache."""
        self._ram_cache.clear()

    @property
    def size(self) -> int:
        """Get number of cached items (RAM mode only)."""
        return len(self._ram_cache)


class LRUImageBuffer:
    """
    LRU (Least Recently Used) buffer for mosaic augmentation.

    Caches recently accessed image-target pairs to speed up mosaic
    operations that often reuse images from recent batches.

    This is particularly effective for mosaic augmentation where
    multiple images are combined, and the same images may be
    accessed multiple times within a short window.
    """

    def __init__(self, capacity: int = 64):
        """
        Initialize LRU buffer.

        Args:
            capacity: Maximum number of items to cache.
        """
        self.capacity = capacity
        self._cache: OrderedDict = OrderedDict()

    def get(self, key: int) -> Optional[Any]:
        """
        Get item from buffer and mark as recently used.

        Args:
            key: Item key (typically image index).

        Returns:
            Cached item or None if not found.
        """
        if key not in self._cache:
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: int, value: Any) -> None:
        """
        Add item to buffer.

        If buffer is full, removes least recently used item.

        Args:
            key: Item key.
            value: Item to cache.
        """
        # Skip if capacity is 0
        if self.capacity <= 0:
            return

        if key in self._cache:
            # Update existing and mark as recently used
            self._cache.move_to_end(key)
            self._cache[key] = value
            return

        # Remove oldest if at capacity
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)

        self._cache[key] = value

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Get number of cached items."""
        return len(self._cache)

    def __contains__(self, key: int) -> bool:
        """Check if key is in buffer."""
        return key in self._cache

    def __len__(self) -> int:
        """Get number of cached items."""
        return len(self._cache)
