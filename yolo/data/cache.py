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
    Unified image caching for faster data loading using LMDB.

    Supports two caching modes with the same underlying LMDB storage:
    - RAM: Pre-faults all pages into memory for fastest access
    - Disk: Uses OS-managed memory-mapping for lazy loading

    LMDB (Lightning Memory-Mapped Database) provides:
    - Zero-copy reads via memory-mapping
    - Multi-process safe concurrent reads
    - Efficient B+ tree indexing
    - Automatic page management

    Attributes:
        mode: Caching mode ('none', 'ram', or 'disk').
        target_size: Target image size (width, height) for resizing, or None for original.
    """

    # Cache version for compatibility checking
    # v3.0.0: Initial LMDB implementation
    # v3.1.0: Fast hash computation (sampled paths instead of all)
    # v3.2.0: Store original image size for correct bbox transformation with resized cache
    # v4.0.0: Unified cache system with cache_format: jpeg|raw
    CACHE_VERSION = "4.0.0"

    # Dtype mapping for efficient serialization
    DTYPE_TO_ID = {
        np.dtype("uint8"): 0,
        np.dtype("float32"): 1,
        np.dtype("float64"): 2,
        np.dtype("int32"): 3,
        np.dtype("int64"): 4,
    }
    ID_TO_DTYPE = {v: k for k, v in DTYPE_TO_ID.items()}

    def __init__(
        self,
        mode: Literal["none", "ram", "disk"] = "none",
        cache_dir: Optional[Path] = None,
        max_memory_gb: float = 8.0,
        target_size: Optional[Tuple[int, int]] = None,
        encryption_key: Optional[str] = None,
        cache_suffix: Optional[str] = None,
        refresh: bool = False,
        sync: bool = False,
        expected_total_images: Optional[int] = None,
        cache_format: Literal["jpeg", "raw"] = "jpeg",
        jpeg_quality: int = 95,
        compress: bool = False,  # Deprecated: LZ4 is now automatic for raw format
    ):
        """
        Initialize image cache.

        Args:
            mode: Caching mode - 'none', 'ram', or 'disk'.
            cache_dir: Directory for cache storage.
            max_memory_gb: Maximum memory to use for caching.
            target_size: Target image size (width, height) for resizing. None = keep original size.
            encryption_key: Hex-encoded AES-256 key for encrypting cached values.
            cache_suffix: Optional suffix for cache folder (e.g., "640x640_f0.1").
            refresh: Force cache regeneration (delete existing cache).
            sync: Enable LMDB fsync for crash safety. Default False for external volume compatibility.
            expected_total_images: Expected total number of images for map size estimation.
                Useful when multiple datasets will share the same cache.
            cache_format: Cache format - 'jpeg' (compressed, smaller) or 'raw' (lossless).
                JPEG is ~10x smaller and uses TurboJPEG for fast encoding/decoding.
                RAW stores uncompressed numpy arrays with automatic LZ4 compression.
            jpeg_quality: JPEG quality (1-100). Only used when cache_format='jpeg'. Default 95.
            compress: Deprecated. LZ4 compression is now automatic for 'raw' format only.
        """
        self.mode = mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_gb = max_memory_gb
        self.target_size = target_size
        self.cache_suffix = cache_suffix
        self._refresh = refresh
        self._sync = sync
        self._enabled = mode != "none"
        self._expected_total_images = expected_total_images

        # Cache format settings
        self._cache_format = cache_format
        self._jpeg_quality = jpeg_quality

        # LZ4 compression: automatic for RAW format, disabled for JPEG (already compressed)
        self._compress = (cache_format == "raw")

        # LMDB environment
        self._env = None
        self._db_path: Optional[Path] = None
        self._cache_dir_path: Optional[Path] = None
        self._num_images: int = 0
        self._cached_indices: set = set()
        self._initialized = False

        # Setup encryption (key is NEVER stored - read from env var in workers)
        self._crypto = None
        self._encrypt_cache = False
        if encryption_key is not None and mode != "none":
            from yolo.data.crypto import CryptoManager
            self._crypto = CryptoManager(key_hex=encryption_key)
            self._encrypt_cache = True
            logger.info("Disk cache encryption enabled")

        # Setup TurboJPEG for JPEG format (fast JPEG encoding/decoding)
        self._turbojpeg = None
        self._tjpf_rgb = None
        if cache_format == "jpeg" and mode != "none":
            try:
                from turbojpeg import TurboJPEG, TJPF_RGB
                self._turbojpeg = TurboJPEG()
                self._tjpf_rgb = TJPF_RGB
                logger.info(f"Disk cache JPEG format enabled (quality={jpeg_quality})")
            except ImportError:
                raise ImportError(
                    "JPEG cache format requested but PyTurboJPEG not installed. "
                    "Install with: pip install PyTurboJPEG\n"
                    "Also requires libjpeg-turbo system library:\n"
                    "  macOS: brew install libjpeg-turbo\n"
                    "  Ubuntu: sudo apt-get install libturbojpeg"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize TurboJPEG: {e}\n"
                    "Make sure libjpeg-turbo is installed:\n"
                    "  macOS: brew install libjpeg-turbo\n"
                    "  Ubuntu: sudo apt-get install libturbojpeg"
                )

        # Setup LZ4 for RAW format
        if self._compress and mode != "none":
            try:
                import lz4.frame
                logger.info("Disk cache LZ4 compression enabled (RAW format)")
            except ImportError:
                raise ImportError(
                    "LZ4 compression required for RAW format but lz4 not installed. "
                    "Install with: pip install lz4"
                )

    def _build_cache_dir(self, base_dir: Path) -> Path:
        """Build cache directory path with optional suffix."""
        cache_folder_name = ".yolo_cache"
        if self.cache_suffix:
            cache_folder_name = f".yolo_cache_{self.cache_suffix}"

        if self.cache_dir is not None:
            return Path(self.cache_dir) / cache_folder_name
        return base_dir / cache_folder_name

    def initialize(
        self,
        num_images: int,
        cache_dir: Path,
        paths: List[Path],
    ) -> bool:
        """
        Initialize LMDB cache.

        Creates or opens an LMDB database for caching images.
        Both RAM and Disk modes use the same LMDB format.

        Args:
            num_images: Number of images to cache
            cache_dir: Directory for cache storage
            paths: Image paths (for validation)

        Returns:
            True if cache exists and is valid (reuse), False if needs building
        """
        if self.mode == "none":
            return False

        import lmdb

        self._num_images = num_images
        self._cache_dir_path = self._build_cache_dir(cache_dir)
        self._cache_dir_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self._cache_dir_path / "cache.lmdb"

        # Estimate map size (LMDB requires pre-allocation)
        map_size = self._estimate_map_size(num_images, paths)

        # Check memory limit for RAM mode
        map_size_gb = map_size / (1024**3)
        if self.mode == "ram" and map_size_gb > self.max_memory_gb:
            logger.warning(
                f"Cache would use {map_size_gb:.1f}GB, exceeding limit of {self.max_memory_gb:.1f}GB. "
                f"Disabling cache."
            )
            self._enabled = False
            return False

        # Force refresh: delete existing cache
        if self._refresh and self._db_path.exists():
            import shutil
            shutil.rmtree(self._db_path)
            self._refresh_requested = True
        else:
            self._refresh_requested = False

        # Check if valid cache exists (skip if refresh requested)
        if not self._refresh and self._validate_existing_cache(paths):
            self._open_db(readonly=True)
            self._load_cached_indices()
            self._initialized = True
            return True

        # Create new cache - clear old one first
        if self._db_path.exists():
            import shutil
            shutil.rmtree(self._db_path)

        self._open_db(readonly=False, map_size=map_size)
        self._save_metadata(paths)
        self._cached_indices.clear()
        self._initialized = True
        return False

    def _estimate_map_size(self, num_images: int, paths: List[Path]) -> int:
        """Estimate LMDB map size based on image count and sizes."""
        # Use expected_total_images if provided (for multi-dataset caches)
        effective_num_images = self._expected_total_images or num_images

        if self.target_size is not None:
            w, h = self.target_size
            bytes_per_image = w * h * 3 + 32  # RGB + header overhead
        else:
            # Estimate from sample or use default
            bytes_per_image = 640 * 640 * 3 + 32  # Default estimate

        # Add 50% margin for LMDB overhead and growth
        total_size = int(effective_num_images * bytes_per_image * 1.5)

        # Minimum 100MB, maximum based on available space
        return max(100 * 1024 * 1024, total_size)

    def _validate_existing_cache(self, paths: List[Path]) -> bool:
        """Check if existing cache is valid. Sets _invalidation_reason if invalid."""
        self._invalidation_reason: Optional[str] = None

        if not self._db_path or not self._db_path.exists():
            self._invalidation_reason = "cache not found"
            return False

        try:
            import lmdb

            env = lmdb.open(
                str(self._db_path),
                readonly=True,
                lock=False,
                readahead=False,
            )

            with env.begin() as txn:
                # Check metadata
                meta_data = txn.get(b"__metadata__")
                if meta_data is None:
                    env.close()
                    self._invalidation_reason = "missing metadata"
                    return False

                meta = pickle.loads(meta_data)

                # Validate version
                if meta.get("version") != self.CACHE_VERSION:
                    old_version = meta.get("version", "unknown")
                    self._invalidation_reason = f"version changed ({old_version} → {self.CACHE_VERSION})"
                    env.close()
                    return False

                # Validate suffix
                if meta.get("cache_suffix") != self.cache_suffix:
                    self._invalidation_reason = "settings changed (size/fraction)"
                    env.close()
                    return False

                # Validate encryption setting
                cached_encrypted = meta.get("encrypted", False)
                if cached_encrypted != self._encrypt_cache:
                    old_enc = "encrypted" if cached_encrypted else "unencrypted"
                    new_enc = "encrypted" if self._encrypt_cache else "unencrypted"
                    self._invalidation_reason = f"encryption changed ({old_enc} → {new_enc})"
                    env.close()
                    return False

                # Validate cache format (jpeg vs raw)
                cached_format = meta.get("cache_format", "raw")  # Default raw for old caches
                if cached_format != self._cache_format:
                    self._invalidation_reason = f"format changed ({cached_format} → {self._cache_format})"
                    env.close()
                    return False

                # Validate number of images
                cached_count = meta.get("num_images", 0)
                if cached_count != len(paths):
                    self._invalidation_reason = f"image count changed ({cached_count:,} → {len(paths):,})"
                    env.close()
                    return False

                # Validate path hash
                expected_hash = self._compute_paths_hash(paths)
                if meta.get("paths_hash") != expected_hash:
                    self._invalidation_reason = "image files changed"
                    env.close()
                    return False

            env.close()
            return True

        except Exception as e:
            logger.debug(f"Cache validation failed: {e}")
            return False

    def _compute_paths_hash(self, paths: List[Path]) -> str:
        """
        Compute hash of image paths for validation.

        Uses a fast approach: hash count + first/last paths + sampled paths.
        This avoids iterating through all 100k+ paths while still detecting changes.
        """
        hasher = hashlib.md5()

        # Include count (detects additions/removals)
        hasher.update(f"count:{len(paths)}".encode())

        if not paths:
            return hasher.hexdigest()

        # Sort once for consistent ordering
        sorted_paths = sorted(paths)

        # Hash first and last paths (detects boundary changes)
        hasher.update(str(sorted_paths[0]).encode())
        hasher.update(str(sorted_paths[-1]).encode())

        # Sample ~100 paths evenly distributed (detects changes in middle)
        n_samples = min(100, len(sorted_paths))
        step = max(1, len(sorted_paths) // n_samples)
        for i in range(0, len(sorted_paths), step):
            hasher.update(str(sorted_paths[i]).encode())

        return hasher.hexdigest()

    def _open_db(self, readonly: bool, map_size: int = 0):
        """Open LMDB environment."""
        import lmdb

        self._env = lmdb.open(
            str(self._db_path),
            map_size=map_size if not readonly else 0,
            readonly=readonly,
            lock=not readonly,
            readahead=self.mode == "disk",  # Disk mode benefits from readahead
            meminit=False,  # Don't zero-init for performance
            max_dbs=0,
            sync=self._sync,  # Disable with sync=False for external volumes (macOS)
            metasync=self._sync,  # Disable with sync=False for external volumes
        )

        # RAM mode: pre-fault pages into memory
        if self.mode == "ram" and not readonly:
            self._prefault_pages()

    def _prefault_pages(self):
        """Pre-fault all pages into RAM for faster access."""
        if self._env is None:
            return

        try:
            # Read through all data to bring into page cache
            with self._env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    pass  # Just iterate to fault pages
            logger.debug("Pre-faulted cache pages into RAM")
        except Exception as e:
            logger.debug(f"Failed to pre-fault pages: {e}")

    def _save_metadata(
        self,
        paths: List[Path],
        labels: Optional[List[Dict]] = None,
        coco_annotations: Optional[Dict] = None,
    ) -> None:
        """
        Save cache metadata including image paths for cache-only mode.

        Args:
            paths: List of image paths.
            labels: Optional list of label dictionaries for YOLO format cache-only mode.
            coco_annotations: Optional COCO annotations dict for COCO format cache-only mode.
                Should contain 'annotations' (dict mapping index to annotation list)
                and 'categories' (list of category dicts).
        """
        if self._env is None:
            return

        # Store paths as strings for portability
        paths_list = [str(p) for p in paths]

        meta = {
            "version": self.CACHE_VERSION,
            "cache_suffix": self.cache_suffix,
            "num_images": len(paths),
            "paths_hash": self._compute_paths_hash(paths),
            "target_size": self.target_size,
            "encrypted": self._encrypt_cache,
            "cache_format": self._cache_format,  # 'jpeg' or 'raw'
            "jpeg_quality": self._jpeg_quality if self._cache_format == "jpeg" else None,
            "compressed": self._compress,  # LZ4 compression (auto for raw)
            "image_paths": paths_list,  # Store paths for cache-only mode
        }

        # Store labels if provided (for YOLO format cache-only mode)
        if labels is not None:
            meta["labels"] = labels

        # Store COCO annotations if provided (for COCO format cache-only mode)
        if coco_annotations is not None:
            meta["coco_annotations"] = coco_annotations

        with self._env.begin(write=True) as txn:
            txn.put(b"__metadata__", pickle.dumps(meta))

    def _load_cached_indices(self) -> None:
        """Load set of cached image indices from LMDB."""
        if self._env is None:
            return

        self._cached_indices.clear()
        with self._env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                if key != b"__metadata__":
                    try:
                        idx = int(key.decode())
                        self._cached_indices.add(idx)
                    except (ValueError, UnicodeDecodeError):
                        pass

    def finalize(self) -> None:
        """Finalize cache after all images have been written."""
        if self._env is not None:
            # Attempt sync for data consistency if enabled
            # Ignore errors on external volumes (macOS) - cache is derived data
            if self._sync:
                try:
                    self._env.sync()
                except Exception as e:
                    logger.debug(f"LMDB sync skipped (safe for cache data): {e}")

            # Re-open as readonly for all modes (safer for concurrent reads)
            self._env.close()
            self._open_db(readonly=True)
            self._load_cached_indices()

            # For RAM mode, prefault pages into memory
            if self.mode == "ram":
                self._prefault_pages()

    @property
    def cache_path(self) -> Optional[Path]:
        """Get the cache directory path."""
        return self._cache_dir_path

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get cache metadata including image paths and labels.

        This is useful for cache-only mode where the original images
        are not available and all data must come from the cache.

        Returns:
            Dictionary containing:
            - version: Cache format version
            - num_images: Number of cached images
            - image_paths: List of original image paths (strings)
            - labels: List of label dictionaries (if stored)
            - target_size: Image size tuple or None
            - encrypted: Whether cache is encrypted
            Or None if cache is not initialized.
        """
        if self._env is None:
            return None

        try:
            with self._env.begin() as txn:
                meta_data = txn.get(b"__metadata__")
                if meta_data is None:
                    return None
                return pickle.loads(meta_data)
        except Exception as e:
            logger.debug(f"Failed to read cache metadata: {e}")
            return None

    def get_image_paths(self) -> Optional[List[str]]:
        """
        Get list of image paths stored in cache metadata.

        This enables cache-only mode where images are loaded
        from cache without needing the original files.

        Returns:
            List of image path strings, or None if not available.
        """
        meta = self.get_metadata()
        if meta is None:
            return None
        return meta.get("image_paths")

    def get_labels(self) -> Optional[List[Dict]]:
        """
        Get list of labels stored in cache metadata.

        This enables cache-only mode where labels are loaded
        from cache without needing the original label files.

        Returns:
            List of label dictionaries, or None if not available.
        """
        meta = self.get_metadata()
        if meta is None:
            return None
        return meta.get("labels")

    def save_labels(self, labels: List[Dict]) -> None:
        """
        Save labels to cache metadata for YOLO format cache-only mode.

        Args:
            labels: List of label dictionaries to store.
        """
        if self._env is None:
            return

        meta = self.get_metadata() or {}
        meta["labels"] = labels

        with self._env.begin(write=True) as txn:
            txn.put(b"__metadata__", pickle.dumps(meta))

    def save_split_indices(self, train_indices: List[int], val_indices: List[int]) -> None:
        """
        Save train/val split indices to cache metadata.

        This enables cache-only mode where split information is loaded
        from cache without needing the original train.txt/val.txt files.

        Args:
            train_indices: List of indices for training split.
            val_indices: List of indices for validation split.
        """
        if self._env is None:
            return

        meta = self.get_metadata() or {}
        meta["train_indices"] = train_indices
        meta["val_indices"] = val_indices

        with self._env.begin(write=True) as txn:
            txn.put(b"__metadata__", pickle.dumps(meta))

    def get_split_indices(self) -> Optional[Dict[str, List[int]]]:
        """
        Get train/val split indices from cache metadata.

        This enables cache-only mode where split information is loaded
        from cache without needing the original train.txt/val.txt files.

        Returns:
            Dict with 'train' and 'val' keys containing index lists, or None
            if split indices are not available in the cache.
        """
        meta = self.get_metadata()
        if meta is None:
            return None

        train_indices = meta.get("train_indices")
        val_indices = meta.get("val_indices")

        if train_indices is None or val_indices is None:
            return None

        return {"train": train_indices, "val": val_indices}

    def save_coco_annotations(self, coco_annotations: Dict) -> None:
        """
        Save COCO annotations to cache metadata for COCO format cache-only mode.

        Args:
            coco_annotations: Dictionary containing:
                - 'annotations': Dict mapping image index to annotation list
                - 'categories': List of category dictionaries
        """
        if self._env is None:
            return

        meta = self.get_metadata() or {}
        meta["coco_annotations"] = coco_annotations

        with self._env.begin(write=True) as txn:
            txn.put(b"__metadata__", pickle.dumps(meta))

    def save_format(self, data_format: str) -> None:
        """
        Save dataset format to cache metadata.

        Args:
            data_format: Dataset format ('coco' or 'yolo').
        """
        if self._env is None:
            return

        meta = self.get_metadata() or {}
        meta["format"] = data_format

        with self._env.begin(write=True) as txn:
            txn.put(b"__metadata__", pickle.dumps(meta))

    def get_format(self) -> Optional[str]:
        """
        Get dataset format stored in cache metadata.

        Returns:
            Format string ('coco' or 'yolo'), or None if not available.
        """
        meta = self.get_metadata()
        if meta is None:
            return None
        return meta.get("format")

    def sanity_check(self) -> Dict[str, Any]:
        """
        Perform sanity check on cache integrity.

        Verifies that the cache is complete and consistent:
        - Metadata exists and is readable
        - Number of cached images matches metadata
        - Labels exist (for YOLO format)
        - Image size matches metadata
        - Encryption status is consistent

        Returns:
            Dictionary with check results:
            - 'valid': bool - True if all checks pass
            - 'errors': List[str] - List of error messages
            - 'warnings': List[str] - List of warning messages
            - 'stats': Dict - Cache statistics (num_images, format, etc.)
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        # Check if cache is initialized
        if self._env is None:
            result["valid"] = False
            result["errors"].append("Cache not initialized (LMDB environment is None)")
            return result

        # Get metadata
        meta = self.get_metadata()
        if meta is None:
            result["valid"] = False
            result["errors"].append("Cache metadata not found or unreadable")
            return result

        # Extract stats
        result["stats"]["format"] = meta.get("format", "unknown")
        result["stats"]["num_images_metadata"] = meta.get("num_images", 0)
        result["stats"]["target_size"] = meta.get("target_size")
        result["stats"]["encrypted"] = meta.get("encrypted", False)
        result["stats"]["version"] = meta.get("version", "unknown")

        # Check cached images count
        num_cached = len(self._cached_indices)
        num_expected = meta.get("num_images", 0)
        result["stats"]["num_images_cached"] = num_cached

        if num_cached != num_expected:
            result["valid"] = False
            result["errors"].append(
                f"Image count mismatch: cached {num_cached}, expected {num_expected}"
            )

        # Check image_paths exist
        image_paths = meta.get("image_paths")
        if image_paths is None:
            result["valid"] = False
            result["errors"].append("image_paths not found in metadata (required for cache-only mode)")
        else:
            result["stats"]["num_image_paths"] = len(image_paths)
            if len(image_paths) != num_expected:
                result["valid"] = False
                result["errors"].append(
                    f"image_paths count mismatch: {len(image_paths)}, expected {num_expected}"
                )

        # Check format-specific data
        data_format = meta.get("format")
        if data_format == "yolo":
            labels = meta.get("labels")
            if labels is None:
                result["valid"] = False
                result["errors"].append("YOLO labels not found in metadata")
            else:
                result["stats"]["num_labels"] = len(labels)
                if len(labels) != num_expected:
                    result["valid"] = False
                    result["errors"].append(
                        f"Labels count mismatch: {len(labels)}, expected {num_expected}"
                    )

        elif data_format == "coco":
            coco_ann = meta.get("coco_annotations")
            if coco_ann is None:
                result["valid"] = False
                result["errors"].append("COCO annotations not found in metadata")
            else:
                annotations = coco_ann.get("annotations", {})
                categories = coco_ann.get("categories", [])
                result["stats"]["num_annotations"] = len(annotations)
                result["stats"]["num_categories"] = len(categories)

        # Check split indices (optional but recommended)
        train_indices = meta.get("train_indices")
        val_indices = meta.get("val_indices")
        if train_indices is not None and val_indices is not None:
            result["stats"]["num_train_indices"] = len(train_indices)
            result["stats"]["num_val_indices"] = len(val_indices)
        else:
            result["warnings"].append(
                "Split indices not found in metadata (will need split files for training)"
            )

        # Check encryption consistency
        if meta.get("encrypted", False) and not self._encrypt_cache:
            result["valid"] = False
            result["errors"].append(
                "Cache is encrypted but cache_encrypt=False in config"
            )

        return result

    def get_coco_annotations(self) -> Optional[Dict]:
        """
        Get COCO annotations stored in cache metadata.

        Returns:
            Dictionary with 'annotations' and 'categories', or None if not available.
        """
        meta = self.get_metadata()
        if meta is None:
            return None
        return meta.get("coco_annotations")

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for pickling (spawn multiprocessing)."""
        state = self.__dict__.copy()
        # Don't serialize LMDB env - workers will re-open it
        state["_env"] = None
        state["_crypto"] = None  # Will be re-initialized if needed
        state["_turbojpeg"] = None  # Will be re-initialized if needed
        state["_tjpf_rgb"] = None
        # Don't serialize cached_indices - will be reloaded from LMDB in worker
        state["_cached_indices"] = set()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after unpickling."""
        import os
        self.__dict__.update(state)

        # Re-open LMDB in read-only mode for workers
        if self._initialized and self._db_path is not None and self._db_path.exists():
            try:
                self._open_db(readonly=True)
                # Reload cached indices from LMDB
                self._load_cached_indices()
            except Exception as e:
                logger.warning(f"Failed to open cache in worker: {e}")
                self._env = None

        # Re-initialize crypto from environment variable (key is NEVER pickled)
        if self._encrypt_cache:
            encryption_key = os.environ.get("YOLO_ENCRYPTION_KEY")
            if encryption_key:
                from yolo.data.crypto import CryptoManager
                self._crypto = CryptoManager(key_hex=encryption_key)
            else:
                logger.warning(
                    "Encrypted cache but YOLO_ENCRYPTION_KEY not set in worker. "
                    "Cache reads will fail."
                )

        # Re-initialize TurboJPEG for JPEG format
        if self._cache_format == "jpeg":
            try:
                from turbojpeg import TurboJPEG, TJPF_RGB
                self._turbojpeg = TurboJPEG()
                self._tjpf_rgb = TJPF_RGB
            except Exception as e:
                logger.warning(f"Failed to initialize TurboJPEG in worker: {e}")

    def _serialize(self, arr: np.ndarray, orig_size: Optional[Tuple[int, int]] = None) -> bytes:
        """
        Serialize numpy array to bytes.

        Dispatches to JPEG or RAW serialization based on cache_format setting.

        Args:
            arr: Image array to serialize (RGB, uint8)
            orig_size: Original image size (width, height) before resize, or None if not resized

        Returns:
            Serialized bytes
        """
        if self._cache_format == "jpeg":
            return self._serialize_jpeg(arr, orig_size)
        else:
            return self._serialize_raw(arr, orig_size)

    def _deserialize(self, data: bytes) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """
        Deserialize bytes to numpy array and original size.

        Dispatches to JPEG or RAW deserialization based on cache_format setting.

        Returns:
            Tuple of (image array, original size or None)
        """
        if self._cache_format == "jpeg":
            return self._deserialize_jpeg(data)
        else:
            return self._deserialize_raw(data)

    def _serialize_jpeg(self, arr: np.ndarray, orig_size: Optional[Tuple[int, int]] = None) -> bytes:
        """
        Serialize image as JPEG bytes using TurboJPEG (5-10x faster than PIL).

        Format:
        - Header: orig_w (4B) + orig_h (4B) - only if orig_size provided
        - JPEG data

        Args:
            arr: Image array (RGB, uint8, shape HxWx3)
            orig_size: Original image size (width, height) before resize

        Returns:
            Serialized bytes with optional header + JPEG data
        """
        import struct

        # TurboJPEG encode - expects numpy array in RGB format
        jpeg_bytes = self._turbojpeg.encode(
            arr,
            quality=self._jpeg_quality,
            pixel_format=self._tjpf_rgb
        )

        # Header: has_orig_size (1B) + orig_w (4B) + orig_h (4B) if present
        has_orig_size = 1 if orig_size is not None else 0
        header = struct.pack("<B", has_orig_size)
        if orig_size is not None:
            header += struct.pack("<II", orig_size[0], orig_size[1])

        data = header + jpeg_bytes

        # NO LZ4 for JPEG (already compressed, LZ4 would be counterproductive)
        # Only Encryption
        if self._encrypt_cache and self._crypto is not None:
            data = self._crypto.encrypt(data)

        return data

    def _deserialize_jpeg(self, data: bytes) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """
        Deserialize JPEG bytes to numpy array using TurboJPEG.

        Returns:
            Tuple of (image array RGB, original size or None)
        """
        import struct

        # Decryption only (no LZ4 for JPEG)
        if self._encrypt_cache and self._crypto is not None:
            data = self._crypto.decrypt(data)

        # Parse header
        has_orig_size = struct.unpack("<B", data[:1])[0]
        header_end = 1
        orig_size = None
        if has_orig_size:
            orig_w, orig_h = struct.unpack("<II", data[1:9])
            orig_size = (orig_w, orig_h)
            header_end = 9

        jpeg_bytes = data[header_end:]

        # TurboJPEG decode - returns numpy array in RGB format
        arr = self._turbojpeg.decode(jpeg_bytes, pixel_format=self._tjpf_rgb)

        return arr, orig_size

    def _serialize_raw(self, arr: np.ndarray, orig_size: Optional[Tuple[int, int]] = None) -> bytes:
        """
        Serialize numpy array to bytes with compact header (RAW format with LZ4).

        Args:
            arr: Image array to serialize
            orig_size: Original image size (width, height) before resize, or None if not resized

        Returns:
            Serialized bytes with header containing dtype, shape, and orig_size
        """
        import struct

        # Header format (little-endian):
        # - dtype_id: 1 byte
        # - ndim: 1 byte
        # - has_orig_size: 1 byte (0 or 1)
        # - shape: 4 bytes * ndim
        # - orig_size (if has_orig_size): 4 bytes * 2 (orig_w, orig_h)
        dtype_id = self.DTYPE_TO_ID.get(arr.dtype, 0)
        has_orig_size = 1 if orig_size is not None else 0

        header = struct.pack(f"<BBB{arr.ndim}I", dtype_id, arr.ndim, has_orig_size, *arr.shape)
        if orig_size is not None:
            header += struct.pack("<II", orig_size[0], orig_size[1])

        data = header + arr.tobytes()

        # LZ4 compression (automatic for RAW format, before encryption)
        if self._compress:
            import lz4.frame
            data = lz4.frame.compress(data)

        # Encryption (after compression for better ratio)
        if self._encrypt_cache and self._crypto is not None:
            data = self._crypto.encrypt(data)

        return data

    def _deserialize_raw(self, data: bytes) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """
        Deserialize RAW bytes to numpy array and original size.

        Returns:
            Tuple of (image array, original size or None)
        """
        import struct

        # Decryption (first)
        if self._encrypt_cache and self._crypto is not None:
            data = self._crypto.decrypt(data)

        # LZ4 decompression (after decryption, automatic for RAW format)
        if self._compress:
            import lz4.frame
            data = lz4.frame.decompress(data)

        # Parse header (little-endian)
        dtype_id, ndim, has_orig_size = struct.unpack("<BBB", data[:3])
        shape = struct.unpack(f"<{ndim}I", data[3:3 + 4 * ndim])
        dtype = self.ID_TO_DTYPE.get(dtype_id, np.dtype("uint8"))

        # Parse orig_size if present
        header_end = 3 + 4 * ndim
        orig_size = None
        if has_orig_size:
            orig_w, orig_h = struct.unpack("<II", data[header_end:header_end + 8])
            orig_size = (orig_w, orig_h)
            header_end += 8

        # Extract array data
        arr_data = data[header_end:]
        arr = np.frombuffer(arr_data, dtype=dtype).reshape(shape)

        return arr, orig_size

    def get(self, idx: int) -> Optional[Tuple[np.ndarray, Optional[Tuple[int, int]]]]:
        """
        Get cached image by index.

        Args:
            idx: Image index.

        Returns:
            Tuple of (image array, original size) or None if not cached.
            Original size is (width, height) before resize, or None if not resized.
        """
        if not self._enabled or self._env is None:
            return None

        if idx not in self._cached_indices:
            return None

        try:
            key = str(idx).encode()
            with self._env.begin(buffers=True) as txn:
                value = txn.get(key)
                if value is None:
                    return None
                return self._deserialize(bytes(value))
        except Exception as e:
            logger.debug(f"Failed to read from cache: {e}")
            return None

    def put(self, idx: int, arr: np.ndarray, orig_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Store image in cache.

        Args:
            idx: Image index.
            arr: Image array to cache.
            orig_size: Original image size (width, height) before resize, or None if not resized.
        """
        if not self._enabled or self._env is None:
            return

        if idx in self._cached_indices:
            return  # Already cached

        try:
            key = str(idx).encode()
            value = self._serialize(arr, orig_size)

            with self._env.begin(write=True) as txn:
                txn.put(key, value)

            self._cached_indices.add(idx)
        except Exception as e:
            logger.debug(f"Failed to write to cache: {e}")

    def is_cached(self, idx: int) -> bool:
        """Check if an image is cached."""
        return idx in self._cached_indices

    def estimate_memory(
        self,
        paths: List[Path],
        sample_size: int = 50,
        image_loader: Optional[Any] = None,
    ) -> float:
        """
        Estimate memory required to cache all images.

        Args:
            paths: List of image paths.
            sample_size: Number of images to sample for estimation.
            image_loader: Optional custom image loader.

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
            return estimated_gb

        # Sample from paths to estimate original image sizes
        sample_count = min(sample_size, len(paths))
        samples = random.sample(list(paths), sample_count)
        total_bytes = 0
        valid_samples = 0

        for path in samples:
            try:
                path_str = str(path)
                if image_loader is not None:
                    img = image_loader(path_str)
                    w, h = img.size
                    img.close()
                else:
                    with Image.open(path_str) as img:
                        w, h = img.size
                total_bytes += w * h * 3
                valid_samples += 1
            except Exception:
                continue

        if valid_samples == 0:
            return 0.0

        avg_bytes = total_bytes / valid_samples
        estimated_gb = (avg_bytes * len(paths) * 1.2) / (1024**3)
        return estimated_gb

    def can_cache_in_ram(self, estimated_gb: float) -> bool:
        """Check if there's enough RAM for caching."""
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            max_allowed = min(self.max_memory_gb, available_gb * 0.8)
            return estimated_gb < max_allowed
        except ImportError:
            return estimated_gb < self.max_memory_gb

    def clear(self) -> None:
        """Clear cache and remove cache files."""
        self._cached_indices.clear()

        if self._env is not None:
            self._env.close()
            self._env = None

        if self._db_path is not None and self._db_path.exists():
            import shutil
            try:
                shutil.rmtree(self._db_path)
            except Exception:
                pass

    @property
    def size(self) -> int:
        """Get number of cached items."""
        return len(self._cached_indices)


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
