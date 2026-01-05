"""
Tests for Dataset Caching System.

Tests cover:
- DatasetCache: Hash computation, validation, save/load
- ImageCache: Unified LMDB-based RAM and disk caching
- LRUImageBuffer: Capacity, eviction, LRU ordering
"""

import pickle
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from yolo.data.cache import DatasetCache, ImageCache, LRUImageBuffer


class TestDatasetCache:
    """Tests for DatasetCache class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for cache tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_files(self, temp_dir):
        """Create sample files for testing."""
        files = []
        for i in range(3):
            path = temp_dir / f"file_{i}.txt"
            path.write_text(f"content {i}")
            files.append(path)
        return files

    def test_hash_consistent(self, sample_files):
        """Test that hash is consistent for same files."""
        hash1 = DatasetCache.compute_hash(sample_files)
        hash2 = DatasetCache.compute_hash(sample_files)
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length

    def test_hash_changes_on_modification(self, sample_files, temp_dir):
        """Test that hash changes when files are modified."""
        hash1 = DatasetCache.compute_hash(sample_files)

        # Modify a file
        time.sleep(0.01)  # Ensure different mtime
        sample_files[0].write_text("modified content")

        hash2 = DatasetCache.compute_hash(sample_files)
        assert hash1 != hash2

    def test_hash_changes_on_file_addition(self, sample_files, temp_dir):
        """Test that hash changes when files are added."""
        hash1 = DatasetCache.compute_hash(sample_files)

        # Add new file
        new_file = temp_dir / "file_new.txt"
        new_file.write_text("new content")
        sample_files.append(new_file)

        hash2 = DatasetCache.compute_hash(sample_files)
        assert hash1 != hash2

    def test_hash_order_independent(self, sample_files):
        """Test that hash is independent of file order."""
        hash1 = DatasetCache.compute_hash(sample_files)
        hash2 = DatasetCache.compute_hash(list(reversed(sample_files)))
        assert hash1 == hash2  # Should be same because we sort internally

    def test_save_load_cycle(self, temp_dir, sample_files):
        """Test saving and loading cache."""
        cache = DatasetCache(temp_dir, "test")
        labels = [{"boxes_norm": [[0.5, 0.5, 0.1, 0.1]], "labels": [0]}]
        stats = {"count": 1}

        cache.save(labels, sample_files, stats)
        assert cache.cache_path.exists()

        # Load and verify
        loaded = cache.load()
        assert loaded["version"] == DatasetCache.VERSION
        assert loaded["labels"] == labels
        assert loaded["stats"] == stats
        assert "hash" in loaded
        assert "timestamp" in loaded

    def test_validation_passes_unchanged(self, temp_dir, sample_files):
        """Test cache validation passes when files unchanged."""
        cache = DatasetCache(temp_dir, "test")
        labels = [{"boxes_norm": [], "labels": []}]

        cache.save(labels, sample_files, {})
        assert cache.is_valid(sample_files) is True

    def test_validation_fails_on_file_change(self, temp_dir, sample_files):
        """Test cache validation fails when files change."""
        cache = DatasetCache(temp_dir, "test")
        labels = [{"boxes_norm": [], "labels": []}]

        cache.save(labels, sample_files, {})

        # Modify a file
        time.sleep(0.01)
        sample_files[0].write_text("changed content")

        assert cache.is_valid(sample_files) is False

    def test_validation_fails_nonexistent(self, temp_dir, sample_files):
        """Test cache validation fails when cache doesn't exist."""
        cache = DatasetCache(temp_dir, "nonexistent")
        assert cache.is_valid(sample_files) is False

    def test_validation_fails_version_mismatch(self, temp_dir, sample_files):
        """Test cache validation fails on version mismatch."""
        cache = DatasetCache(temp_dir, "test")

        # Manually create cache with wrong version
        data = {
            "version": "0.0.0",
            "hash": DatasetCache.compute_hash(sample_files),
            "labels": [],
            "stats": {},
            "timestamp": time.time(),
        }
        with open(cache.cache_path, "wb") as f:
            pickle.dump(data, f)

        assert cache.is_valid(sample_files) is False

    def test_delete_cache(self, temp_dir, sample_files):
        """Test cache deletion."""
        cache = DatasetCache(temp_dir, "test")
        cache.save([], sample_files, {})

        assert cache.cache_path.exists()
        result = cache.delete()
        assert result is True
        assert not cache.cache_path.exists()

        # Delete non-existent cache
        result = cache.delete()
        assert result is False

    def test_labels_property(self, temp_dir, sample_files):
        """Test labels property loads data lazily."""
        cache = DatasetCache(temp_dir, "test")
        labels = [{"boxes_norm": [[0.1, 0.2, 0.3, 0.4]], "labels": [1, 2]}]
        cache.save(labels, sample_files, {})

        # Create new cache instance to test lazy loading
        cache2 = DatasetCache(temp_dir, "test")
        assert cache2._data is None
        loaded_labels = cache2.labels
        assert loaded_labels == labels
        assert cache2._data is not None

    def test_stats_property(self, temp_dir, sample_files):
        """Test stats property."""
        cache = DatasetCache(temp_dir, "test")
        stats = {"count": 10, "total_boxes": 50}
        cache.save([], sample_files, stats)

        cache2 = DatasetCache(temp_dir, "test")
        assert cache2.stats == stats

    def test_handles_missing_files_in_hash(self, temp_dir):
        """Test hash computation handles missing files gracefully."""
        paths = [temp_dir / "missing1.txt", temp_dir / "missing2.txt"]
        # Should not raise, should produce consistent hash
        hash1 = DatasetCache.compute_hash(paths)
        hash2 = DatasetCache.compute_hash(paths)
        assert hash1 == hash2


class TestImageCache:
    """Tests for unified LMDB ImageCache class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for cache tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_paths(self, temp_dir):
        """Create sample image paths for testing."""
        return [temp_dir / f"image_{i}.jpg" for i in range(10)]

    def test_none_mode_disabled(self):
        """Test that none mode disables caching."""
        cache = ImageCache(mode="none")
        assert cache._enabled is False
        assert cache.get(0) is None

        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        cache.put(0, arr)
        assert cache.get(0) is None

    def test_ram_cache_initialize_and_put_get(self, temp_dir, sample_paths):
        """Test RAM caching with unified LMDB backend."""
        cache = ImageCache(mode="ram", target_size=(64, 64))

        # Initialize cache
        cache_exists = cache.initialize(
            num_images=len(sample_paths),
            cache_dir=temp_dir,
            paths=sample_paths,
        )
        assert cache_exists is False  # New cache
        assert cache._enabled is True

        # Store and retrieve images
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cache.put(0, arr)
        cache.finalize()

        retrieved = cache.get(0)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, arr)

        # Cleanup
        cache.clear()

    def test_disk_cache_initialize_and_put_get(self, temp_dir, sample_paths):
        """Test disk caching with unified LMDB backend."""
        cache = ImageCache(mode="disk", target_size=(64, 64))

        # Initialize cache
        cache_exists = cache.initialize(
            num_images=len(sample_paths),
            cache_dir=temp_dir,
            paths=sample_paths,
        )
        assert cache_exists is False  # New cache
        assert cache._enabled is True

        # Store and retrieve images
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cache.put(0, arr)
        cache.finalize()

        retrieved = cache.get(0)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, arr)

        # Cleanup
        cache.clear()

    def test_cache_size_tracking(self, temp_dir, sample_paths):
        """Test cache size tracking."""
        cache = ImageCache(mode="disk", target_size=(32, 32))
        cache.initialize(len(sample_paths), temp_dir, sample_paths)

        assert cache.size == 0

        for i in range(5):
            arr = np.zeros((32, 32, 3), dtype=np.uint8)
            cache.put(i, arr)

        assert cache.size == 5

        cache.clear()

    def test_cache_reuse(self, temp_dir, sample_paths):
        """Test that existing cache is detected and reused."""
        # Create first cache
        cache1 = ImageCache(mode="disk", target_size=(32, 32))
        cache1.initialize(len(sample_paths), temp_dir, sample_paths)

        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cache1.put(0, arr)
        cache1.finalize()

        # Create second cache with same settings - should find existing
        cache2 = ImageCache(mode="disk", target_size=(32, 32))
        cache_exists = cache2.initialize(len(sample_paths), temp_dir, sample_paths)

        assert cache_exists is True  # Should detect existing cache

        # Should be able to retrieve the stored data
        retrieved = cache2.get(0)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, arr)

        cache2.clear()

    def test_cache_invalidation_on_path_change(self, temp_dir, sample_paths):
        """Test that cache is invalidated when paths change."""
        # Create first cache
        cache1 = ImageCache(mode="disk", target_size=(32, 32))
        cache1.initialize(len(sample_paths), temp_dir, sample_paths)

        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        cache1.put(0, arr)
        cache1.finalize()

        # Create second cache with different paths
        different_paths = [temp_dir / f"different_{i}.jpg" for i in range(10)]
        cache2 = ImageCache(mode="disk", target_size=(32, 32))
        cache_exists = cache2.initialize(len(different_paths), temp_dir, different_paths)

        assert cache_exists is False  # Cache should be invalidated

        cache2.clear()

    def test_returns_none_if_not_cached(self, temp_dir, sample_paths):
        """Test cache returns None for non-cached indices."""
        cache = ImageCache(mode="disk", target_size=(32, 32))
        cache.initialize(len(sample_paths), temp_dir, sample_paths)

        result = cache.get(999)  # Index not cached
        assert result is None

        cache.clear()

    def test_is_cached_method(self, temp_dir, sample_paths):
        """Test is_cached method."""
        cache = ImageCache(mode="disk", target_size=(32, 32))
        cache.initialize(len(sample_paths), temp_dir, sample_paths)

        assert cache.is_cached(0) is False

        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        cache.put(0, arr)

        assert cache.is_cached(0) is True
        assert cache.is_cached(1) is False

        cache.clear()

    def test_can_cache_in_ram_estimation(self):
        """Test RAM capacity estimation."""
        cache = ImageCache(mode="ram", max_memory_gb=1.0)

        # Small dataset should fit
        assert cache.can_cache_in_ram(0.5) is True

        # Large dataset should not fit
        assert cache.can_cache_in_ram(100.0) is False

    def test_target_size_stored(self):
        """Test that target_size is stored correctly."""
        cache = ImageCache(mode="ram", target_size=(640, 640))
        assert cache.target_size == (640, 640)

        cache2 = ImageCache(mode="ram", target_size=None)
        assert cache2.target_size is None

    def test_estimate_memory_with_target_size(self, temp_dir):
        """Test memory estimation uses target_size when set."""
        from PIL import Image

        # Create a test image
        img = Image.new("RGB", (1920, 1080), color="red")
        img_path = temp_dir / "test.jpg"
        img.save(img_path)

        # Without target_size - estimates based on original size
        cache_no_resize = ImageCache(mode="ram", target_size=None)
        est_original = cache_no_resize.estimate_memory([img_path])

        # With target_size - estimates based on resized dimensions
        cache_resized = ImageCache(mode="ram", target_size=(640, 640))
        est_resized = cache_resized.estimate_memory([img_path])

        # Resized estimate should be much smaller
        assert est_resized < est_original

    def test_estimate_memory_target_size_exact_calculation(self):
        """Test that target_size gives exact memory calculation without sampling."""
        cache = ImageCache(mode="ram", target_size=(640, 480))

        # Even with fake paths, we can calculate exact memory
        paths = [Path(f"fake_{i}.jpg") for i in range(1000)]
        estimated_gb = cache.estimate_memory(paths)

        expected_bytes_per_img = 640 * 480 * 3
        expected_gb = (expected_bytes_per_img * 1000 * 1.2) / (1024**3)

        assert abs(estimated_gb - expected_gb) < 0.01  # Within 0.01 GB

    def test_cache_suffix_differentiates_caches(self, temp_dir, sample_paths):
        """Test that different cache_suffix creates different caches."""
        # Create cache with suffix1
        cache1 = ImageCache(mode="disk", target_size=(32, 32), cache_suffix="suffix1")
        cache1.initialize(len(sample_paths), temp_dir, sample_paths)

        arr1 = np.ones((32, 32, 3), dtype=np.uint8) * 100
        cache1.put(0, arr1)
        cache1.finalize()

        # Create cache with suffix2
        cache2 = ImageCache(mode="disk", target_size=(32, 32), cache_suffix="suffix2")
        cache_exists = cache2.initialize(len(sample_paths), temp_dir, sample_paths)

        # Should be a new cache (different suffix)
        assert cache_exists is False

        arr2 = np.ones((32, 32, 3), dtype=np.uint8) * 200
        cache2.put(0, arr2)
        cache2.finalize()

        # Verify they store different data
        cache1_reopen = ImageCache(mode="disk", target_size=(32, 32), cache_suffix="suffix1")
        cache1_reopen.initialize(len(sample_paths), temp_dir, sample_paths)
        retrieved1 = cache1_reopen.get(0)

        cache2_reopen = ImageCache(mode="disk", target_size=(32, 32), cache_suffix="suffix2")
        cache2_reopen.initialize(len(sample_paths), temp_dir, sample_paths)
        retrieved2 = cache2_reopen.get(0)

        assert not np.array_equal(retrieved1, retrieved2)

        cache1.clear()
        cache2.clear()


class TestImageCacheMultiprocessing:
    """Tests for LMDB cache multiprocessing compatibility."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for cache tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_paths(self, temp_dir):
        """Create sample image paths for testing."""
        return [temp_dir / f"image_{i}.jpg" for i in range(10)]

    def test_cache_serialization_for_workers(self, temp_dir, sample_paths):
        """Test that cache can be pickled/unpickled for worker processes."""
        import pickle

        cache = ImageCache(mode="disk", target_size=(32, 32))
        cache.initialize(len(sample_paths), temp_dir, sample_paths)

        # Store some data
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cache.put(0, arr)
        cache.finalize()

        # Simulate worker process: pickle and unpickle
        state = pickle.dumps(cache)
        worker_cache = pickle.loads(state)

        # Worker should be able to read the data
        retrieved = worker_cache.get(0)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, arr)

        # Cleanup
        cache.clear()

    def test_multiple_indices(self, temp_dir, sample_paths):
        """Test storing images at various indices."""
        cache = ImageCache(mode="disk", target_size=(32, 32))
        cache.initialize(len(sample_paths), temp_dir, sample_paths)

        # Store at various indices
        test_indices = [0, 3, 5, 7, 9]
        images = {}
        for idx in test_indices:
            arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            images[idx] = arr
            cache.put(idx, arr)

        cache.finalize()

        # Verify all stored
        assert cache.size == len(test_indices)

        # Verify retrieval
        for idx, expected in images.items():
            retrieved = cache.get(idx)
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved, expected)

        # Cleanup
        cache.clear()

    def test_clear_removes_lmdb(self, temp_dir, sample_paths):
        """Test that clear() removes the LMDB database."""
        cache = ImageCache(mode="disk", target_size=(32, 32))
        cache.initialize(len(sample_paths), temp_dir, sample_paths)

        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        cache.put(0, arr)
        cache.finalize()

        cache_path = cache.cache_path
        assert cache_path is not None
        assert cache_path.exists()

        cache.clear()

        assert cache.size == 0
        # LMDB directory should be removed
        assert not (cache_path / "cache.lmdb").exists()


class TestImageCacheEncryption:
    """Tests for encrypted LMDB cache."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for cache tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_paths(self, temp_dir):
        """Create sample image paths for testing."""
        return [temp_dir / f"image_{i}.jpg" for i in range(5)]

    @pytest.fixture
    def test_key(self):
        """Generate a test encryption key (64 hex chars = 32 bytes)."""
        return "0123456789abcdef" * 4  # 64 hex chars

    def test_encrypted_cache_roundtrip(self, temp_dir, sample_paths, test_key):
        """Test encrypted cache can save and load correctly."""
        cache = ImageCache(mode="disk", target_size=(50, 50), encryption_key=test_key)
        cache.initialize(len(sample_paths), temp_dir, sample_paths)

        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cache.put(0, arr)
        cache.finalize()

        retrieved = cache.get(0)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, arr)

        cache.clear()

    def test_plain_cache_not_encrypted(self, temp_dir, sample_paths):
        """Test that cache without key creates plain LMDB."""
        cache = ImageCache(mode="disk", target_size=(50, 50), encryption_key=None)
        cache.initialize(len(sample_paths), temp_dir, sample_paths)

        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cache.put(0, arr)
        cache.finalize()

        # LMDB directory should exist
        cache_path = cache.cache_path
        assert cache_path is not None
        assert (cache_path / "cache.lmdb").exists()

        cache.clear()


class TestLRUImageBuffer:
    """Tests for LRUImageBuffer class."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = LRUImageBuffer(capacity=10)
        assert buffer.capacity == 10
        assert len(buffer) == 0

    def test_put_get_basic(self):
        """Test basic put and get operations."""
        buffer = LRUImageBuffer(capacity=5)

        buffer.put(0, "value0")
        buffer.put(1, "value1")

        assert buffer.get(0) == "value0"
        assert buffer.get(1) == "value1"
        assert buffer.get(2) is None

    def test_capacity_enforced(self):
        """Test that capacity is enforced."""
        buffer = LRUImageBuffer(capacity=3)

        for i in range(5):
            buffer.put(i, f"value{i}")

        assert len(buffer) == 3
        # First two should be evicted
        assert buffer.get(0) is None
        assert buffer.get(1) is None
        # Last three should remain
        assert buffer.get(2) == "value2"
        assert buffer.get(3) == "value3"
        assert buffer.get(4) == "value4"

    def test_lru_eviction_order(self):
        """Test LRU eviction order."""
        buffer = LRUImageBuffer(capacity=3)

        buffer.put(0, "value0")
        buffer.put(1, "value1")
        buffer.put(2, "value2")

        # Access key 0 to make it recently used
        buffer.get(0)

        # Add new item, should evict key 1 (least recently used)
        buffer.put(3, "value3")

        assert buffer.get(0) == "value0"  # Still present (was accessed)
        assert buffer.get(1) is None  # Evicted
        assert buffer.get(2) == "value2"
        assert buffer.get(3) == "value3"

    def test_access_promotes_to_most_recent(self):
        """Test that accessing an item promotes it to most recent."""
        buffer = LRUImageBuffer(capacity=3)

        buffer.put(0, "value0")
        buffer.put(1, "value1")
        buffer.put(2, "value2")

        # Access key 0 multiple times
        buffer.get(0)
        buffer.get(0)

        # Add two new items
        buffer.put(3, "value3")
        buffer.put(4, "value4")

        # Key 0 should still be present
        assert buffer.get(0) == "value0"
        # Keys 1 and 2 should be evicted
        assert buffer.get(1) is None
        assert buffer.get(2) is None

    def test_update_existing_key(self):
        """Test updating an existing key."""
        buffer = LRUImageBuffer(capacity=3)

        buffer.put(0, "value0")
        buffer.put(1, "value1")
        buffer.put(0, "updated0")  # Update key 0

        assert buffer.get(0) == "updated0"
        assert len(buffer) == 2

    def test_clear(self):
        """Test buffer clearing."""
        buffer = LRUImageBuffer(capacity=5)

        for i in range(3):
            buffer.put(i, f"value{i}")

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.get(0) is None

    def test_contains(self):
        """Test __contains__ method."""
        buffer = LRUImageBuffer(capacity=3)

        buffer.put(0, "value0")

        assert 0 in buffer
        assert 1 not in buffer

    def test_size_property(self):
        """Test size property."""
        buffer = LRUImageBuffer(capacity=5)

        assert buffer.size == 0

        buffer.put(0, "a")
        buffer.put(1, "b")

        assert buffer.size == 2

    def test_zero_capacity(self):
        """Test buffer with zero capacity."""
        buffer = LRUImageBuffer(capacity=0)

        buffer.put(0, "value0")
        assert buffer.get(0) is None
        assert len(buffer) == 0

    def test_single_capacity(self):
        """Test buffer with capacity of 1."""
        buffer = LRUImageBuffer(capacity=1)

        buffer.put(0, "value0")
        assert buffer.get(0) == "value0"

        buffer.put(1, "value1")
        assert buffer.get(0) is None
        assert buffer.get(1) == "value1"
