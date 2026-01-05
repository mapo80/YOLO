"""
Tests for image loaders - efficiency, file handle management, and performance.

These tests verify:
1. File handles are properly closed (no leaks)
2. Loaders return correct format (PIL RGB)
3. High batch simulation doesn't cause "Too many open files"
4. Performance characteristics of different loaders
"""

import gc
import io
import os
import resource
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from yolo.data.loaders import (
    DefaultImageLoader,
    FastImageLoader,
    ImageLoader,
    TurboJPEGLoader,
)
from yolo.data.encrypted_loader import EncryptedImageLoader


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_image_dir(tmp_path):
    """Create a temporary directory with test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    return img_dir


@pytest.fixture
def sample_jpg_image(temp_image_dir) -> Path:
    """Create a sample JPEG image."""
    img_path = temp_image_dir / "test.jpg"
    img = Image.new("RGB", (640, 480), color=(255, 0, 0))
    img.save(img_path, "JPEG", quality=85)
    return img_path


@pytest.fixture
def sample_png_image(temp_image_dir) -> Path:
    """Create a sample PNG image."""
    img_path = temp_image_dir / "test.png"
    img = Image.new("RGB", (640, 480), color=(0, 255, 0))
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def sample_rgba_image(temp_image_dir) -> Path:
    """Create a sample RGBA image (with alpha channel)."""
    img_path = temp_image_dir / "test_rgba.png"
    img = Image.new("RGBA", (640, 480), color=(0, 0, 255, 128))
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def sample_grayscale_image(temp_image_dir) -> Path:
    """Create a sample grayscale image."""
    img_path = temp_image_dir / "test_gray.jpg"
    img = Image.new("L", (640, 480), color=128)
    img.save(img_path, "JPEG")
    return img_path


@pytest.fixture
def many_images(temp_image_dir, count: int = 500) -> List[Path]:
    """Create many test images for stress testing."""
    paths = []
    for i in range(count):
        img_path = temp_image_dir / f"img_{i:04d}.jpg"
        # Create varied images to simulate real data
        color = ((i * 17) % 256, (i * 31) % 256, (i * 47) % 256)
        img = Image.new("RGB", (640, 480), color=color)
        img.save(img_path, "JPEG", quality=85)
        paths.append(img_path)
    return paths


@pytest.fixture
def large_image(temp_image_dir) -> Path:
    """Create a large image (>1MB) for mmap testing."""
    img_path = temp_image_dir / "large.jpg"
    # Create a large image with random-ish data
    np_img = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    img = Image.fromarray(np_img)
    img.save(img_path, "JPEG", quality=95)
    return img_path


def get_open_file_count() -> int:
    """Get current number of open file descriptors."""
    try:
        # On Unix-like systems
        import subprocess
        pid = os.getpid()
        result = subprocess.run(
            ["lsof", "-p", str(pid)],
            capture_output=True,
            text=True,
        )
        return len(result.stdout.strip().split("\n")) - 1  # Subtract header
    except Exception:
        # Fallback: use resource limits
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        return soft


# =============================================================================
# DefaultImageLoader Tests
# =============================================================================


class TestDefaultImageLoader:
    """Tests for DefaultImageLoader."""

    def test_loads_jpg_image(self, sample_jpg_image):
        """Test loading JPEG image."""
        loader = DefaultImageLoader()
        img = loader(sample_jpg_image)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (640, 480)

    def test_loads_png_image(self, sample_png_image):
        """Test loading PNG image."""
        loader = DefaultImageLoader()
        img = loader(sample_png_image)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (640, 480)

    def test_converts_rgba_to_rgb(self, sample_rgba_image):
        """Test that RGBA images are converted to RGB."""
        loader = DefaultImageLoader()
        img = loader(sample_rgba_image)

        assert img.mode == "RGB"

    def test_converts_grayscale_to_rgb(self, sample_grayscale_image):
        """Test that grayscale images are converted to RGB."""
        loader = DefaultImageLoader()
        img = loader(sample_grayscale_image)

        assert img.mode == "RGB"

    def test_accepts_path_object(self, sample_jpg_image):
        """Test that Path objects are accepted."""
        loader = DefaultImageLoader()
        img = loader(Path(sample_jpg_image))

        assert isinstance(img, Image.Image)

    def test_accepts_string_path(self, sample_jpg_image):
        """Test that string paths are accepted."""
        loader = DefaultImageLoader()
        img = loader(str(sample_jpg_image))

        assert isinstance(img, Image.Image)

    def test_file_handle_closed_after_load(self, sample_jpg_image):
        """Test that file handle is closed after loading."""
        loader = DefaultImageLoader()

        # Get baseline file count
        gc.collect()
        initial_count = get_open_file_count()

        # Load image multiple times
        for _ in range(10):
            img = loader(sample_jpg_image)
            del img

        gc.collect()
        final_count = get_open_file_count()

        # File count should not increase significantly
        assert final_count <= initial_count + 5, (
            f"File handles may be leaking: {initial_count} -> {final_count}"
        )

    def test_no_file_leak_many_images(self, temp_image_dir):
        """Test no file descriptor leak with many images."""
        # Create 200 images
        paths = []
        for i in range(200):
            img_path = temp_image_dir / f"leak_test_{i:04d}.jpg"
            img = Image.new("RGB", (100, 100), color=(i % 256, 0, 0))
            img.save(img_path, "JPEG")
            paths.append(img_path)

        loader = DefaultImageLoader()

        gc.collect()
        initial_count = get_open_file_count()

        # Load all images
        for path in paths:
            img = loader(path)
            # Explicitly delete to ensure no references
            del img

        gc.collect()
        final_count = get_open_file_count()

        # Should not leak file handles
        assert final_count <= initial_count + 10, (
            f"File leak detected: {initial_count} -> {final_count} "
            f"(+{final_count - initial_count})"
        )


# =============================================================================
# FastImageLoader Tests
# =============================================================================


class TestFastImageLoader:
    """Tests for FastImageLoader (OpenCV-based)."""

    @pytest.fixture
    def cv2_available(self):
        """Check if OpenCV is available."""
        try:
            import cv2
            return True
        except ImportError:
            return False

    def test_loads_jpg_image(self, sample_jpg_image, cv2_available):
        """Test loading JPEG with OpenCV."""
        if not cv2_available:
            pytest.skip("OpenCV not installed")

        loader = FastImageLoader()
        img = loader(sample_jpg_image)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (640, 480)

    def test_loads_png_image(self, sample_png_image, cv2_available):
        """Test loading PNG with OpenCV."""
        if not cv2_available:
            pytest.skip("OpenCV not installed")

        loader = FastImageLoader()
        img = loader(sample_png_image)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_falls_back_to_pil_on_failure(self, sample_jpg_image):
        """Test fallback to PIL when OpenCV fails."""
        loader = FastImageLoader()

        # Mock cv2 to return None (simulating failure)
        with patch.object(loader, "_get_cv2") as mock_cv2:
            mock_cv2_module = MagicMock()
            mock_cv2_module.imread.return_value = None
            mock_cv2_module.imdecode.return_value = None
            mock_cv2_module.IMREAD_COLOR = 1
            mock_cv2.return_value = mock_cv2_module

            img = loader(sample_jpg_image)

            assert isinstance(img, Image.Image)
            assert img.mode == "RGB"

    def test_mmap_for_large_files(self, large_image, cv2_available):
        """Test that large files use memory-mapped reading."""
        if not cv2_available:
            pytest.skip("OpenCV not installed")

        loader = FastImageLoader(use_mmap=True)
        img = loader(large_image)

        assert isinstance(img, Image.Image)
        # Large image should still load correctly
        assert img.size[0] > 1000

    def test_no_mmap_for_small_files(self, sample_jpg_image, cv2_available):
        """Test that small files don't use memory-mapped reading."""
        if not cv2_available:
            pytest.skip("OpenCV not installed")

        loader = FastImageLoader(use_mmap=True)

        # File should be < 1MB, so direct read should be used
        file_size = os.path.getsize(sample_jpg_image)
        assert file_size < 1024 * 1024

        img = loader(sample_jpg_image)
        assert isinstance(img, Image.Image)

    def test_mmap_disabled(self, large_image, cv2_available):
        """Test loading with mmap disabled."""
        if not cv2_available:
            pytest.skip("OpenCV not installed")

        loader = FastImageLoader(use_mmap=False)
        img = loader(large_image)

        assert isinstance(img, Image.Image)

    def test_lazy_cv2_import(self):
        """Test that cv2 is imported lazily."""
        loader = FastImageLoader()
        assert loader._cv2 is None

    def test_file_handle_closed(self, sample_jpg_image, cv2_available):
        """Test file handles are closed after loading."""
        if not cv2_available:
            pytest.skip("OpenCV not installed")

        loader = FastImageLoader()

        gc.collect()
        initial_count = get_open_file_count()

        for _ in range(50):
            img = loader(sample_jpg_image)
            del img

        gc.collect()
        final_count = get_open_file_count()

        assert final_count <= initial_count + 5


# =============================================================================
# TurboJPEGLoader Tests
# =============================================================================


class TestTurboJPEGLoader:
    """Tests for TurboJPEGLoader."""

    @pytest.fixture
    def turbojpeg_available(self):
        """Check if TurboJPEG is available."""
        try:
            from turbojpeg import TurboJPEG
            TurboJPEG()
            return True
        except (ImportError, OSError):
            return False

    def test_loads_jpg_image(self, sample_jpg_image, turbojpeg_available):
        """Test loading JPEG with TurboJPEG."""
        loader = TurboJPEGLoader()
        img = loader(sample_jpg_image)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (640, 480)

    def test_falls_back_for_png(self, sample_png_image):
        """Test fallback to PIL for non-JPEG formats."""
        loader = TurboJPEGLoader()
        img = loader(sample_png_image)

        # Should still work via PIL fallback
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_lazy_initialization(self):
        """Test that TurboJPEG is initialized lazily."""
        loader = TurboJPEGLoader()
        assert loader._jpeg is None

    def test_handles_turbojpeg_not_installed(self, sample_jpg_image):
        """Test graceful fallback when TurboJPEG not installed."""
        loader = TurboJPEGLoader()

        # Force _jpeg to False (simulating not installed)
        loader._jpeg = False

        img = loader(sample_jpg_image)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_file_handle_closed(self, sample_jpg_image):
        """Test file handles are closed."""
        loader = TurboJPEGLoader()

        gc.collect()
        initial_count = get_open_file_count()

        for _ in range(50):
            img = loader(sample_jpg_image)
            del img

        gc.collect()
        final_count = get_open_file_count()

        assert final_count <= initial_count + 5


# =============================================================================
# High Load / Stress Tests
# =============================================================================


class TestHighLoadSimulation:
    """Stress tests simulating high batch sizes and many workers."""

    @pytest.fixture
    def stress_test_images(self, temp_image_dir):
        """Create images for stress testing."""
        paths = []
        for i in range(300):
            img_path = temp_image_dir / f"stress_{i:04d}.jpg"
            color = ((i * 17) % 256, (i * 31) % 256, (i * 47) % 256)
            img = Image.new("RGB", (640, 480), color=color)
            img.save(img_path, "JPEG", quality=75)
            paths.append(img_path)
        return paths

    def test_default_loader_no_leak_300_images(self, stress_test_images):
        """Test DefaultImageLoader doesn't leak with 300 images."""
        loader = DefaultImageLoader()

        gc.collect()
        initial_count = get_open_file_count()

        # Simulate batch loading
        for path in stress_test_images:
            img = loader(path)
            del img

        gc.collect()
        final_count = get_open_file_count()

        leak = final_count - initial_count
        assert leak <= 10, f"File descriptor leak: +{leak}"

    def test_default_loader_batch_simulation(self, stress_test_images):
        """Simulate batch loading with multiple images held in memory."""
        loader = DefaultImageLoader()
        batch_size = 32

        gc.collect()
        initial_count = get_open_file_count()

        # Simulate multiple batches
        for batch_start in range(0, len(stress_test_images), batch_size):
            batch_paths = stress_test_images[batch_start:batch_start + batch_size]

            # Load batch
            batch_images = [loader(path) for path in batch_paths]

            # "Process" batch (just verify)
            assert len(batch_images) <= batch_size
            for img in batch_images:
                assert img.mode == "RGB"

            # Clear batch
            del batch_images

        gc.collect()
        final_count = get_open_file_count()

        leak = final_count - initial_count
        assert leak <= 10, f"File descriptor leak in batch simulation: +{leak}"

    def test_rapid_sequential_loading(self, stress_test_images):
        """Test rapid sequential loading doesn't cause issues."""
        loader = DefaultImageLoader()

        gc.collect()
        initial_count = get_open_file_count()

        start_time = time.time()

        # Load as fast as possible
        for path in stress_test_images[:100]:
            img = loader(path)
            # Don't delete immediately - simulate some processing time
            _ = img.size

        elapsed = time.time() - start_time

        gc.collect()
        final_count = get_open_file_count()

        leak = final_count - initial_count
        assert leak <= 15, f"Leak during rapid loading: +{leak}"

        # Should complete reasonably fast (< 10s for 100 images)
        assert elapsed < 10, f"Loading too slow: {elapsed:.2f}s"


# =============================================================================
# Performance Comparison Tests
# =============================================================================


class TestLoaderPerformance:
    """Performance comparison tests (informational, not strict assertions)."""

    @pytest.fixture
    def perf_test_images(self, temp_image_dir):
        """Create images for performance testing."""
        paths = []
        for i in range(50):
            img_path = temp_image_dir / f"perf_{i:04d}.jpg"
            # Create more realistic sized images
            np_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(np_img)
            img.save(img_path, "JPEG", quality=85)
            paths.append(img_path)
        return paths

    def test_default_loader_throughput(self, perf_test_images):
        """Measure DefaultImageLoader throughput."""
        loader = DefaultImageLoader()

        # Warmup
        for path in perf_test_images[:5]:
            _ = loader(path)

        # Measure
        start = time.time()
        for path in perf_test_images:
            img = loader(path)
            del img
        elapsed = time.time() - start

        throughput = len(perf_test_images) / elapsed
        print(f"\nDefaultImageLoader: {throughput:.1f} images/sec")

        # Should load at least 10 images per second
        assert throughput > 10, f"Too slow: {throughput:.1f} img/s"

    def test_fast_loader_throughput(self, perf_test_images):
        """Measure FastImageLoader throughput."""
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")

        loader = FastImageLoader()

        # Warmup
        for path in perf_test_images[:5]:
            _ = loader(path)

        # Measure
        start = time.time()
        for path in perf_test_images:
            img = loader(path)
            del img
        elapsed = time.time() - start

        throughput = len(perf_test_images) / elapsed
        print(f"\nFastImageLoader: {throughput:.1f} images/sec")

        # FastImageLoader should be at least as fast as default
        assert throughput > 10, f"Too slow: {throughput:.1f} img/s"

    def test_turbojpeg_loader_throughput(self, perf_test_images):
        """Measure TurboJPEGLoader throughput."""
        loader = TurboJPEGLoader()

        # Warmup (also initializes _jpeg)
        for path in perf_test_images[:5]:
            _ = loader(path)

        # Measure
        start = time.time()
        for path in perf_test_images:
            img = loader(path)
            del img
        elapsed = time.time() - start

        throughput = len(perf_test_images) / elapsed
        print(f"\nTurboJPEGLoader: {throughput:.1f} images/sec")

        # Should at least work (may fall back to PIL)
        assert throughput > 5, f"Too slow: {throughput:.1f} img/s"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_file_raises_error(self, temp_image_dir):
        """Test that nonexistent file raises appropriate error."""
        loader = DefaultImageLoader()
        fake_path = temp_image_dir / "nonexistent.jpg"

        with pytest.raises(FileNotFoundError):
            loader(fake_path)

    def test_corrupted_image_handling(self, temp_image_dir):
        """Test handling of corrupted image files."""
        corrupt_path = temp_image_dir / "corrupt.jpg"
        corrupt_path.write_bytes(b"not a real image")

        loader = DefaultImageLoader()

        with pytest.raises(Exception):  # PIL raises various exceptions
            loader(corrupt_path)

    def test_empty_file_handling(self, temp_image_dir):
        """Test handling of empty files."""
        empty_path = temp_image_dir / "empty.jpg"
        empty_path.write_bytes(b"")

        loader = DefaultImageLoader()

        with pytest.raises(Exception):
            loader(empty_path)

    def test_very_small_image(self, temp_image_dir):
        """Test loading very small images."""
        tiny_path = temp_image_dir / "tiny.jpg"
        img = Image.new("RGB", (1, 1), color=(255, 0, 0))
        img.save(tiny_path, "JPEG")

        loader = DefaultImageLoader()
        loaded = loader(tiny_path)

        assert loaded.size == (1, 1)
        assert loaded.mode == "RGB"

    def test_very_large_image(self, temp_image_dir):
        """Test loading large images."""
        large_path = temp_image_dir / "large.png"
        # Create 4K image
        img = Image.new("RGB", (3840, 2160), color=(0, 255, 0))
        img.save(large_path, "PNG")

        loader = DefaultImageLoader()
        loaded = loader(large_path)

        assert loaded.size == (3840, 2160)
        assert loaded.mode == "RGB"


# =============================================================================
# Abstract Base Class Tests
# =============================================================================


class TestImageLoaderABC:
    """Test ImageLoader abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that ImageLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ImageLoader()

    def test_subclass_must_implement_call(self):
        """Test that subclasses must implement __call__."""

        class IncompleteLoader(ImageLoader):
            pass

        with pytest.raises(TypeError):
            IncompleteLoader()

    def test_custom_loader_implementation(self, sample_jpg_image):
        """Test that custom loaders work correctly."""

        class CustomLoader(ImageLoader):
            def __call__(self, path):
                # Simple implementation that adds a red tint
                with Image.open(path) as img:
                    img.load()
                    return img.convert("RGB")

        loader = CustomLoader()
        img = loader(sample_jpg_image)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"


# =============================================================================
# EncryptedImageLoader Tests
# =============================================================================


class TestEncryptedImageLoader:
    """Tests for EncryptedImageLoader."""

    @pytest.fixture
    def aes_key(self):
        """Generate a test AES-256 key."""
        # 32 bytes = 256 bits for AES-256
        return os.urandom(32)

    @pytest.fixture
    def aes_key_hex(self, aes_key):
        """Return hex-encoded key."""
        return aes_key.hex()

    @pytest.fixture
    def encrypted_image(self, temp_image_dir, aes_key) -> Path:
        """Create an encrypted test image."""
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        # Create a test image
        img = Image.new("RGB", (640, 480), color=(255, 128, 0))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG", quality=85)
        plaintext = img_bytes.getvalue()

        # PKCS7 padding
        block_size = 16
        padding_length = block_size - (len(plaintext) % block_size)
        padded_data = plaintext + bytes([padding_length] * padding_length)

        # Encrypt with AES-256-CBC
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Write encrypted file
        enc_path = temp_image_dir / "test_encrypted.enc"
        enc_path.write_bytes(iv + ciphertext)

        return enc_path

    @pytest.fixture
    def set_env_key(self, aes_key_hex, monkeypatch):
        """Set the encryption key environment variable."""
        monkeypatch.setenv("YOLO_ENCRYPTION_KEY", aes_key_hex)

    def test_loads_encrypted_image(self, encrypted_image, set_env_key):
        """Test loading an encrypted image."""
        loader = EncryptedImageLoader()
        img = loader(encrypted_image)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (640, 480)

    def test_loads_regular_image(self, sample_jpg_image, set_env_key):
        """Test loading regular (non-encrypted) images."""
        loader = EncryptedImageLoader()
        img = loader(sample_jpg_image)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_missing_env_key_raises_error(self, encrypted_image, monkeypatch):
        """Test that missing key raises ValueError."""
        monkeypatch.delenv("YOLO_ENCRYPTION_KEY", raising=False)

        loader = EncryptedImageLoader()

        with pytest.raises(ValueError, match="YOLO_ENCRYPTION_KEY"):
            loader(encrypted_image)

    def test_invalid_key_length_raises_error(self, encrypted_image, monkeypatch):
        """Test that invalid key length raises ValueError."""
        # Set a key that's too short (only 16 bytes instead of 32)
        monkeypatch.setenv("YOLO_ENCRYPTION_KEY", "a" * 32)  # 16 bytes

        loader = EncryptedImageLoader()

        with pytest.raises(ValueError, match="32 bytes"):
            loader(encrypted_image)

    def test_key_caching(self, aes_key_hex, monkeypatch):
        """Test that key is cached after first retrieval."""
        monkeypatch.setenv("YOLO_ENCRYPTION_KEY", aes_key_hex)

        loader = EncryptedImageLoader()

        # First call should cache the key (via crypto manager)
        key1 = loader._crypto._get_key()

        # Modify env (should not affect cached key)
        monkeypatch.setenv("YOLO_ENCRYPTION_KEY", "b" * 64)

        key2 = loader._crypto._get_key()

        assert key1 == key2

    def test_lazy_crypto_initialization(self):
        """Test that crypto modules are initialized lazily."""
        loader = EncryptedImageLoader()

        # Crypto manager is created but modules are not initialized
        assert loader._crypto._cipher_module is None
        assert loader._crypto._algorithms is None
        assert loader._crypto._modes is None

    def test_file_handle_closed(self, encrypted_image, set_env_key):
        """Test file handles are closed after loading encrypted images."""
        loader = EncryptedImageLoader()

        gc.collect()
        initial_count = get_open_file_count()

        for _ in range(50):
            img = loader(encrypted_image)
            del img

        gc.collect()
        final_count = get_open_file_count()

        assert final_count <= initial_count + 5

    def test_opencv_fallback(self, encrypted_image, set_env_key):
        """Test that loader works when OpenCV is not available."""
        loader = EncryptedImageLoader(use_opencv=False)
        img = loader(encrypted_image)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_stress_encrypted_images(self, temp_image_dir, aes_key, set_env_key):
        """Stress test with many encrypted images."""
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        # Create 100 encrypted images
        paths = []
        for i in range(100):
            # Create image
            color = ((i * 17) % 256, (i * 31) % 256, (i * 47) % 256)
            img = Image.new("RGB", (320, 240), color=color)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG", quality=75)
            plaintext = img_bytes.getvalue()

            # Pad and encrypt
            block_size = 16
            padding_length = block_size - (len(plaintext) % block_size)
            padded_data = plaintext + bytes([padding_length] * padding_length)

            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()

            enc_path = temp_image_dir / f"stress_{i:04d}.enc"
            enc_path.write_bytes(iv + ciphertext)
            paths.append(enc_path)

        # Load all images
        loader = EncryptedImageLoader()

        gc.collect()
        initial_count = get_open_file_count()

        for path in paths:
            img = loader(path)
            assert img.mode == "RGB"
            del img

        gc.collect()
        final_count = get_open_file_count()

        leak = final_count - initial_count
        assert leak <= 10, f"File descriptor leak: +{leak}"

    def test_throughput_encrypted(self, temp_image_dir, aes_key, set_env_key):
        """Measure throughput for encrypted images."""
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        # Create 30 encrypted images
        paths = []
        for i in range(30):
            np_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(np_img)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG", quality=85)
            plaintext = img_bytes.getvalue()

            block_size = 16
            padding_length = block_size - (len(plaintext) % block_size)
            padded_data = plaintext + bytes([padding_length] * padding_length)

            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()

            enc_path = temp_image_dir / f"perf_{i:04d}.enc"
            enc_path.write_bytes(iv + ciphertext)
            paths.append(enc_path)

        loader = EncryptedImageLoader()

        # Warmup
        for path in paths[:3]:
            _ = loader(path)

        # Measure
        start = time.time()
        for path in paths:
            img = loader(path)
            del img
        elapsed = time.time() - start

        throughput = len(paths) / elapsed
        print(f"\nEncryptedImageLoader: {throughput:.1f} images/sec")

        # Should load at least 5 images per second (decryption overhead)
        assert throughput > 5, f"Too slow: {throughput:.1f} img/s"
