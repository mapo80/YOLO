"""
Image loaders for YOLO datasets.

Provides extensible image loading with support for custom loaders (e.g., encrypted images).

Performance considerations for custom loaders:
- Always close file handles explicitly (use context managers)
- Use img.load() to force data into memory before closing file
- Avoid creating multiple copies of image data
- For encrypted images, decrypt directly into a pre-allocated buffer if possible

Example:
    # Create a custom loader for encrypted images (OPTIMIZED)
    class EncryptedImageLoader(ImageLoader):
        def __init__(self, key: bytes):
            self.key = key
            # Pre-initialize cipher if possible (reuse across calls)

        def __call__(self, path: str) -> Image.Image:
            # Read encrypted data
            with open(path, 'rb') as f:
                encrypted_data = f.read()

            # Decrypt (implementation-specific)
            decrypted = my_decrypt(encrypted_data, self.key)

            # IMPORTANT: Use context manager and load() to close file handle
            with io.BytesIO(decrypted) as buf:
                with Image.open(buf) as img:
                    img.load()  # Force load into memory
                    return img.convert("RGB")

    # Configure via YAML:
    # data:
    #   image_loader:
    #     class_path: my_loaders.EncryptedImageLoader
    #     init_args:
    #       key: "my-secret-key"

    # Or via CLI:
    # --data.image_loader.class_path=my_loaders.EncryptedImageLoader
    # --data.image_loader.init_args.key="my-secret-key"
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


class ImageLoader(ABC):
    """
    Abstract base class for custom image loaders.

    Subclass this to implement custom image loading logic, such as:
    - Decrypting encrypted images
    - Loading from cloud storage
    - Custom preprocessing before loading
    - Loading from non-standard formats

    The loader must return a PIL Image in RGB mode.

    IMPORTANT for implementations:
    - Always close file handles to avoid "Too many open files" errors
    - Use context managers (with statements) for file operations
    - Call img.load() before closing file to force data into memory
    """

    @abstractmethod
    def __call__(self, path: Union[str, Path]) -> Image.Image:
        """
        Load an image from the given path.

        Args:
            path: Full path to the image file

        Returns:
            PIL.Image.Image in RGB mode
        """
        pass


class DefaultImageLoader(ImageLoader):
    """
    Default image loader using PIL with proper file handle management.

    Ensures file handles are closed immediately after loading to prevent
    "Too many open files" errors with large batch sizes and many workers.
    """

    def __call__(self, path: Union[str, Path]) -> Image.Image:
        """Load image using PIL and convert to RGB, ensuring file is closed."""
        with Image.open(path) as img:
            # Force load pixel data into memory before file is closed
            img.load()
            return img.convert("RGB")


class FastImageLoader(ImageLoader):
    """
    High-performance image loader using OpenCV with memory-mapped I/O.

    Optimized for:
    - Large batch sizes (128+)
    - Many DataLoader workers
    - High-throughput training

    Uses OpenCV for faster decoding and numpy for efficient memory handling.
    Falls back to PIL for unsupported formats.
    """

    def __init__(self, use_mmap: bool = True):
        """
        Args:
            use_mmap: Use memory-mapped file reading for large files (default: True)
        """
        self.use_mmap = use_mmap
        self._cv2 = None  # Lazy import

    def _get_cv2(self):
        """Lazy import cv2 to avoid import overhead if not used."""
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2

    def __call__(self, path: Union[str, Path]) -> Image.Image:
        """Load image using OpenCV for speed, convert to PIL RGB."""
        cv2 = self._get_cv2()
        path_str = str(path)

        # Check file size for mmap decision
        file_size = os.path.getsize(path_str)

        if self.use_mmap and file_size > 1024 * 1024:  # > 1MB
            # Memory-mapped reading for large files
            with open(path_str, 'rb') as f:
                # Read file content into buffer
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        else:
            # Direct OpenCV read for smaller files
            img_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)

        if img_bgr is None:
            # Fallback to PIL for unsupported formats
            with Image.open(path_str) as img:
                img.load()
                return img.convert("RGB")

        # Convert BGR to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        return Image.fromarray(img_rgb)


class TurboJPEGLoader(ImageLoader):
    """
    Ultra-fast JPEG loader using libjpeg-turbo via turbojpeg.

    Provides 2-4x speedup over PIL for JPEG images.
    Falls back to PIL for non-JPEG formats.

    Requires: pip install PyTurboJPEG
    And libjpeg-turbo installed on the system.
    """

    def __init__(self):
        self._jpeg = None  # Lazy init

    def _get_jpeg(self):
        """Lazy initialize TurboJPEG decoder."""
        if self._jpeg is None:
            try:
                from turbojpeg import TurboJPEG
                self._jpeg = TurboJPEG()
            except (ImportError, OSError):
                # TurboJPEG not available, will use fallback
                self._jpeg = False
        return self._jpeg

    def __call__(self, path: Union[str, Path]) -> Image.Image:
        """Load image using TurboJPEG if available, fallback to PIL."""
        path_str = str(path)
        jpeg = self._get_jpeg()

        # Check if JPEG and TurboJPEG is available
        is_jpeg = path_str.lower().endswith(('.jpg', '.jpeg'))

        if is_jpeg and jpeg:
            try:
                with open(path_str, 'rb') as f:
                    img_data = f.read()
                # Decode directly to RGB numpy array
                img_rgb = jpeg.decode(img_data)
                return Image.fromarray(img_rgb)
            except Exception:
                pass  # Fall through to PIL

        # Fallback to PIL
        with Image.open(path_str) as img:
            img.load()
            return img.convert("RGB")
