"""
High-performance encrypted image loader for YOLOV9MIT.

This module provides an ImageLoader that can load AES-256 encrypted images (.enc files).
Optimized for high-throughput training with proper file handle management.

Performance optimizations:
- Pre-initialized cipher context reuse
- Proper file handle closure (no leaks with batch_size > 128)
- Memory-efficient buffer handling
- Optional OpenCV decoding for faster image processing

Usage with YOLOV9MIT:
    Configure in YAML:
        data:
          encryption_key: "your-64-char-hex-key"  # or use env var
          image_loader:
            class_path: yolo.data.encrypted_loader.EncryptedImageLoader

    Or via CLI:
        --data.image_loader.class_path=yolo.data.encrypted_loader.EncryptedImageLoader

Environment:
    Set YOLO_ENCRYPTION_KEY with the hex-encoded AES-256 key:
        export YOLO_ENCRYPTION_KEY=<64-char-hex-key>
"""

import io
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from yolo.data.loaders import ImageLoader
from yolo.data.crypto import CryptoManager


class EncryptedImageLoader(ImageLoader):
    """
    High-performance image loader for AES-256 encrypted images.

    Supports both encrypted (.enc) and regular image files.
    For .enc files, decrypts using the configured encryption key.
    For regular files, loads normally with PIL.

    The encryption key can be provided via:
    1. Constructor parameter (key_hex)
    2. YOLO_ENCRYPTION_KEY environment variable
    3. YAML config: data.encryption_key

    Optimized for:
    - Large batch sizes (128+)
    - Many DataLoader workers
    - High-throughput training

    Args:
        key_hex: Hex-encoded 32-byte AES key (64 chars). If None, uses env var.
        use_opencv: Use OpenCV for faster image decoding (default: True if available)

    Usage:
        # In YAML config:
        data:
          encryption_key: "your-64-char-hex-key"
          image_loader:
            class_path: yolo.data.encrypted_loader.EncryptedImageLoader
            init_args:
              use_opencv: true
    """

    def __init__(self, key_hex: Optional[str] = None, use_opencv: bool = True):
        self._crypto = CryptoManager(key_hex=key_hex)
        self._use_opencv = use_opencv
        self._cv2 = None  # Lazy import

    def __getstate__(self):
        """Prepare state for pickling (required for spawn multiprocessing)."""
        return {
            "_crypto": self._crypto.__getstate__(),
            "_use_opencv": self._use_opencv,
        }

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self._crypto = CryptoManager()
        self._crypto.__setstate__(state.get("_crypto", {}))
        self._use_opencv = state.get("_use_opencv", True)
        self._cv2 = None

    def _get_cv2(self):
        """Lazy import cv2 to avoid import overhead if not used."""
        if self._cv2 is None and self._use_opencv:
            try:
                import cv2

                self._cv2 = cv2
            except ImportError:
                self._cv2 = False  # Mark as unavailable
        return self._cv2 if self._cv2 else None

    def _decode_image_opencv(self, image_bytes: bytes) -> Optional[Image.Image]:
        """Decode image bytes using OpenCV (faster than PIL)."""
        cv2 = self._get_cv2()
        if cv2 is None:
            return None

        try:
            import numpy as np

            # Decode from bytes
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                return None

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
        except Exception:
            return None

    def _decode_image_pil(self, image_bytes: bytes) -> Image.Image:
        """Decode image bytes using PIL with proper handle management."""
        with io.BytesIO(image_bytes) as buf:
            with Image.open(buf) as img:
                img.load()  # Force load into memory before closing
                return img.convert("RGB")

    def __call__(self, path: Union[str, Path]) -> Image.Image:
        """
        Load an image from the given path.

        For .enc files, decrypts the image first.
        For other files, loads directly.

        Args:
            path: Full path to the image file

        Returns:
            PIL.Image.Image in RGB mode
        """
        path_str = str(path)

        if path_str.lower().endswith(".enc"):
            # Encrypted image - decrypt first
            with open(path_str, "rb") as f:
                encrypted_data = f.read()

            decrypted = self._crypto.decrypt(encrypted_data)

            # Try OpenCV first (faster), fallback to PIL
            img = self._decode_image_opencv(decrypted)
            if img is not None:
                return img

            return self._decode_image_pil(decrypted)
        else:
            # Regular image - load with proper handle management
            # Try OpenCV for speed
            cv2 = self._get_cv2()
            if cv2 is not None:
                img_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(img_rgb)

            # Fallback to PIL with proper handle closure
            with Image.open(path_str) as img:
                img.load()
                return img.convert("RGB")


# Expose for easy import
__all__ = ["EncryptedImageLoader"]
