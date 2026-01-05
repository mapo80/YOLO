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
          image_loader:
            class_path: yolo.data.encrypted_loader.EncryptedImageLoader

    Or via CLI:
        --data.image_loader.class_path=yolo.data.encrypted_loader.EncryptedImageLoader

Environment:
    Set YOLO_IMAGE_ENCRYPTION_KEY with the hex-encoded AES-256 key:
        export YOLO_IMAGE_ENCRYPTION_KEY=<64-char-hex-key>
"""

import io
import os
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from yolo.data.loaders import ImageLoader

# Environment variable name for the encryption key
ENV_KEY_NAME = "YOLO_IMAGE_ENCRYPTION_KEY"


class EncryptedImageLoader(ImageLoader):
    """
    High-performance image loader for AES-256 encrypted images.

    Supports both encrypted (.enc) and regular image files.
    For .enc files, decrypts using the key from YOLO_IMAGE_ENCRYPTION_KEY env var.
    For regular files, loads normally with PIL.

    Optimized for:
    - Large batch sizes (128+)
    - Many DataLoader workers
    - High-throughput training

    Args:
        use_opencv: Use OpenCV for faster image decoding (default: True if available)

    Usage:
        # In YAML config:
        data:
          image_loader:
            class_path: yolo.data.encrypted_loader.EncryptedImageLoader
            init_args:
              use_opencv: true
    """

    def __init__(self, use_opencv: bool = True):
        self._key: Optional[bytes] = None
        self._use_opencv = use_opencv
        self._cv2 = None  # Lazy import
        # Lazy import cryptography
        self._cipher_module = None
        self._algorithms = None
        self._modes = None
        self._backend = None

    def _init_crypto(self):
        """Lazy initialize cryptography modules."""
        if self._cipher_module is None:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

            self._cipher_module = Cipher
            self._algorithms = algorithms
            self._modes = modes
            self._backend = default_backend()

    def _get_cv2(self):
        """Lazy import cv2 to avoid import overhead if not used."""
        if self._cv2 is None and self._use_opencv:
            try:
                import cv2

                self._cv2 = cv2
            except ImportError:
                self._cv2 = False  # Mark as unavailable
        return self._cv2 if self._cv2 else None

    def _get_key(self) -> bytes:
        """
        Get the AES-256 encryption key from environment variable.

        The key is cached after first retrieval for performance.

        Returns:
            bytes: 32-byte AES key

        Raises:
            ValueError: If YOLO_IMAGE_ENCRYPTION_KEY is not set
        """
        if self._key is None:
            key_hex = os.environ.get(ENV_KEY_NAME)
            if not key_hex:
                raise ValueError(
                    f"Environment variable {ENV_KEY_NAME} not set. "
                    f"Export the key with: export {ENV_KEY_NAME}=<64-char-hex-key>"
                )
            self._key = bytes.fromhex(key_hex)
            if len(self._key) != 32:
                raise ValueError(f"Key must be 32 bytes (64 hex chars), got {len(self._key)} bytes")
        return self._key

    def _decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using AES-256-CBC.

        Input format: IV (16 bytes) + encrypted_data

        Args:
            encrypted_data: IV + ciphertext

        Returns:
            Decrypted plaintext bytes
        """
        self._init_crypto()
        key = self._get_key()

        # Extract IV and ciphertext
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]

        # Create cipher and decrypt
        cipher = self._cipher_module(
            self._algorithms.AES(key), self._modes.CBC(iv), backend=self._backend
        )
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        # Remove PKCS7 padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]

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

            decrypted = self._decrypt(encrypted_data)

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
