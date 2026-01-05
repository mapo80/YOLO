"""
Centralized encryption/decryption module for YOLO data pipeline.

Provides AES-256-CBC encryption for:
- Encrypted image loading (EncryptedImageLoader)
- Encrypted disk cache (ImageCache with disk mode)

The encryption key can be provided via:
1. Environment variable YOLO_ENCRYPTION_KEY (highest priority)
2. YAML configuration: data.encryption_key
3. Direct parameter passing

Usage:
    from yolo.data.crypto import CryptoManager

    # Create manager (key from env or parameter)
    crypto = CryptoManager(key_hex="...")  # or from env

    # Encrypt/decrypt data
    encrypted = crypto.encrypt(plaintext_bytes)
    decrypted = crypto.decrypt(encrypted_bytes)

    # Encrypt/decrypt numpy arrays (for cache)
    encrypted = crypto.encrypt_array(np_array)
    decrypted = crypto.decrypt_array(encrypted_bytes)
"""

import io
import os
from typing import Optional

import numpy as np

# Environment variable name for the encryption key
ENV_KEY_NAME = "YOLO_ENCRYPTION_KEY"


class CryptoManager:
    """
    Centralized encryption manager using AES-256-CBC.

    Handles encryption/decryption for both image files and numpy cache arrays.
    Key can be provided directly or loaded from environment variable.

    Args:
        key_hex: Hex-encoded 32-byte AES key (64 hex characters).
            If None, will try to load from YOLO_ENCRYPTION_KEY env var.

    Raises:
        ValueError: If no key is provided and env var is not set.
    """

    def __init__(self, key_hex: Optional[str] = None):
        self._key: Optional[bytes] = None
        self._key_hex = key_hex
        # Lazy import cryptography modules
        self._cipher_module = None
        self._algorithms = None
        self._modes = None
        self._backend = None

    def __getstate__(self):
        """Prepare state for pickling (required for spawn multiprocessing)."""
        return {
            "_key": self._key,
            "_key_hex": self._key_hex,
        }

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self._key = state.get("_key")
        self._key_hex = state.get("_key_hex")
        # Reset lazy-loaded modules
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

    def _get_key(self) -> bytes:
        """
        Get the AES-256 encryption key.

        Priority:
        1. Key from constructor parameter (key_hex)
        2. YOLO_ENCRYPTION_KEY environment variable

        Returns:
            bytes: 32-byte AES key

        Raises:
            ValueError: If no key is available
        """
        if self._key is not None:
            return self._key

        # Try constructor parameter first
        key_hex = self._key_hex

        # Then try environment variable
        if not key_hex:
            key_hex = os.environ.get(ENV_KEY_NAME)

        if not key_hex:
            raise ValueError(
                f"No encryption key provided. Either:\n"
                f"  1. Set {ENV_KEY_NAME} environment variable\n"
                f"  2. Configure data.encryption_key in YAML\n"
                f"  3. Pass key_hex parameter to CryptoManager"
            )

        self._key = bytes.fromhex(key_hex)
        if len(self._key) != 32:
            raise ValueError(
                f"Encryption key must be 32 bytes (64 hex chars), got {len(self._key)} bytes"
            )

        return self._key

    def is_configured(self) -> bool:
        """Check if encryption key is available (without raising error)."""
        try:
            self._get_key()
            return True
        except ValueError:
            return False

    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data using AES-256-CBC.

        Output format: IV (16 bytes) + ciphertext

        Args:
            data: Plaintext bytes to encrypt

        Returns:
            Encrypted bytes (IV + ciphertext)
        """
        self._init_crypto()
        key = self._get_key()

        # Generate random IV
        iv = os.urandom(16)

        # Pad data to block size (PKCS7)
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padded_data = data + bytes([padding_length] * padding_length)

        # Encrypt
        cipher = self._cipher_module(
            self._algorithms.AES(key), self._modes.CBC(iv), backend=self._backend
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        return iv + ciphertext

    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt data using AES-256-CBC.

        Input format: IV (16 bytes) + ciphertext

        Args:
            data: Encrypted bytes (IV + ciphertext)

        Returns:
            Decrypted plaintext bytes
        """
        self._init_crypto()
        key = self._get_key()

        # Extract IV and ciphertext
        iv = data[:16]
        ciphertext = data[16:]

        # Decrypt
        cipher = self._cipher_module(
            self._algorithms.AES(key), self._modes.CBC(iv), backend=self._backend
        )
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        # Remove PKCS7 padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]

    def encrypt_array(self, arr: np.ndarray) -> bytes:
        """
        Encrypt a numpy array for secure disk caching.

        Args:
            arr: Numpy array to encrypt

        Returns:
            Encrypted bytes
        """
        # Serialize array to bytes
        buffer = io.BytesIO()
        np.save(buffer, arr, allow_pickle=False)
        array_bytes = buffer.getvalue()

        return self.encrypt(array_bytes)

    def decrypt_array(self, data: bytes) -> np.ndarray:
        """
        Decrypt a numpy array from encrypted cache.

        Args:
            data: Encrypted bytes

        Returns:
            Decrypted numpy array
        """
        array_bytes = self.decrypt(data)
        buffer = io.BytesIO(array_bytes)
        return np.load(buffer, allow_pickle=False)


# Global instance for convenience (lazy initialized)
_global_crypto: Optional[CryptoManager] = None


def get_crypto(key_hex: Optional[str] = None) -> CryptoManager:
    """
    Get or create a CryptoManager instance.

    If key_hex is provided, creates a new instance.
    Otherwise, returns/creates a global instance using env var.

    Args:
        key_hex: Optional hex-encoded key

    Returns:
        CryptoManager instance
    """
    global _global_crypto

    if key_hex is not None:
        return CryptoManager(key_hex=key_hex)

    if _global_crypto is None:
        _global_crypto = CryptoManager()

    return _global_crypto


__all__ = ["CryptoManager", "get_crypto", "ENV_KEY_NAME"]
