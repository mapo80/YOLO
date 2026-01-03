"""
Image loaders for YOLO datasets.

Provides extensible image loading with support for custom loaders (e.g., encrypted images).

Example:
    # Create a custom loader for encrypted images
    class EncryptedImageLoader(ImageLoader):
        def __init__(self, key: str):
            self.key = key

        def __call__(self, path: str) -> Image.Image:
            with open(path, 'rb') as f:
                encrypted_data = f.read()
            decrypted = my_decrypt(encrypted_data, self.key)
            return Image.open(io.BytesIO(decrypted)).convert("RGB")

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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

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
    Default image loader using PIL.

    This is the standard loader that mimics torchvision's CocoDetection behavior.
    Simply opens the image with PIL and converts to RGB.
    """

    def __call__(self, path: Union[str, Path]) -> Image.Image:
        """Load image using PIL and convert to RGB."""
        return Image.open(path).convert("RGB")
