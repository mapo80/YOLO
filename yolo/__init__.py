"""
YOLO - Official Implementation with PyTorch Lightning.

Usage:
    # Training
    python -m yolo.cli fit --config yolo/config/experiment/default.yaml

    # Validation
    python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml

    # Or use the installed command
    yolo fit --config yolo/config/experiment/default.yaml
"""

# Suppress noisy warnings from dependencies
# Must be at the very top before any imports that might trigger them
import warnings

# pkg_resources deprecation warnings from torchmetrics
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*Deprecated call to.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*pkg_resources.*")

__all__ = [
    # Model
    "YOLO",
    "create_model",
    # Config
    "ModelConfig",
    "NMSConfig",
    # Utilities
    "Vec2Box",
    "Anc2Box",
    "bbox_nms",
    "create_converter",
]

_LAZY_ATTRS = {
    # Model
    "YOLO": ("yolo.model.yolo", "YOLO"),
    "create_model": ("yolo.model.yolo", "create_model"),
    # Config
    "ModelConfig": ("yolo.config.config", "ModelConfig"),
    "NMSConfig": ("yolo.config.config", "NMSConfig"),
    # Utilities
    "Vec2Box": ("yolo.utils.bounding_box_utils", "Vec2Box"),
    "Anc2Box": ("yolo.utils.bounding_box_utils", "Anc2Box"),
    "bbox_nms": ("yolo.utils.bounding_box_utils", "bbox_nms"),
    "create_converter": ("yolo.utils.bounding_box_utils", "create_converter"),
}


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(_LAZY_ATTRS.keys()))
