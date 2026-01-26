"""
Mock configuration for enabling/disabling mock components.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class MockConfig:
    """
    Configuration for mock components.

    Set via environment variables or programmatically.

    Environment variables:
        YOLO_MOCK_VERBOSE=1 - Enable verbose logging
        YOLO_MOCK_LOG_FREQ=10 - Log every N calls
        YOLO_MOCK_BCE=1 - Use mock BCE loss
        YOLO_MOCK_BOX=1 - Use mock box loss
        YOLO_MOCK_DFL=1 - Use mock DFL loss
        YOLO_MOCK_MATCHER=1 - Use mock matcher
        YOLO_MOCK_VEC2BOX=1 - Use mock vec2box
        YOLO_MOCK_EMA=1 - Use mock EMA
    """
    verbose: bool = field(default_factory=lambda: os.environ.get("YOLO_MOCK_VERBOSE", "0") == "1")
    log_freq: int = field(default_factory=lambda: int(os.environ.get("YOLO_MOCK_LOG_FREQ", "10")))

    # Individual component mocks
    mock_bce: bool = field(default_factory=lambda: os.environ.get("YOLO_MOCK_BCE", "0") == "1")
    mock_box: bool = field(default_factory=lambda: os.environ.get("YOLO_MOCK_BOX", "0") == "1")
    mock_dfl: bool = field(default_factory=lambda: os.environ.get("YOLO_MOCK_DFL", "0") == "1")
    mock_matcher: bool = field(default_factory=lambda: os.environ.get("YOLO_MOCK_MATCHER", "0") == "1")
    mock_vec2box: bool = field(default_factory=lambda: os.environ.get("YOLO_MOCK_VEC2BOX", "0") == "1")
    mock_ema: bool = field(default_factory=lambda: os.environ.get("YOLO_MOCK_EMA", "0") == "1")

    # Statistics tracking
    stats: Dict[str, Any] = field(default_factory=dict)

    _instance = None

    @classmethod
    def get_instance(cls) -> "MockConfig":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton instance."""
        cls._instance = None


# Global config instance
MOCK_CONFIG = MockConfig.get_instance()


def log_mock(component: str, message: str, force: bool = False):
    """Log a mock message if verbose mode is enabled."""
    config = MockConfig.get_instance()
    if config.verbose or force:
        print(f"[MOCK:{component}] {message}")


def should_log(component: str, call_count: int) -> bool:
    """Check if we should log this call based on frequency."""
    config = MockConfig.get_instance()
    return config.verbose and (call_count % config.log_freq == 0)
