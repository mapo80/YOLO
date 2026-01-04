"""
YOLO CLI entry point.

This module allows running YOLO with: python -m yolo

Usage:
    python -m yolo fit --config yolo/config/experiment/default.yaml
"""

from yolo.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
