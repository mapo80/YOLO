"""
Class name loading utilities for YOLO datasets.

Provides functions to automatically load class names from dataset files:
- YOLO format: data.yaml with 'names' list
- COCO format: annotations JSON with 'categories'

This module is used by both training (YOLODataModule) and validation (validate.py)
to ensure consistent class name handling.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from yolo.utils.logger import logger


def load_class_names(
    data_root: Union[str, Path],
    data_format: str = "coco",
    ann_file: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> Dict[int, str]:
    """
    Load class names from dataset files.

    Automatically detects and loads class names from:
    - YOLO format: data.yaml in data_root with 'names' key
    - COCO format: annotation JSON with 'categories' list

    Args:
        data_root: Root directory of the dataset
        data_format: Dataset format - 'coco' or 'yolo'
        ann_file: Path to annotation file (relative to data_root), required for COCO format
        num_classes: Number of classes (used for fallback if loading fails)

    Returns:
        Dictionary mapping class index to class name {0: "name0", 1: "name1", ...}
        Falls back to numeric names {0: "0", 1: "1", ...} if loading fails.

    Examples:
        >>> # YOLO format
        >>> names = load_class_names("dataset/", data_format="yolo")
        >>> # COCO format
        >>> names = load_class_names("data/coco", data_format="coco", ann_file="annotations/instances_val2017.json")
    """
    root = Path(data_root)

    if data_format == "yolo":
        return _load_from_yolo_yaml(root, num_classes)
    else:
        return _load_from_coco_json(root, ann_file, num_classes)


def _load_from_yolo_yaml(root: Path, num_classes: Optional[int] = None) -> Dict[int, str]:
    """
    Load class names from YOLO data.yaml file.

    Looks for data.yaml in the dataset root directory.
    Expected format:
        names:
          - class_name_0
          - class_name_1
          ...
        nc: 7  # optional, number of classes

    Args:
        root: Dataset root directory
        num_classes: Fallback number of classes

    Returns:
        Dictionary mapping class index to class name
    """
    # Look for data.yaml in root
    data_yaml_path = root / "data.yaml"

    if not data_yaml_path.exists():
        logger.debug(f"No data.yaml found at {data_yaml_path}")
        return _create_numeric_names(num_classes)

    try:
        import yaml

        with open(data_yaml_path) as f:
            data_config = yaml.safe_load(f)

        if "names" not in data_config:
            logger.warning(f"data.yaml missing 'names' key: {data_yaml_path}")
            return _create_numeric_names(num_classes or data_config.get("nc"))

        names_list = data_config["names"]

        # Handle both list and dict formats
        if isinstance(names_list, list):
            class_names = {i: name for i, name in enumerate(names_list)}
        elif isinstance(names_list, dict):
            # Some YOLO formats use {0: "name", 1: "name", ...}
            class_names = {int(k): v for k, v in names_list.items()}
        else:
            logger.warning(f"Invalid 'names' format in data.yaml: {type(names_list)}")
            return _create_numeric_names(num_classes)

        logger.info(f"Loaded {len(class_names)} class names from {data_yaml_path}")
        return class_names

    except Exception as e:
        logger.warning(f"Failed to load class names from data.yaml: {e}")
        return _create_numeric_names(num_classes)


def _load_from_coco_json(
    root: Path,
    ann_file: Optional[str],
    num_classes: Optional[int] = None,
) -> Dict[int, str]:
    """
    Load class names from COCO annotation JSON.

    Expected format:
        {
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "bicycle"},
                ...
            ],
            ...
        }

    Note: COCO category IDs are typically 1-indexed and may have gaps.
    We map them to 0-indexed contiguous indices for training.

    Args:
        root: Dataset root directory
        ann_file: Path to annotation file (relative to root)
        num_classes: Fallback number of classes

    Returns:
        Dictionary mapping class index (0-indexed) to class name
    """
    if ann_file is None:
        logger.debug("No annotation file provided for COCO format")
        return _create_numeric_names(num_classes)

    ann_path = root / ann_file

    if not ann_path.exists():
        logger.warning(f"Annotation file not found: {ann_path}")
        return _create_numeric_names(num_classes)

    try:
        with open(ann_path) as f:
            coco_data = json.load(f)

        if "categories" not in coco_data:
            logger.warning(f"COCO JSON missing 'categories' key: {ann_path}")
            return _create_numeric_names(num_classes)

        # Sort categories by id to ensure consistent ordering
        categories = sorted(coco_data["categories"], key=lambda x: x["id"])

        # Map to 0-indexed contiguous indices
        class_names = {i: cat["name"] for i, cat in enumerate(categories)}

        logger.info(f"Loaded {len(class_names)} class names from {ann_path}")
        return class_names

    except Exception as e:
        logger.warning(f"Failed to load class names from COCO JSON: {e}")
        return _create_numeric_names(num_classes)


def _create_numeric_names(num_classes: Optional[int]) -> Dict[int, str]:
    """
    Create fallback numeric class names.

    Args:
        num_classes: Number of classes

    Returns:
        Dictionary {0: "0", 1: "1", ...}
    """
    if num_classes is None:
        return {}
    return {i: str(i) for i in range(num_classes)}


def class_names_to_list(class_names: Dict[int, str]) -> List[str]:
    """
    Convert class names dict to list.

    Args:
        class_names: Dictionary {0: "name0", 1: "name1", ...}

    Returns:
        List ["name0", "name1", ...]
    """
    if not class_names:
        return []
    max_idx = max(class_names.keys())
    return [class_names.get(i, str(i)) for i in range(max_idx + 1)]
