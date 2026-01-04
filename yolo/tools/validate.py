"""
Standalone validation module for YOLO models.

This module provides functions for validating trained YOLO models on datasets
without requiring training, computing comprehensive detection metrics.

Usage:
    python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml
    python -m yolo.cli validate --checkpoint best.ckpt --data.root dataset/ --data.format yolo
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolo.data.datamodule import CocoDetectionWrapper, YOLODataModule, YOLOFormatDataset
from yolo.data.transforms import create_val_transforms
from yolo.training.module import YOLOModule
from yolo.utils.bounding_box_utils import Vec2Box, bbox_nms
from yolo.utils.metrics import DetMetrics

console = Console()


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the device to use for validation.

    Args:
        device: Device string ('cuda', 'mps', 'cpu') or None for auto-detect

    Returns:
        torch.device to use
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _collate_fn(batch):
    """Collate function for validation dataloader."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def create_validation_dataloader(
    data_root: str,
    data_format: str = "coco",
    val_images: str = "val2017",
    val_labels: Optional[str] = None,
    val_ann: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (640, 640),
) -> DataLoader:
    """
    Create a validation DataLoader from dataset configuration.

    Args:
        data_root: Root directory of the dataset
        data_format: Dataset format ('coco' or 'yolo')
        val_images: Path to validation images (relative to root)
        val_labels: Path to validation labels for YOLO format (relative to root)
        val_ann: Path to validation annotations for COCO format (relative to root)
        batch_size: Batch size for validation
        num_workers: Number of data loading workers
        image_size: Target image size (width, height)

    Returns:
        DataLoader for validation
    """
    root = Path(data_root)
    val_transforms = create_val_transforms(image_size=image_size)

    if data_format == "yolo":
        if val_labels is None:
            raise ValueError("val_labels is required for YOLO format")
        dataset = YOLOFormatDataset(
            images_dir=str(root / val_images),
            labels_dir=str(root / val_labels),
            transforms=val_transforms,
            image_size=image_size,
        )
    else:  # coco format
        if val_ann is None:
            raise ValueError("val_ann is required for COCO format")
        dataset = CocoDetectionWrapper(
            root=str(root / val_images),
            annFile=str(root / val_ann),
            transforms=val_transforms,
            image_size=image_size,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )


def _format_targets(targets: List, device: torch.device) -> List[Dict[str, torch.Tensor]]:
    """Format targets for metrics computation."""
    formatted = []
    for target in targets:
        if isinstance(target, dict):
            # Already in dict format from CocoDetection
            boxes = target.get("boxes", torch.zeros((0, 4), device=device))
            labels = target.get("labels", torch.zeros(0, dtype=torch.long, device=device))
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.tensor(boxes, device=device)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long, device=device)
            formatted.append({
                "boxes": boxes.to(device),
                "labels": labels.to(device),
            })
        elif isinstance(target, torch.Tensor):
            # Tensor format: [class, x_center, y_center, w, h] normalized
            if len(target) == 0:
                formatted.append({
                    "boxes": torch.zeros((0, 4), device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                })
            else:
                # Convert from center format to corner format if needed
                formatted.append({
                    "boxes": target[:, 1:5].to(device),
                    "labels": target[:, 0].long().to(device),
                })
        else:
            formatted.append({
                "boxes": torch.zeros((0, 4), device=device),
                "labels": torch.zeros(0, dtype=torch.long, device=device),
            })
    return formatted


def validate(
    checkpoint_path: str,
    data_root: str,
    data_format: str = "coco",
    val_images: str = "val2017",
    val_labels: Optional[str] = None,
    val_ann: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (640, 640),
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.6,
    max_detections: int = 300,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_plots: bool = True,
    save_json: bool = True,
    verbose: bool = True,
) -> Dict[str, Union[float, Dict]]:
    """
    Run validation on a trained YOLO model.

    Args:
        checkpoint_path: Path to model checkpoint (.ckpt file)
        data_root: Root directory of the dataset
        data_format: Dataset format ('coco' or 'yolo')
        val_images: Path to validation images (relative to root)
        val_labels: Path to validation labels for YOLO format
        val_ann: Path to validation annotations for COCO format
        batch_size: Batch size for validation
        num_workers: Number of data loading workers
        image_size: Target image size (width, height)
        conf_threshold: Confidence threshold for filtering predictions
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image
        device: Device to use ('cuda', 'mps', 'cpu' or None for auto)
        output_dir: Directory to save results
        save_plots: Whether to save metric plots
        save_json: Whether to save results as JSON
        verbose: Whether to print progress

    Returns:
        Dictionary with validation metrics
    """
    device = get_device(device)

    if verbose:
        console.print(f"\n[bold blue]Loading model from:[/] {checkpoint_path}")
        console.print(f"[bold blue]Device:[/] {device}")

    # Load model
    model = YOLOModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

    # Get class names from model
    num_classes = model.hparams.num_classes
    class_names = {i: str(i) for i in range(num_classes)}

    # Try to get class names from model if available
    if hasattr(model, "_class_names") and model._class_names is not None:
        class_names = model._class_names

    if verbose:
        console.print(f"[bold blue]Number of classes:[/] {num_classes}")
        console.print(f"[bold blue]Image size:[/] {image_size}")

    # Create dataloader
    dataloader = create_validation_dataloader(
        data_root=data_root,
        data_format=data_format,
        val_images=val_images,
        val_labels=val_labels,
        val_ann=val_ann,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )

    if verbose:
        console.print(f"[bold blue]Dataset size:[/] {len(dataloader.dataset)} images")
        console.print(f"[bold blue]Batch size:[/] {batch_size}")
        console.print()

    # Initialize metrics
    metrics = DetMetrics(names=class_names)

    # Create NMS configuration
    from dataclasses import dataclass

    @dataclass
    class NMSConfig:
        min_confidence: float
        min_iou: float
        max_bbox: int

    nms_cfg = NMSConfig(
        min_confidence=conf_threshold,
        min_iou=iou_threshold,
        max_bbox=max_detections,
    )

    # Initialize vec2box converter - use model's internal converter or create one
    vec2box = None
    if hasattr(model, "_vec2box") and model._vec2box is not None:
        vec2box = model._vec2box
    elif hasattr(model, "_model_cfg") and hasattr(model, "model"):
        # Create vec2box converter using model's config
        try:
            model_cfg = model._model_cfg
            if model_cfg is not None and hasattr(model_cfg, "anchor"):
                vec2box = Vec2Box(model.model, model_cfg.anchor, image_size, device)
                if verbose:
                    console.print("[green]Created Vec2Box converter from model config[/]")
        except Exception as e:
            if verbose:
                console.print(f"[yellow]Warning: Could not create Vec2Box: {e}[/]")

    # Run validation
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", disable=not verbose)
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Extract main predictions from output dict
            if isinstance(outputs, dict) and "Main" in outputs:
                main_output = outputs["Main"]
            else:
                main_output = outputs

            # Decode predictions using vec2box
            if vec2box is not None:
                pred_cls, pred_anc, pred_box = vec2box(main_output)

                # Apply NMS
                predictions = bbox_nms(pred_cls, pred_box, nms_cfg=nms_cfg)
            else:
                # Fallback: manual decoding
                predictions = decode_predictions(
                    main_output,
                    None,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    max_detections=max_detections,
                )

            # Format predictions for metrics (pred format: [class, x1, y1, x2, y2, confidence])
            preds_list = []
            for pred in predictions:
                if pred is not None and len(pred) > 0:
                    preds_list.append({
                        "boxes": pred[:, 1:5],     # x1, y1, x2, y2
                        "scores": pred[:, 5],      # confidence
                        "labels": pred[:, 0].long(),  # class
                    })
                else:
                    preds_list.append({
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros(0, device=device),
                        "labels": torch.zeros(0, dtype=torch.long, device=device),
                    })

            # Format targets for metrics
            targets_formatted = _format_targets(targets, device)

            # Update metrics
            metrics.update(preds_list, targets_formatted)

    # Process and compute final metrics
    output_path = Path(output_dir) if output_dir else Path("validation_results")
    output_path.mkdir(parents=True, exist_ok=True)

    results = metrics.process(
        save_dir=output_path if save_plots else None,
        plot=save_plots,
    )

    # Print results
    if verbose:
        print_results(results, metrics, class_names)

    # Save results
    if save_json:
        results_file = output_path / "results.json"
        full_results = {
            "metrics": results,
            "per_class": metrics.summary(),
            "config": {
                "checkpoint": checkpoint_path,
                "data_root": data_root,
                "data_format": data_format,
                "image_size": list(image_size),
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
            },
        }
        with open(results_file, "w") as f:
            json.dump(full_results, f, indent=2)
        if verbose:
            console.print(f"\n[green]Results saved to:[/] {results_file}")

    # Save CSV
    csv_file = output_path / "per_class_metrics.csv"
    with open(csv_file, "w") as f:
        f.write(metrics.to_csv())
    if verbose:
        console.print(f"[green]Per-class metrics saved to:[/] {csv_file}")

    return results


def decode_predictions(
    outputs: torch.Tensor,
    vec2box: Optional[Vec2Box],
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.6,
    max_detections: int = 300,
) -> List[torch.Tensor]:
    """
    Decode raw model outputs to bounding box predictions.

    Args:
        outputs: Raw model outputs
        vec2box: Vec2Box converter for decoding
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image

    Returns:
        List of prediction tensors [N, 6] with [x1, y1, x2, y2, conf, class]
    """
    from torchvision.ops import batched_nms

    predictions = []

    # Handle different output formats
    if isinstance(outputs, (list, tuple)):
        # Multi-head output - take the main detection output
        if len(outputs) > 0:
            outputs = outputs[0]

    # Outputs shape: [batch, num_predictions, 4 + num_classes] or similar
    batch_size = outputs.shape[0]

    for i in range(batch_size):
        pred = outputs[i]

        # Decode with vec2box if available
        if vec2box is not None:
            try:
                pred = vec2box(pred.unsqueeze(0))[0]
            except Exception:
                pass

        # Extract boxes, confidence, and classes
        if pred.shape[-1] > 6:
            # Format: [x1, y1, x2, y2, objectness, class_scores...]
            boxes = pred[:, :4]
            obj_conf = pred[:, 4:5] if pred.shape[-1] > 5 else torch.ones_like(pred[:, :1])
            class_scores = pred[:, 5:] if pred.shape[-1] > 5 else pred[:, 4:]

            # Get class predictions
            class_conf, class_pred = class_scores.max(1, keepdim=True)
            conf = obj_conf * class_conf

            # Filter by confidence
            mask = conf.squeeze(-1) > conf_threshold
            boxes = boxes[mask]
            conf = conf[mask]
            class_pred = class_pred[mask]

            if len(boxes) == 0:
                predictions.append(torch.zeros((0, 6), device=outputs.device))
                continue

            # Apply NMS
            keep = batched_nms(boxes, conf.squeeze(-1), class_pred.squeeze(-1), iou_threshold)
            keep = keep[:max_detections]

            # Combine results
            result = torch.cat([
                boxes[keep],
                conf[keep],
                class_pred[keep].float(),
            ], dim=1)
            predictions.append(result)
        else:
            # Already in [x1, y1, x2, y2, conf, class] format
            mask = pred[:, 4] > conf_threshold
            pred = pred[mask]
            if len(pred) > max_detections:
                pred = pred[:max_detections]
            predictions.append(pred)

    return predictions


def print_results(
    results: Dict[str, float],
    metrics: DetMetrics,
    class_names: Dict[int, str],
) -> None:
    """
    Print validation results in a formatted table.

    Args:
        results: Dictionary of aggregate metrics
        metrics: DetMetrics object with per-class data
        class_names: Dictionary mapping class indices to names
    """
    console.print("\n" + "=" * 60)
    console.print("[bold green]Validation Results[/]")
    console.print("=" * 60)

    # Summary metrics table
    summary_table = Table(title="Summary Metrics", show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="green")

    summary_table.add_row("mAP@0.5:0.95", f"{results['map']:.4f}")
    summary_table.add_row("mAP@0.5", f"{results['map50']:.4f}")
    summary_table.add_row("mAP@0.75", f"{results['map75']:.4f}")
    summary_table.add_row("Precision", f"{results['precision']:.4f}")
    summary_table.add_row("Recall", f"{results['recall']:.4f}")
    summary_table.add_row("F1 Score", f"{results['f1']:.4f}")

    console.print(summary_table)

    # Per-class metrics table
    per_class = metrics.summary()
    if per_class:
        class_table = Table(title="Per-Class Metrics", show_header=True, header_style="bold cyan")
        class_table.add_column("Class", style="cyan")
        class_table.add_column("Instances", justify="right")
        class_table.add_column("P", justify="right")
        class_table.add_column("R", justify="right")
        class_table.add_column("F1", justify="right")
        class_table.add_column("mAP50", justify="right")
        class_table.add_column("mAP50-95", justify="right")

        for row in per_class:
            class_table.add_row(
                str(row["Class"]),
                str(row["Instances"]),
                f"{row['P']:.3f}",
                f"{row['R']:.3f}",
                f"{row['F1']:.3f}",
                f"{row['mAP50']:.3f}",
                f"{row['mAP50-95']:.3f}",
            )

        console.print(class_table)

    console.print("=" * 60 + "\n")
