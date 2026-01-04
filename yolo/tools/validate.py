"""
Standalone validation module for YOLO models.

This module provides functions for validating trained YOLO models on datasets
without requiring training, computing comprehensive detection metrics.

Features:
- Full COCO metrics (mAP, AR, size-based AP)
- Operative metrics at production confidence threshold
- Interactive eval dashboard display
- Per-class metrics and confusion analysis
- Optional benchmark mode for latency/memory profiling

Usage:
    python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml
    python -m yolo.cli validate --checkpoint best.ckpt --data.root dataset/ --data.format yolo
    python -m yolo.cli validate --checkpoint best.ckpt --benchmark  # Include latency benchmark
"""

import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
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
from yolo.utils.eval_dashboard import EvalDashboard, EvalConfig

console = Console()


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    latency_mean_ms: float
    latency_std_ms: float
    fps: float
    memory_mb: Optional[float] = None
    model_size_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_mean_ms": self.latency_mean_ms,
            "latency_std_ms": self.latency_std_ms,
            "fps": self.fps,
            "memory_mb": self.memory_mb,
            "model_size_mb": self.model_size_mb,
        }


def get_gpu_memory() -> Optional[float]:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return None


def run_benchmark(
    model: torch.nn.Module,
    device: torch.device,
    image_size: Tuple[int, int],
    batch_size: int = 1,
    warmup: int = 10,
    runs: int = 100,
    checkpoint_path: Optional[str] = None,
) -> BenchmarkResult:
    """
    Run inference benchmark on a model.

    Args:
        model: Model to benchmark
        device: Device to run on
        image_size: Input image size (width, height)
        batch_size: Batch size for inference
        warmup: Number of warmup iterations
        runs: Number of benchmark runs
        checkpoint_path: Path to checkpoint for model size

    Returns:
        BenchmarkResult with timing statistics
    """
    model.eval()

    # Get model size from checkpoint if available
    model_size_mb = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        model_size_mb = os.path.getsize(checkpoint_path) / 1e6

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, image_size[1], image_size[0], device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Synchronize if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    fps = (batch_size * 1000) / mean_latency

    # Get memory usage
    memory_mb = get_gpu_memory()

    return BenchmarkResult(
        latency_mean_ms=mean_latency,
        latency_std_ms=std_latency,
        fps=fps,
        memory_mb=memory_mb,
        model_size_mb=model_size_mb,
    )


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
    conf_prod: float = 0.25,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_plots: bool = True,
    save_json: bool = True,
    verbose: bool = True,
    benchmark: bool = False,
    benchmark_warmup: int = 10,
    benchmark_runs: int = 100,
    skip_metrics: bool = False,
) -> Dict[str, Any]:
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
        conf_threshold: Confidence threshold for filtering predictions (NMS)
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image
        conf_prod: Production confidence threshold for operative metrics
        device: Device to use ('cuda', 'mps', 'cpu' or None for auto)
        output_dir: Directory to save results
        save_plots: Whether to save metric plots
        save_json: Whether to save results as JSON
        verbose: Whether to print progress
        benchmark: Whether to run latency/memory benchmark
        benchmark_warmup: Number of warmup iterations for benchmark
        benchmark_runs: Number of benchmark runs
        skip_metrics: Skip metrics computation (benchmark only mode)

    Returns:
        Dictionary with validation metrics including:
        - COCO metrics (map, map50, map75, ar_100, etc.)
        - Operative metrics (precision_at_conf, recall_at_conf, etc.)
        - Per-class data and top confusions
        - Deploy metrics (if benchmark=True)
    """
    device = get_device(device)

    if verbose:
        console.print(f"\n[bold blue]Loading model from:[/] {checkpoint_path}")
        console.print(f"[bold blue]Device:[/] {device}")

    # Load model
    model = YOLOModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

    # Get class names from model or dataset files
    num_classes = model.hparams.num_classes
    class_names = {i: str(i) for i in range(num_classes)}

    # Try to load class names based on format
    if data_format == "yolo":
        # YOLO format: load from data.yaml
        data_yaml_path = Path(data_root) / "data.yaml"
        if data_yaml_path.exists():
            try:
                import yaml
                with open(data_yaml_path) as f:
                    data_config = yaml.safe_load(f)
                if "names" in data_config:
                    names_list = data_config["names"]
                    class_names = {i: name for i, name in enumerate(names_list)}
            except Exception:
                pass  # Keep default numeric names
    else:
        # COCO format: load from annotation JSON
        if val_ann:
            ann_path = Path(data_root) / val_ann
            if ann_path.exists():
                try:
                    with open(ann_path) as f:
                        coco_data = json.load(f)
                    if "categories" in coco_data:
                        # COCO categories have 'id' and 'name' fields
                        # Sort by id to ensure correct mapping
                        categories = sorted(coco_data["categories"], key=lambda x: x["id"])
                        class_names = {i: cat["name"] for i, cat in enumerate(categories)}
                except Exception:
                    pass  # Keep default numeric names

    # Fallback: try to get class names from model if dataset files didn't provide them
    if all(isinstance(v, str) and v.isdigit() for v in class_names.values()):
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

    # Process metrics with extended COCO evaluation
    results = metrics.process(
        save_dir=output_path if save_plots else None,
        plot=save_plots,
        conf_prod=conf_prod,
    )

    # Run benchmark if requested
    benchmark_result = None
    if benchmark:
        if verbose:
            console.print("\n[bold yellow]Running inference benchmark...[/]")
        benchmark_result = run_benchmark(
            model=model,
            device=device,
            image_size=image_size,
            batch_size=1,  # Benchmark with batch=1 for latency
            warmup=benchmark_warmup,
            runs=benchmark_runs,
            checkpoint_path=checkpoint_path,
        )
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print results using eval dashboard
    if verbose:
        dashboard = EvalDashboard(EvalConfig(
            conf_prod=conf_prod,
            nms_iou=iou_threshold,
            max_det=max_detections,
        ))
        dashboard.print(
            metrics=results,
            num_images=len(dataloader.dataset),
            image_size=image_size,
            latency_ms=benchmark_result.latency_mean_ms if benchmark_result else None,
            memory_mb=benchmark_result.memory_mb if benchmark_result else None,
            model_size_mb=benchmark_result.model_size_mb if benchmark_result else None,
        )

    # Build full results dict
    full_results = {
        "metrics": {
            k: v for k, v in results.items()
            if not isinstance(v, (list, dict))
        },
        "per_class": results.get("per_class", []),
        "top_confusions": results.get("top_confusions", []),
        "threshold_sweep": results.get("threshold_sweep", {}),
        "config": {
            "checkpoint": checkpoint_path,
            "data_root": data_root,
            "data_format": data_format,
            "image_size": list(image_size),
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "conf_prod": conf_prod,
        },
    }

    # Add benchmark results
    if benchmark_result:
        full_results["deploy"] = benchmark_result.to_dict()

    # Save results
    if save_json:
        results_file = output_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(full_results, f, indent=2, default=str)
        if verbose:
            console.print(f"\n[green]Results saved to:[/] {results_file}")

    # Save CSV
    csv_file = output_path / "per_class_metrics.csv"
    with open(csv_file, "w") as f:
        f.write(metrics.to_csv())
    if verbose:
        console.print(f"[green]Per-class metrics saved to:[/] {csv_file}")

    if verbose:
        if save_plots:
            console.print(f"[green]Plots saved to:[/] {output_path}")

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


