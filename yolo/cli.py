"""
YOLO Training CLI - LightningCLI entry point.

Usage:
    # Training (COCO format - default)
    python -m yolo.cli fit --config yolo/config/experiment/default.yaml
    python -m yolo.cli fit --config yolo/config/experiment/default.yaml --model.learning_rate=0.001

    # Training (YOLO format)
    python -m yolo.cli fit --config config/yolo-format.yaml  # with data.format=yolo in YAML
    python -m yolo.cli fit --config config/default.yaml --data.format=yolo  # CLI override

    # Standalone Validation
    python -m yolo.cli validate --checkpoint best.ckpt --data.root dataset/ --data.format yolo
    python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml

    # Validation with benchmark (latency/memory)
    python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml --benchmark

    # Inference
    python -m yolo.cli predict --checkpoint best.ckpt --source image.jpg
    python -m yolo.cli predict --checkpoint best.ckpt --source images/ --output results/

    # Export
    python -m yolo.cli export --checkpoint best.ckpt --format onnx
    python -m yolo.cli export --checkpoint best.ckpt --format onnx --half --simplify

    # Cache Management
    python -m yolo.cli cache-create --config config.yaml --size 640 --encrypt
    python -m yolo.cli cache-export --cache-dir dataset/.yolo_cache_640x640_f1.0 -o cache.tar.gz
    python -m yolo.cli cache-import --archive cache.tar.gz --output /data/dataset/
    python -m yolo.cli cache-info --cache-dir dataset/.yolo_cache_640x640_f1.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def _coerce_system_exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolo",
        description="YOLO CLI (training via LightningCLI + utility subcommands). "
        "Run as `yolo ...` (installed) or `python -m yolo.cli ...`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train
  yolo fit --config yolo/config/experiment/default.yaml
  python -m yolo.cli fit --config yolo/config/experiment/default.yaml

  # Validate checkpoint on a dataset (standalone metrics)
  yolo validate --checkpoint runs/best.ckpt --config config.yaml

  # Validate with benchmark (latency/memory)
  yolo validate --checkpoint runs/best.ckpt --config config.yaml --benchmark

  # Inference on an image
  yolo predict --checkpoint runs/best.ckpt --source image.jpg --no-draw --save-json

  # Export checkpoint
  yolo export --checkpoint runs/best.ckpt --format onnx --simplify

  # QAT fine-tuning for INT8 quantization
  yolo qat-finetune --checkpoint best.ckpt --config config.yaml --epochs 20

  # Cache management
  yolo cache-create --config config.yaml --size 640 --encrypt
  yolo cache-export --cache-dir dataset/.yolo_cache_640x640_f1.0 -o cache.tar.gz
  yolo cache-import --archive cache.tar.gz --output /data/dataset/
  yolo cache-info --cache-dir dataset/.yolo_cache_640x640_f1.0
        """,
    )
    subparsers = parser.add_subparsers(title="commands", metavar="<command>")

    subparsers.add_parser("fit", help="Train a model (LightningCLI).")
    subparsers.add_parser("test", help="Test a model (LightningCLI).")
    subparsers.add_parser("predict", help="Run inference on images.")
    subparsers.add_parser("export", help="Export a checkpoint to ONNX/TFLite/SavedModel.")
    subparsers.add_parser("validate", help="Standalone validation with detection metrics and optional benchmark.")
    subparsers.add_parser("qat-finetune", help="Quantization-Aware Training fine-tuning for INT8 deployment.")

    # Cache management commands
    subparsers.add_parser("cache-create", help="Create dataset cache without training.")
    subparsers.add_parser("cache-export", help="Export cache to compressed archive.")
    subparsers.add_parser("cache-import", help="Import cache from archive.")
    subparsers.add_parser("cache-info", help="Show cache statistics and metadata.")

    return parser


class YOLOLightningCLI:
    """
    Custom LightningCLI wrapper that automatically adds required callbacks.

    Automatically adds:
    - TrainingSummaryCallback: Shows training configuration before fit
    - ClassNamesCallback: Loads class names from dataset for metrics display

    Propagates model.image_size to the datamodule automatically.
    """

    def __init__(self, model_class, datamodule_class, **kwargs):
        from lightning.pytorch.cli import LightningCLI

        class _CLI(LightningCLI):
            def add_arguments_to_parser(self, parser):
                """Link model.image_size to data.image_size for automatic propagation.

                This ensures data.image_size is set from model.image_size at instantiation time,
                BEFORE datamodule.setup() is called. Users should NOT specify data.image_size
                manually - it will be automatically set from model.image_size.
                """
                parser.link_arguments(
                    "model.image_size",
                    "data.image_size",
                    apply_on="instantiate"
                )

            def _add_callback_if_missing(self, callback_cls, *args, **cb_kwargs):
                """Add callback if not already present."""
                has_callback = any(
                    isinstance(cb, callback_cls) for cb in self.trainer.callbacks
                )
                if not has_callback:
                    self.trainer.callbacks.append(callback_cls(*args, **cb_kwargs))

            def _validate_image_size(self):
                """Validate that model and datamodule image_size match."""
                if self.model is None or self.datamodule is None:
                    return

                model_size = tuple(self.model.image_size)
                data_size = self.datamodule._image_size

                if model_size != data_size:
                    raise ValueError(
                        f"Image size mismatch: model.image_size={model_size} != data.image_size={data_size}. "
                        f"Do NOT specify data.image_size manually - it is automatically linked from model.image_size."
                    )

            def before_fit(self):
                """Add callbacks before training starts."""
                from yolo.training.callbacks import (
                    TrainingSummaryCallback,
                    ClassNamesCallback,
                )

                # Validate image_size consistency
                self._validate_image_size()

                self._add_callback_if_missing(TrainingSummaryCallback)
                self._add_callback_if_missing(ClassNamesCallback)

            def before_validate(self):
                """Add callbacks before validation starts."""
                from yolo.training.callbacks import ClassNamesCallback

                # Validate image_size consistency
                self._validate_image_size()

                self._add_callback_if_missing(ClassNamesCallback)

        _CLI(model_class, datamodule_class, **kwargs)


def train_main(argv: Optional[List[str]] = None) -> int:
    """Run training/validation/test using LightningCLI."""
    import torch
    from yolo.data.datamodule import YOLODataModule
    from yolo.training.module import YOLOModule

    # Enable Tensor Cores on NVIDIA GPUs (A100, A40, RTX 30xx/40xx, etc.)
    # This trades off some precision for significantly faster matmul operations
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    try:
        if argv is None:
            YOLOLightningCLI(
                YOLOModule,
                YOLODataModule,
                save_config_kwargs={"overwrite": True},
            )
        else:
            old_argv = sys.argv
            try:
                sys.argv = [old_argv[0], *argv]
                YOLOLightningCLI(
                    YOLOModule,
                    YOLODataModule,
                    save_config_kwargs={"overwrite": True},
                )
            finally:
                sys.argv = old_argv
    except SystemExit as exc:
        return _coerce_system_exit_code(exc)
    return 0


def predict_main(argv: Optional[List[str]] = None) -> int:
    """Run inference on images using a trained checkpoint."""
    parser = argparse.ArgumentParser(
        description="YOLO Inference - Run predictions on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python -m yolo.cli predict --checkpoint best.ckpt --source image.jpg

  # Directory of images
  python -m yolo.cli predict --checkpoint best.ckpt --source images/ --output results/

  # Without drawing boxes (JSON output only)
  python -m yolo.cli predict --checkpoint best.ckpt --source image.jpg --no-draw

  # Custom thresholds
  python -m yolo.cli predict --checkpoint best.ckpt --source image.jpg --conf 0.5 --iou 0.5
        """,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Path to image or directory of images",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for results (default: {source}_predictions/)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.65,
        help="IoU threshold for NMS (default: 0.65)",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image (default: 300)",
    )
    parser.add_argument(
        "--draw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw bounding boxes on images (default: True)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu, default: auto)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Path to JSON file with class names (list format)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save predictions to JSON file",
    )

    try:
        args = parser.parse_args(sys.argv[2:] if argv is None else argv)  # skip subcommand
    except SystemExit as exc:
        return _coerce_system_exit_code(exc)

    draw_boxes = bool(args.draw)

    # Import here to avoid slow startup for training commands
    from yolo.tools.inference import predict_directory, predict_image

    # Load class names if provided
    class_names = None
    if args.class_names:
        with open(args.class_names) as f:
            class_names = json.load(f)

    source = Path(args.source)
    image_size = (args.size, args.size)

    if source.is_file():
        # Single image
        if args.output is None:
            output = source.parent / f"{source.stem}_prediction{source.suffix}"
        else:
            output = Path(args.output)
            # If output looks like a directory (no extension or ends with /), create filename inside it
            if output.is_dir() or not output.suffix or str(args.output).endswith('/'):
                output.mkdir(parents=True, exist_ok=True)
                output = output / f"{source.stem}_prediction{source.suffix}"

        print(f"Running inference on: {source}")
        results = predict_image(
            checkpoint_path=args.checkpoint,
            image_path=str(source),
            output_path=str(output) if draw_boxes else None,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_detections=args.max_det,
            draw_boxes=draw_boxes,
            class_names=class_names,
            device=args.device,
            image_size=image_size,
        )

        print(f"\nDetections: {results['num_detections']}")
        for det in results["detections"]:
            print(f"  - {det['class_name']}: {det['confidence']:.2%}")

        if draw_boxes and "output_path" in results:
            print(f"\nOutput saved to: {results['output_path']}")

        if args.save_json:
            json_path = output.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"JSON saved to: {json_path}")

    elif source.is_dir():
        # Directory of images
        if args.output is None:
            output_dir = source.parent / f"{source.name}_predictions"
        else:
            output_dir = Path(args.output)

        print(f"Running inference on directory: {source}")
        print(f"Output directory: {output_dir}")

        results = predict_directory(
            checkpoint_path=args.checkpoint,
            input_dir=str(source),
            output_dir=str(output_dir),
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_detections=args.max_det,
            draw_boxes=draw_boxes,
            class_names=class_names,
            device=args.device,
            image_size=image_size,
        )

        total_detections = sum(r["num_detections"] for r in results)
        print(f"\nProcessed {len(results)} images, {total_detections} total detections")

        if args.save_json:
            json_path = output_dir / "predictions.json"
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"JSON saved to: {json_path}")

    else:
        print(f"Error: Source not found: {source}", file=sys.stderr)
        return 1
    return 0


def export_main(argv: Optional[List[str]] = None) -> int:
    """Export model to ONNX or TFLite format."""
    parser = argparse.ArgumentParser(
        description="YOLO Export - Export model to ONNX or TFLite format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to ONNX
  python -m yolo.cli export --checkpoint best.ckpt --format onnx

  # Export with custom output path
  python -m yolo.cli export --checkpoint best.ckpt --output model.onnx

  # Export with FP16 (CUDA only for ONNX)
  python -m yolo.cli export --checkpoint best.ckpt --half

  # Export with dynamic batch size (ONNX only)
  python -m yolo.cli export --checkpoint best.ckpt --dynamic-batch

  # Export to TFLite (FP32)
  python -m yolo.cli export --checkpoint best.ckpt --format tflite

  # Export to TFLite with FP16 quantization
  python -m yolo.cli export --checkpoint best.ckpt --format tflite --quantization fp16

  # Export to TFLite with INT8 quantization
  python -m yolo.cli export --checkpoint best.ckpt --format tflite \\
      --quantization int8 --calibration-images /path/to/images/

  # Export to TensorFlow SavedModel
  python -m yolo.cli export --checkpoint best.ckpt --format saved_model
        """,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path (default: checkpoint path with appropriate extension)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="onnx",
        choices=["onnx", "tflite", "saved_model"],
        help="Export format: onnx, tflite, or saved_model (default: onnx)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Simplify ONNX model using onnx-simplifier (default: False)",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Enable dynamic batch size (ONNX only)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16 half precision (ONNX with CUDA only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto)",
    )
    # TFLite-specific arguments
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="TFLite quantization mode: fp32, fp16, or int8 (default: fp32)",
    )
    parser.add_argument(
        "--calibration-images",
        type=str,
        default=None,
        help="Directory with calibration images for INT8 quantization",
    )
    parser.add_argument(
        "--num-calibration",
        type=int,
        default=100,
        help="Number of calibration images for INT8 (default: 100)",
    )
    parser.add_argument(
        "--xnnpack-optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply XNNPACK graph rewrites (SiLU→HardSwish, DFL Conv3D→Conv2D) (default: True)",
    )
    try:
        args = parser.parse_args(sys.argv[2:] if argv is None else argv)  # skip subcommand
    except SystemExit as exc:
        return _coerce_system_exit_code(exc)

    image_size = (args.size, args.size)

    if args.format == "onnx":
        # Import here to avoid slow startup
        from yolo.tools.export import export_onnx

        output_path = export_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            image_size=image_size,
            opset_version=args.opset,
            simplify=args.simplify,
            dynamic_batch=args.dynamic_batch,
            half=args.half,
            device=args.device,
        )
        print(f"\nExport complete: {output_path}")

    elif args.format == "tflite":
        from yolo.tools.export import export_tflite

        output_path = export_tflite(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            image_size=image_size,
            quantization=args.quantization,
            calibration_images=args.calibration_images,
            num_calibration_images=args.num_calibration,
            device=args.device,
            xnnpack_optimize=args.xnnpack_optimize,
        )
        print(f"\nExport complete: {output_path}")

    elif args.format == "saved_model":
        from yolo.tools.export import export_saved_model

        output_path = export_saved_model(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            image_size=image_size,
            device=args.device,
        )
        print(f"\nExport complete: {output_path}")

    else:
        print(f"Error: Unsupported format: {args.format}", file=sys.stderr)
        return 1
    return 0


def qat_finetune_main(argv: Optional[List[str]] = None) -> int:
    """Run QAT (Quantization-Aware Training) fine-tuning on a pre-trained model."""
    parser = argparse.ArgumentParser(
        description="YOLO QAT Fine-tuning - Quantization-Aware Training for INT8 deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic QAT fine-tuning
  python -m yolo.cli qat-finetune --checkpoint best.ckpt --config config.yaml

  # QAT with custom epochs and learning rate
  python -m yolo.cli qat-finetune --checkpoint best.ckpt --config config.yaml \\
      --epochs 20 --lr 0.0001

  # QAT with specific backend
  python -m yolo.cli qat-finetune --checkpoint best.ckpt --config config.yaml \\
      --backend qnnpack

  # QAT with validation every 5 epochs
  python -m yolo.cli qat-finetune --checkpoint best.ckpt --config config.yaml \\
      --val-every 5

  # Export QAT model to INT8 TFLite after training
  python -m yolo.cli qat-finetune --checkpoint best.ckpt --config config.yaml \\
      --export-tflite --calibration-images /path/to/images/
        """,
    )
    # Required arguments
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to pre-trained model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file (for data configuration)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of QAT fine-tuning epochs (default: 20)",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=0.0001,
        dest="learning_rate",
        help="Learning rate for QAT fine-tuning (default: 0.0001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )

    # QAT-specific arguments
    parser.add_argument(
        "--backend",
        type=str,
        default="qnnpack",
        choices=["qnnpack", "x86", "fbgemm"],
        help="Quantization backend: qnnpack (mobile), x86, fbgemm (default: qnnpack)",
    )
    parser.add_argument(
        "--freeze-bn-after",
        type=int,
        default=5,
        help="Freeze batch norm statistics after this epoch (default: 5)",
    )

    # Validation arguments
    parser.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="Run validation every N epochs (default: 1)",
    )

    # Output arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="runs/qat",
        help="Output directory for checkpoints and logs (default: runs/qat)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )

    # Export arguments
    parser.add_argument(
        "--export-tflite",
        action="store_true",
        help="Export to INT8 TFLite after training",
    )
    parser.add_argument(
        "--calibration-images",
        type=str,
        default=None,
        help="Directory with calibration images for TFLite INT8 export",
    )
    parser.add_argument(
        "--num-calibration",
        type=int,
        default=100,
        help="Number of calibration images for TFLite export (default: 100)",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu, default: auto)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)",
    )

    try:
        args = parser.parse_args(sys.argv[2:] if argv is None else argv)
    except SystemExit as exc:
        return _coerce_system_exit_code(exc)

    # Import dependencies
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
    from omegaconf import OmegaConf

    from yolo.data.datamodule import YOLODataModule
    from yolo.training.qat_module import QATModule

    # Load config
    print(f"\n{'=' * 60}")
    print("🔧 YOLO QAT Fine-tuning")
    print("=" * 60)
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Config: {args.config}")
    print(f"   Backend: {args.backend}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.learning_rate}")
    print("=" * 60 + "\n")

    config = OmegaConf.load(args.config)

    # Extract data configuration
    if not hasattr(config, "data"):
        print("Error: Config file must have 'data' section", file=sys.stderr)
        return 1

    data_cfg = config.data

    # Get image size from checkpoint
    # Skip pretrained weight loading since checkpoint already has trained weights
    from yolo.training.module import YOLOModule
    base_module = YOLOModule.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        weight_path=None,  # Don't load pretrained weights, use checkpoint state_dict
    )
    image_size = tuple(base_module.hparams.image_size)
    del base_module

    # Create datamodule with all parameters from config
    # CLI args override config values if specified
    batch_size = getattr(data_cfg, "batch_size", args.batch_size)
    num_workers = getattr(data_cfg, "num_workers", args.workers)

    datamodule = YOLODataModule(
        root=data_cfg.root,
        format=getattr(data_cfg, "format", "coco"),
        train_images=getattr(data_cfg, "train_images", None),
        val_images=getattr(data_cfg, "val_images", None),
        train_labels=getattr(data_cfg, "train_labels", None),
        val_labels=getattr(data_cfg, "val_labels", None),
        train_ann=getattr(data_cfg, "train_ann", None),
        val_ann=getattr(data_cfg, "val_ann", None),
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        # Augmentation parameters from config (defaults to disabled for QAT)
        mosaic_prob=getattr(data_cfg, "mosaic_prob", 0.0),
        mosaic_9_prob=getattr(data_cfg, "mosaic_9_prob", 0.0),
        mixup_prob=getattr(data_cfg, "mixup_prob", 0.0),
        flip_lr=getattr(data_cfg, "flip_lr", 0.0),
        flip_ud=getattr(data_cfg, "flip_ud", 0.0),
        degrees=getattr(data_cfg, "degrees", 0.0),
        translate=getattr(data_cfg, "translate", 0.0),
        scale=getattr(data_cfg, "scale", 0.0),
        shear=getattr(data_cfg, "shear", 0.0),
        perspective=getattr(data_cfg, "perspective", 0.0),
        hsv_h=getattr(data_cfg, "hsv_h", 0.0),
        hsv_s=getattr(data_cfg, "hsv_s", 0.0),
        hsv_v=getattr(data_cfg, "hsv_v", 0.0),
    )

    # Create QAT module
    qat_module = QATModule(
        checkpoint_path=args.checkpoint,
        backend=args.backend,
        learning_rate=args.learning_rate,
        freeze_bn_after_epoch=args.freeze_bn_after,
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(args.output) / (args.name or "qat_finetune"),
            filename="qat-{epoch:02d}-{val_mAP:.4f}",
            monitor="val/mAP",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.output,
        name=args.name or "qat_finetune",
    )

    # Determine accelerator
    if args.device:
        accelerator = args.device
        devices = 1
    else:
        import torch
        if torch.cuda.is_available():
            accelerator = "cuda"
            devices = args.gpus
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            devices = 1

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=args.val_every,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Train
    print("\n🚀 Starting QAT fine-tuning...\n")
    trainer.fit(qat_module, datamodule)

    # Get best checkpoint path
    best_ckpt = callbacks[0].best_model_path
    print(f"\n✅ QAT training complete!")
    print(f"   Best checkpoint: {best_ckpt}")
    print(f"   Best mAP: {callbacks[0].best_model_score:.4f}")

    # Export to TFLite if requested
    if args.export_tflite:
        print("\n📦 Exporting to INT8 TFLite...")

        if args.calibration_images is None:
            print("Error: --calibration-images required for TFLite export", file=sys.stderr)
            return 1

        from yolo.tools.export import export_tflite

        # Load best checkpoint
        best_module = QATModule.load_from_checkpoint(best_ckpt)
        best_module.convert_to_quantized()

        # Save quantized PyTorch model first
        pt_output = Path(args.output) / (args.name or "qat_finetune") / "quantized_model.pt"
        best_module.export_quantized(str(pt_output), image_size)

        # Export to TFLite
        tflite_output = Path(args.output) / (args.name or "qat_finetune") / "model_int8.tflite"
        export_tflite(
            checkpoint_path=best_ckpt,
            output_path=str(tflite_output),
            image_size=image_size,
            quantization="int8",
            calibration_images=args.calibration_images,
            num_calibration_images=args.num_calibration,
        )

        print(f"   TFLite model: {tflite_output}")

    return 0


def validate_main(argv: Optional[List[str]] = None) -> int:
    """Run standalone validation on a trained model."""
    parser = argparse.ArgumentParser(
        description="YOLO Validate - Run validation on a trained model with eval dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate with config file
  python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml

  # Validate with direct parameters (YOLO format)
  python -m yolo.cli validate --checkpoint best.ckpt \\
      --data.root dataset/ --data.format yolo \\
      --data.val_images valid/images --data.val_labels valid/labels

  # Validate with direct parameters (COCO format)
  python -m yolo.cli validate --checkpoint best.ckpt \\
      --data.root dataset/ --data.format coco \\
      --data.val_images val2017 --data.val_ann annotations/instances_val.json

  # Save plots and JSON
  python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml \\
      --output results/ --save-plots --save-json

  # Validate with benchmark (latency/memory)
  python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml --benchmark

  # Benchmark with custom warmup/runs
  python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml \\
      --benchmark --benchmark-warmup 20 --benchmark-runs 200
        """,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (for data configuration)",
    )
    # Data configuration (can override config file)
    parser.add_argument(
        "--data.root",
        type=str,
        default=None,
        dest="data_root",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--data.format",
        type=str,
        choices=["coco", "yolo"],
        default="coco",
        dest="data_format",
        help="Dataset format: coco or yolo (default: coco)",
    )
    parser.add_argument(
        "--data.val_images",
        type=str,
        default=None,
        dest="val_images",
        help="Path to validation images (relative to root)",
    )
    parser.add_argument(
        "--data.val_labels",
        type=str,
        default=None,
        dest="val_labels",
        help="Path to validation labels for YOLO format (relative to root)",
    )
    parser.add_argument(
        "--data.val_ann",
        type=str,
        default=None,
        dest="val_ann",
        help="Path to validation annotations for COCO format (relative to root)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for validation (default: 16)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold (default: 0.001)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for NMS (default: 0.6)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu, default: auto)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="validation_results",
        help="Output directory for results (default: validation_results)",
    )
    plots_group = parser.add_mutually_exclusive_group()
    plots_group.add_argument(
        "--save-plots",
        dest="save_plots",
        action="store_true",
        help="Save metric plots (default: True)",
    )
    plots_group.add_argument(
        "--no-plots",
        dest="save_plots",
        action="store_false",
        help="Do not save metric plots",
    )
    parser.set_defaults(save_plots=True)

    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument(
        "--save-json",
        dest="save_json",
        action="store_true",
        help="Save results as JSON (default: True)",
    )
    json_group.add_argument(
        "--no-json",
        dest="save_json",
        action="store_false",
        help="Do not save results as JSON",
    )
    parser.set_defaults(save_json=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    # Production confidence threshold for operative metrics
    parser.add_argument(
        "--conf-prod",
        type=float,
        default=0.25,
        help="Production confidence threshold for operative metrics (default: 0.25)",
    )
    # Benchmark arguments
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency/memory benchmark",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=10,
        help="Warmup iterations for benchmark (default: 10)",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=100,
        help="Number of benchmark runs (default: 100)",
    )

    try:
        args = parser.parse_args(sys.argv[2:] if argv is None else argv)  # skip subcommand
    except SystemExit as exc:
        return _coerce_system_exit_code(exc)

    # Load config if provided
    data_root = args.data_root
    data_format = args.data_format
    val_images = args.val_images
    val_labels = args.val_labels
    val_ann = args.val_ann

    if args.config:
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.config)
        if hasattr(config, "data"):
            data_cfg = config.data
            data_root = data_root or getattr(data_cfg, "root", None)
            data_format = getattr(data_cfg, "format", data_format)
            val_images = val_images or getattr(data_cfg, "val_images", None)
            val_labels = val_labels or getattr(data_cfg, "val_labels", None)
            val_ann = val_ann or getattr(data_cfg, "val_ann", None)

    # Validate required parameters
    if data_root is None:
        print("Error: --data.root is required (or provide --config with data.root)", file=sys.stderr)
        return 1

    if val_images is None:
        print("Error: --data.val_images is required", file=sys.stderr)
        return 1

    if data_format == "yolo" and val_labels is None:
        print("Error: --data.val_labels is required for YOLO format", file=sys.stderr)
        return 1

    if data_format == "coco" and val_ann is None:
        print("Error: --data.val_ann is required for COCO format", file=sys.stderr)
        return 1

    # Import and run validation
    from yolo.tools.validate import validate

    image_size = (args.size, args.size)
    save_plots = bool(args.save_plots)

    validate(
        checkpoint_path=args.checkpoint,
        data_root=data_root,
        data_format=data_format,
        val_images=val_images,
        val_labels=val_labels,
        val_ann=val_ann,
        batch_size=args.batch_size,
        num_workers=args.workers,
        image_size=image_size,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        conf_prod=args.conf_prod,
        device=args.device,
        output_dir=args.output,
        save_plots=save_plots,
        save_json=bool(args.save_json),
        verbose=True,
        benchmark=args.benchmark,
        benchmark_warmup=args.benchmark_warmup,
        benchmark_runs=args.benchmark_runs,
    )
    return 0


# =============================================================================
# Cache Management Commands
# =============================================================================


def cache_create_main(argv: Optional[List[str]] = None) -> int:
    """Create dataset cache without running training."""
    parser = argparse.ArgumentParser(
        description="Create LMDB cache from dataset without training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create cache from config file
  yolo cache-create --config config.yaml --size 640

  # Create encrypted cache
  export YOLO_ENCRYPTION_KEY=$(python -c "import os; print(os.urandom(32).hex())")
  yolo cache-create --config config.yaml --size 640 --encrypt

  # Create cache with direct parameters (YOLO format)
  yolo cache-create --data.root /path/to/dataset --data.format yolo \\
      --data.train_images train/images --data.train_labels train/labels \\
      --size 640 --encrypt

  # Create cache with direct parameters (COCO format)
  yolo cache-create --data.root /path/to/dataset --data.format coco \\
      --data.train_images train --data.train_ann annotations/train.json \\
      --size 640
        """,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file with data settings",
    )
    parser.add_argument(
        "--data.root",
        type=str,
        dest="data_root",
        default=None,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--data.format",
        type=str,
        dest="data_format",
        default="coco",
        choices=["coco", "yolo"],
        help="Dataset format (default: coco)",
    )
    parser.add_argument(
        "--data.train_images",
        type=str,
        dest="train_images",
        default=None,
        help="Path to training images (relative to data.root)",
    )
    parser.add_argument(
        "--data.val_images",
        type=str,
        dest="val_images",
        default=None,
        help="Path to validation images (relative to data.root)",
    )
    parser.add_argument(
        "--data.train_labels",
        type=str,
        dest="train_labels",
        default=None,
        help="Path to training labels for YOLO format",
    )
    parser.add_argument(
        "--data.val_labels",
        type=str,
        dest="val_labels",
        default=None,
        help="Path to validation labels for YOLO format",
    )
    parser.add_argument(
        "--data.train_ann",
        type=str,
        dest="train_ann",
        default=None,
        help="Path to training annotations for COCO format",
    )
    parser.add_argument(
        "--data.val_ann",
        type=str,
        dest="val_ann",
        default=None,
        help="Path to validation annotations for COCO format",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Image size for cache (default: from config model.image_size, or 640)",
    )
    parser.add_argument(
        "--encrypt",
        action="store_true",
        help="Encrypt cache with AES-256 (requires YOLO_ENCRYPTION_KEY env var)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "val", "both"],
        help="Which split to cache (default: both)",
    )
    parser.add_argument(
        "--data.fraction",
        type=float,
        dest="data_fraction",
        default=1.0,
        help="Fraction of data to cache (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory where cache will be created (default: data.root). "
             "Useful when dataset is on a slow/external volume.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Enable LMDB fsync for crash safety. Default: disabled for external volume compatibility.",
    )

    try:
        args = parser.parse_args(sys.argv[2:] if argv is None else argv)
    except SystemExit as exc:
        return _coerce_system_exit_code(exc)

    # Load config if provided
    data_root = args.data_root
    data_format = args.data_format
    train_images = args.train_images
    val_images = args.val_images
    train_labels = args.train_labels
    val_labels = args.val_labels
    train_ann = args.train_ann
    val_ann = args.val_ann
    data_fraction = args.data_fraction
    train_split = None
    val_split = None

    # Default image size
    image_size = args.size

    if args.config:
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.config)
        if hasattr(config, "data"):
            data_cfg = config.data
            data_root = data_root or getattr(data_cfg, "root", None)
            data_format = getattr(data_cfg, "format", data_format)
            train_images = train_images or getattr(data_cfg, "train_images", None)
            val_images = val_images or getattr(data_cfg, "val_images", None)
            train_labels = train_labels or getattr(data_cfg, "train_labels", None)
            val_labels = val_labels or getattr(data_cfg, "val_labels", None)
            train_ann = train_ann or getattr(data_cfg, "train_ann", None)
            val_ann = val_ann or getattr(data_cfg, "val_ann", None)
            data_fraction = getattr(data_cfg, "data_fraction", data_fraction)
            train_split = getattr(data_cfg, "train_split", None)
            val_split = getattr(data_cfg, "val_split", None)

        # Read image_size from config if --size not provided
        if image_size is None:
            # Try model.image_size first (common in training configs)
            if hasattr(config, "model") and hasattr(config.model, "image_size"):
                cfg_size = config.model.image_size
                # Handle [H, W] list format - use first dimension (assume square)
                # Note: OmegaConf ListConfig doesn't match isinstance(list/tuple)
                try:
                    image_size = int(cfg_size[0])
                except (TypeError, IndexError):
                    image_size = int(cfg_size)
            # Fallback to data.image_size
            elif hasattr(config, "data") and hasattr(config.data, "image_size"):
                cfg_size = config.data.image_size
                try:
                    image_size = int(cfg_size[0])
                except (TypeError, IndexError):
                    image_size = int(cfg_size)

    # Validate required parameters
    if image_size is None:
        print(
            "Error: Image size not specified.\n"
            "Provide --size or set model.image_size in your config YAML.",
            file=sys.stderr
        )
        return 1
    if data_root is None:
        print("Error: --data.root is required (or provide --config with data.root)", file=sys.stderr)
        return 1

    # Import and create cache
    from yolo.tools.cache_archive import create_cache

    try:
        cache_path = create_cache(
            data_root=Path(data_root),
            data_format=data_format,
            image_size=(image_size, image_size),
            train_images=train_images,
            val_images=val_images,
            train_labels=train_labels,
            val_labels=val_labels,
            train_ann=train_ann,
            val_ann=val_ann,
            train_split=train_split,
            val_split=val_split,
            encrypt=args.encrypt,
            workers=args.workers,
            split=args.split,
            data_fraction=data_fraction,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            sync=args.sync,
        )
        print(f"\nCache created successfully: {cache_path}")
        return 0
    except Exception as e:
        print(f"Error creating cache: {e}", file=sys.stderr)
        return 1


def cache_export_main(argv: Optional[List[str]] = None) -> int:
    """Export cache directory to compressed archive."""
    parser = argparse.ArgumentParser(
        description="Export cache to compressed archive for transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export cache to tar.gz
  yolo cache-export --cache-dir dataset/.yolo_cache_640x640_f1.0 --output cache.tar.gz

  # Export without compression (faster, larger file)
  yolo cache-export --cache-dir dataset/.yolo_cache_640x640_f1.0 \\
      --output cache.tar --compression none
        """,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Path to cache directory (.yolo_cache_*)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output archive path (default: {cache_dir}.tar.gz)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["gzip", "none"],
        help="Compression type (default: gzip)",
    )

    try:
        args = parser.parse_args(sys.argv[2:] if argv is None else argv)
    except SystemExit as exc:
        return _coerce_system_exit_code(exc)

    from yolo.tools.cache_archive import export_cache

    try:
        output_path = export_cache(
            cache_dir=Path(args.cache_dir),
            output_path=Path(args.output) if args.output else None,
            compression=args.compression,
        )
        print(f"\nCache exported to: {output_path}")
        return 0
    except Exception as e:
        print(f"Error exporting cache: {e}", file=sys.stderr)
        return 1


def cache_import_main(argv: Optional[List[str]] = None) -> int:
    """Import cache from archive to target directory."""
    parser = argparse.ArgumentParser(
        description="Import cache from archive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import cache to current directory
  yolo cache-import --archive cache.tar.gz

  # Import cache to specific directory
  yolo cache-import --archive cache.tar.gz --output /data/dataset/
        """,
    )
    parser.add_argument(
        "--archive",
        type=str,
        required=True,
        help="Path to cache archive (.tar.gz or .tar)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Target directory for extraction (default: current directory)",
    )

    try:
        args = parser.parse_args(sys.argv[2:] if argv is None else argv)
    except SystemExit as exc:
        return _coerce_system_exit_code(exc)

    from yolo.tools.cache_archive import import_cache

    try:
        cache_path = import_cache(
            archive_path=Path(args.archive),
            output_dir=Path(args.output) if args.output else None,
        )
        print(f"\nCache imported to: {cache_path}")
        return 0
    except Exception as e:
        print(f"Error importing cache: {e}", file=sys.stderr)
        return 1


def cache_info_main(argv: Optional[List[str]] = None) -> int:
    """Display cache statistics and metadata."""
    parser = argparse.ArgumentParser(
        description="Show cache statistics and metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show cache info
  yolo cache-info --cache-dir dataset/.yolo_cache_640x640_f1.0
        """,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Path to cache directory (.yolo_cache_*)",
    )

    try:
        args = parser.parse_args(sys.argv[2:] if argv is None else argv)
    except SystemExit as exc:
        return _coerce_system_exit_code(exc)

    from yolo.tools.cache_archive import print_cache_info

    try:
        print_cache_info(Path(args.cache_dir))
        return 0
    except Exception as e:
        print(f"Error reading cache info: {e}", file=sys.stderr)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point - dispatch to training (LightningCLI) or utility subcommands."""
    args = sys.argv[1:] if argv is None else argv

    if not args or args[0] in {"-h", "--help"}:
        _root_parser().print_help()
        return 0

    # Utility subcommands.
    cmd = args[0]
    if cmd == "predict":
        return predict_main(args[1:])
    if cmd == "export":
        return export_main(args[1:])
    if cmd == "validate":
        return validate_main(args[1:])
    if cmd == "qat-finetune":
        return qat_finetune_main(args[1:])

    # Cache management commands
    if cmd == "cache-create":
        return cache_create_main(args[1:])
    if cmd == "cache-export":
        return cache_export_main(args[1:])
    if cmd == "cache-import":
        return cache_import_main(args[1:])
    if cmd == "cache-info":
        return cache_info_main(args[1:])

    # Fall back to training CLI (supports global options before the subcommand).
    if cmd.startswith("-"):
        return train_main(args)

    # If it's not a known command, provide a clearer error than LightningCLI.
    known = {"fit", "test", "predict", "export", "validate", "qat-finetune",
             "cache-create", "cache-export", "cache-import", "cache-info"}
    if cmd not in known:
        print(f"Error: Unknown command: {cmd}\n", file=sys.stderr)
        _root_parser().print_help()
        return 2

    return train_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
