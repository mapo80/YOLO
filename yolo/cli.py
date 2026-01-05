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
        """,
    )
    subparsers = parser.add_subparsers(title="commands", metavar="<command>")

    subparsers.add_parser("fit", help="Train a model (LightningCLI).")
    subparsers.add_parser("test", help="Test a model (LightningCLI).")
    subparsers.add_parser("predict", help="Run inference on images.")
    subparsers.add_parser("export", help="Export a checkpoint to ONNX/TFLite/SavedModel.")
    subparsers.add_parser("validate", help="Standalone validation with detection metrics and optional benchmark.")

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
                """No argument linking needed - image_size is propagated in before_fit."""
                pass

            def _add_callback_if_missing(self, callback_cls, *args, **cb_kwargs):
                """Add callback if not already present."""
                has_callback = any(
                    isinstance(cb, callback_cls) for cb in self.trainer.callbacks
                )
                if not has_callback:
                    self.trainer.callbacks.append(callback_cls(*args, **cb_kwargs))

            def before_fit(self):
                """Add callbacks and propagate image_size before training starts."""
                from yolo.training.callbacks import (
                    TrainingSummaryCallback,
                    ClassNamesCallback,
                )

                # Propagate image_size from model to datamodule
                # This ensures data._image_size matches model.image_size
                if hasattr(self.model, 'hparams') and hasattr(self.datamodule, '_image_size'):
                    model_image_size = self.model.hparams.image_size
                    self.datamodule._image_size = tuple(model_image_size)

                self._add_callback_if_missing(TrainingSummaryCallback)
                self._add_callback_if_missing(ClassNamesCallback)

            def before_validate(self):
                """Add callbacks and propagate image_size before validation starts."""
                from yolo.training.callbacks import ClassNamesCallback

                # Propagate image_size from model to datamodule
                if hasattr(self.model, 'hparams') and hasattr(self.datamodule, '_image_size'):
                    model_image_size = self.model.hparams.image_size
                    self.datamodule._image_size = tuple(model_image_size)

                self._add_callback_if_missing(ClassNamesCallback)

        _CLI(model_class, datamodule_class, **kwargs)


def train_main(argv: Optional[List[str]] = None) -> int:
    """Run training/validation/test using LightningCLI."""
    from yolo.data.datamodule import YOLODataModule
    from yolo.training.module import YOLOModule

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

    # Fall back to training CLI (supports global options before the subcommand).
    if cmd.startswith("-"):
        return train_main(args)

    # If it's not a known command, provide a clearer error than LightningCLI.
    known = {"fit", "test", "predict", "export", "validate"}
    if cmd not in known:
        print(f"Error: Unknown command: {cmd}\n", file=sys.stderr)
        _root_parser().print_help()
        return 2

    return train_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
