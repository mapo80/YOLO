"""
YOLO Training CLI - LightningCLI entry point.

Usage:
    # Training
    python -m yolo.cli fit --config config/experiment/default.yaml
    python -m yolo.cli fit --config config/experiment/default.yaml --model.learning_rate=0.001

    # Validation
    python -m yolo.cli validate --ckpt_path=runs/best.ckpt

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

from lightning.pytorch.cli import LightningCLI

from yolo.data.datamodule import YOLODataModule
from yolo.training.module import YOLOModule


def train_main():
    """Run training/validation/test using LightningCLI."""
    cli = LightningCLI(
        YOLOModule,
        YOLODataModule,
        save_config_kwargs={"overwrite": True},
    )


def predict_main():
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
        action="store_true",
        default=True,
        help="Draw bounding boxes on images (default: True)",
    )
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="Do not draw bounding boxes",
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

    args = parser.parse_args(sys.argv[2:])  # Skip 'yolo.cli' and 'predict'

    # Handle --no-draw flag
    draw_boxes = not args.no_draw

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
        print(f"Error: Source not found: {source}")
        sys.exit(1)


def export_main():
    """Export model to ONNX format."""
    parser = argparse.ArgumentParser(
        description="YOLO Export - Export model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to ONNX
  python -m yolo.cli export --checkpoint best.ckpt

  # Export with custom output path
  python -m yolo.cli export --checkpoint best.ckpt --output model.onnx

  # Export with FP16 (CUDA only)
  python -m yolo.cli export --checkpoint best.ckpt --half

  # Export with dynamic batch size
  python -m yolo.cli export --checkpoint best.ckpt --dynamic-batch

  # Export without simplification
  python -m yolo.cli export --checkpoint best.ckpt --no-simplify
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
        help="Output path for ONNX file (default: checkpoint path with .onnx extension)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="onnx",
        choices=["onnx"],
        help="Export format (default: onnx)",
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
        help="Enable dynamic batch size",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16 half precision (CUDA only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto)",
    )

    args = parser.parse_args(sys.argv[2:])  # Skip 'yolo.cli' and 'export'

    # Import here to avoid slow startup
    from yolo.tools.export import export_onnx

    image_size = (args.size, args.size)

    if args.format == "onnx":
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
    else:
        print(f"Error: Unsupported format: {args.format}")
        sys.exit(1)


def main():
    """Main entry point - dispatch to training, predict, or export."""
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "export":
        export_main()
    else:
        train_main()


if __name__ == "__main__":
    main()
