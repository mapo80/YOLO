"""
Export module for YOLO models.

This module provides functions for exporting trained YOLO models to various formats.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch

from yolo.training.module import YOLOModule


def export_onnx(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640),
    opset_version: int = 17,
    simplify: bool = True,
    dynamic_batch: bool = False,
    half: bool = False,
    device: Optional[str] = None,
) -> str:
    """
    Export a YOLO model from Lightning checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to the .ckpt file
        output_path: Path for output .onnx file (default: same as checkpoint with .onnx extension)
        image_size: Input image size (height, width)
        opset_version: ONNX opset version (default: 17)
        simplify: Whether to simplify the ONNX model using onnx-simplifier
        dynamic_batch: Whether to use dynamic batch size
        half: Whether to export in FP16 (half precision)
        device: Device to use for export (auto-detected if None)

    Returns:
        Path to the exported ONNX file
    """
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # FP16 only supported on CUDA
    if half and device != "cuda":
        print("Warning: FP16 export requires CUDA. Falling back to FP32.")
        half = False

    # Load model from checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    module = YOLOModule.load_from_checkpoint(checkpoint_path, map_location=device)
    module.eval()
    module.to(device)

    model = module.model

    if half:
        model = model.half()

    # Determine output path
    if output_path is None:
        checkpoint_path = Path(checkpoint_path)
        output_path = checkpoint_path.with_suffix(".onnx")
    else:
        output_path = Path(output_path)

    # Create dummy input
    dtype = torch.float16 if half else torch.float32
    dummy_input = torch.randn(1, 3, image_size[0], image_size[1], dtype=dtype, device=device)

    # Define input/output names
    input_names = ["images"]
    output_names = ["output"]

    # Dynamic axes for batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # Export to ONNX
    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print(f"ONNX model saved to: {output_path}")

    # Simplify if requested
    if simplify:
        try:
            import onnx
            import onnxsim

            print("Simplifying ONNX model...")
            onnx_model = onnx.load(str(output_path))
            onnx_model_simplified, check = onnxsim.simplify(onnx_model)

            if check:
                onnx.save(onnx_model_simplified, str(output_path))
                print("ONNX model simplified successfully")
            else:
                print("Warning: ONNX simplification failed, keeping original model")

        except ImportError:
            print("Warning: onnx-simplifier not installed. Skipping simplification.")
            print("Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"Warning: ONNX simplification error: {e}")
            print("Keeping original model without simplification.")

    # Verify the model
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed")

        # Print model info
        print(f"\nModel Info:")
        print(f"  Input shape: {[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
        print(f"  Opset version: {onnx_model.opset_import[0].version}")

    except ImportError:
        print("Warning: onnx not installed. Skipping validation.")
        print("Install with: pip install onnx")

    return str(output_path)
