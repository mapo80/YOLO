"""
Export module for YOLO models.

This module provides functions for exporting trained YOLO models to various formats
including ONNX and TFLite with quantization support.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn

from yolo.training.module import YOLOModule
from yolo.utils.bounding_box_utils import generate_anchors


class YOLOExportWrapper(nn.Module):
    """Wrapper for ONNX export compatible with original YOLOv9 format.

    This wrapper processes the model output to produce a single tensor
    with shape [B, num_detections, 4+num_classes] containing:
    - First 4 values: xyxy bounding box coordinates (absolute pixels)
    - Remaining values: class probabilities (post-sigmoid)
    """

    def __init__(self, model: nn.Module, image_size: Tuple[int, int], strides: List[int]):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.strides = strides

        # Pre-compute anchor grid and scaler for box decoding
        # image_size is (H, W), generate_anchors expects (W, H)
        anchor_grid, scaler = generate_anchors((image_size[1], image_size[0]), strides)
        self.register_buffer("anchor_grid", anchor_grid)
        self.register_buffer("scaler", scaler)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through base model
        outputs = self.model(x)

        # Get only Main head output (ignore AUX)
        main_outputs = outputs["Main"]

        # Process each scale
        all_boxes = []
        all_cls = []

        for layer_output in main_outputs:
            pred_cls, pred_anc, pred_box = layer_output
            B, C, H, W = pred_cls.shape

            # Reshape: B C H W -> B (H*W) C
            pred_cls = pred_cls.permute(0, 2, 3, 1).reshape(B, -1, C)
            pred_box = pred_box.permute(0, 2, 3, 1).reshape(B, -1, 4)

            all_cls.append(pred_cls)
            all_boxes.append(pred_box)

        # Concatenate all scales
        preds_cls = torch.cat(all_cls, dim=1)  # [B, N, num_classes]
        preds_box = torch.cat(all_boxes, dim=1)  # [B, N, 4]

        # Convert boxes from relative LTRB to absolute xyxy
        pred_LTRB = preds_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        boxes_xyxy = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)

        # Apply sigmoid to class scores
        cls_scores = preds_cls.sigmoid()

        # Final output: [B, N, 4+num_classes]
        return torch.cat([boxes_xyxy, cls_scores], dim=-1)


def _check_dependencies(verbose: bool = True) -> Dict[str, Any]:
    """
    Check and display status of export dependencies.

    Args:
        verbose: Whether to print status to console

    Returns:
        Dictionary with dependency status
    """
    status = {
        "onnx": {"installed": False, "version": None, "error": None},
        "onnxsim": {"installed": False, "version": None, "error": None},
        "tensorflow": {"installed": False, "version": None, "error": None},
        "onnx2tf": {"installed": False, "version": None, "error": None},
    }

    # Check onnx
    try:
        import onnx
        status["onnx"]["installed"] = True
        status["onnx"]["version"] = onnx.__version__
    except ImportError as e:
        status["onnx"]["error"] = str(e)

    # Check onnxsim
    try:
        import onnxsim
        status["onnxsim"]["installed"] = True
        status["onnxsim"]["version"] = getattr(onnxsim, "__version__", "unknown")
    except ImportError as e:
        status["onnxsim"]["error"] = str(e)

    # Check tensorflow
    try:
        import tensorflow as tf
        status["tensorflow"]["installed"] = True
        status["tensorflow"]["version"] = tf.__version__
    except ImportError as e:
        status["tensorflow"]["error"] = str(e)

    # Check onnx2tf
    try:
        import onnx2tf
        status["onnx2tf"]["installed"] = True
        status["onnx2tf"]["version"] = getattr(onnx2tf, "__version__", "unknown")
    except (ImportError, AttributeError) as e:
        status["onnx2tf"]["error"] = str(e)

    if verbose:
        print("\n" + "=" * 60)
        print("📦 Export Dependencies Check")
        print("=" * 60)
        for name, info in status.items():
            if info["installed"]:
                print(f"  ✅ {name}: {info['version']}")
            else:
                print(f"  ❌ {name}: NOT INSTALLED")
                if info["error"]:
                    print(f"      Error: {info['error'][:50]}...")
        print("=" * 60 + "\n")

    return status


def export_onnx(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640),
    opset_version: int = 17,
    simplify: bool = True,
    dynamic_batch: bool = False,
    half: bool = False,
    device: Optional[str] = None,
    verbose: bool = True,
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
        verbose: Whether to print progress messages

    Returns:
        Path to the exported ONNX file
    """
    if verbose:
        _check_dependencies(verbose=True)

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # FP16 only supported on CUDA
    if half and device != "cuda":
        print("⚠️  Warning: FP16 export requires CUDA. Falling back to FP32.")
        half = False

    # Load model from checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
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

    # Determine strides automatically from model output
    print("🔍 Detecting model strides...")
    with torch.no_grad():
        outputs = model(dummy_input)
        strides = []
        for pred in outputs["Main"]:
            _, _, h, w = pred[2].shape  # vector_x shape is (B, 4, H, W)
            strides.append(image_size[1] // w)
    print(f"   Detected strides: {strides}")

    # Create export wrapper for YOLOv9-compatible output format
    export_model = YOLOExportWrapper(model, image_size, strides)
    export_model.to(device)
    export_model.eval()

    if half:
        export_model = export_model.half()

    # Define input/output names (YOLOv9 compatible)
    input_names = ["images"]
    output_names = ["output0"]

    # Dynamic axes for batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch_size", 2: "height", 3: "width"},
            "output0": {0: "batch_size", 1: "num_detections"},
        }

    # Export to ONNX
    print(f"🔄 Exporting to ONNX (opset {opset_version})...")

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print(f"✅ ONNX model saved to: {output_path}")

    # Simplify if requested
    if simplify:
        try:
            import onnx
            import onnxsim

            print("🔧 Simplifying ONNX model...")
            onnx_model = onnx.load(str(output_path))
            onnx_model_simplified, check = onnxsim.simplify(onnx_model)

            if check:
                onnx.save(onnx_model_simplified, str(output_path))
                print("✅ ONNX model simplified successfully")
            else:
                print("⚠️  Warning: ONNX simplification failed, keeping original model")

        except ImportError:
            print("⚠️  Warning: onnx-simplifier not installed. Skipping simplification.")
            print("   Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"⚠️  Warning: ONNX simplification error: {e}")
            print("   Keeping original model without simplification.")

    # Verify the model
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validation passed")

        # Print model info
        print(f"\n📊 Model Info (YOLOv9 compatible format):")
        input_shape = [d.dim_value or d.dim_param for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        output_shape = [d.dim_value or d.dim_param for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]
        print(f"   Input:  {onnx_model.graph.input[0].name} {input_shape}")
        print(f"   Output: {onnx_model.graph.output[0].name} {output_shape}")
        print(f"   Format: [batch, num_detections, 4+num_classes]")
        print(f"   Opset version: {onnx_model.opset_import[0].version}")

    except ImportError:
        print("⚠️  Warning: onnx not installed. Skipping validation.")
        print("   Install with: pip install onnx")

    return str(output_path)


def _get_calibration_images(
    image_dir: str,
    image_size: Tuple[int, int],
    num_images: int = 100,
) -> Generator[np.ndarray, None, None]:
    """
    Generate calibration images for INT8 quantization.

    Args:
        image_dir: Directory containing calibration images
        image_size: Target image size (height, width)
        num_images: Maximum number of images to use

    Yields:
        Preprocessed images as numpy arrays in NHWC format, normalized to [0, 1]
    """
    image_dir = Path(image_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Collect image paths
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    # Limit number of images
    image_paths = sorted(image_paths)[:num_images]
    print(f"📷 Using {len(image_paths)} images for INT8 calibration")

    for img_path in image_paths:
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert("RGB")

            # Resize with letterboxing to maintain aspect ratio
            img_resized = _letterbox_image(img, image_size)

            # Convert to numpy and normalize
            img_array = np.array(img_resized, dtype=np.float32) / 255.0

            # Add batch dimension: (H, W, C) -> (1, H, W, C) for NHWC format
            img_array = np.expand_dims(img_array, axis=0)

            yield img_array

        except Exception as e:
            print(f"⚠️  Warning: Failed to load {img_path}: {e}")
            continue


def _letterbox_image(
    image: Image.Image,
    target_size: Tuple[int, int],
    fill_value: int = 114,
) -> Image.Image:
    """
    Resize image with letterboxing to maintain aspect ratio.

    Args:
        image: PIL Image to resize
        target_size: Target size (height, width)
        fill_value: Fill value for padding (default: 114, gray)

    Returns:
        Letterboxed PIL Image
    """
    target_h, target_w = target_size
    orig_w, orig_h = image.size

    # Calculate scale to fit within target while maintaining aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize image
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Create new image with padding
    new_image = Image.new("RGB", (target_w, target_h), (fill_value, fill_value, fill_value))

    # Paste resized image in center
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_image.paste(image, (paste_x, paste_y))

    return new_image


def _replace_silu_with_hardswish(onnx_path: str, verbose: bool = True) -> int:
    """
    Replace SiLU activations with HardSwish in ONNX model for XNNPACK compatibility.

    SiLU (Sigmoid Linear Unit): y = x * sigmoid(x)
    HardSwish approximation: y = x * clip(x + 3, 0, 6) / 6

    HardSwish is fully supported by XNNPACK delegate, while LOGISTIC (sigmoid) is not.
    This transformation allows full GPU/XNNPACK delegation without retraining.

    Args:
        onnx_path: Path to the ONNX model (modified in-place)
        verbose: Whether to print progress messages

    Returns:
        Number of SiLU patterns replaced
    """
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        if verbose:
            print("   ⚠️  ONNX not available, skipping SiLU→HardSwish replacement")
        return 0

    if verbose:
        print("   🔄 Replacing SiLU with HardSwish for XNNPACK compatibility...")

    model = onnx.load(onnx_path)
    graph = model.graph

    # Build a map of node outputs to nodes
    output_to_node = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    # Build a map of node index for proper insertion
    node_index = {node.name: i for i, node in enumerate(graph.node)}

    # Find all Sigmoid nodes that are part of SiLU pattern
    sigmoid_nodes = {node.name: node for node in graph.node if node.op_type == 'Sigmoid'}

    if verbose:
        print(f"      Found {len(sigmoid_nodes)} Sigmoid operations")

    # Track replacements: (sigmoid_node, mul_node, x_input, output_name)
    replacements_info = []

    for node in list(graph.node):
        if node.op_type == 'Mul':
            # Check if this is x * sigmoid(x) pattern
            inputs = list(node.input)
            if len(inputs) != 2:
                continue

            sig_input = None
            x_input = None

            for inp in inputs:
                # Check if input comes from a Sigmoid
                if inp in output_to_node:
                    sig_node = output_to_node[inp]
                    if sig_node.op_type == 'Sigmoid':
                        # The other input should be the same as sigmoid's input
                        other_inp = [i for i in inputs if i != inp][0]
                        if other_inp == sig_node.input[0]:
                            sig_input = sig_node
                            x_input = other_inp
                            break

            if sig_input and x_input:
                replacements_info.append((sig_input, node, x_input, node.output[0]))

    if verbose:
        print(f"      Found {len(replacements_info)} SiLU patterns to replace")

    if not replacements_info:
        if verbose:
            print("      ℹ️  No SiLU patterns found")
        return 0

    # Create new graph nodes list
    new_nodes = []
    nodes_to_skip = set()
    initializers_to_add = []

    for sig_node, mul_node, _, _ in replacements_info:
        nodes_to_skip.add(sig_node.name)
        nodes_to_skip.add(mul_node.name)

    replacement_idx = 0
    for node in graph.node:
        if node.name in nodes_to_skip:
            # Check if this is a sigmoid node that starts a SiLU pattern
            for sig_node, mul_node, x_input, output_name in replacements_info:
                if node.name == sig_node.name:
                    # Insert HardSwish nodes here (after we see the sigmoid)
                    replacement_idx += 1
                    prefix = f"hardswish_{replacement_idx}"

                    # Constants (add as initializers)
                    const_3 = helper.make_tensor(f"{prefix}_const_3", TensorProto.FLOAT, [], [3.0])
                    const_6 = helper.make_tensor(f"{prefix}_const_6", TensorProto.FLOAT, [], [6.0])
                    const_0 = helper.make_tensor(f"{prefix}_const_0", TensorProto.FLOAT, [], [0.0])
                    initializers_to_add.extend([const_3, const_6, const_0])

                    # Create HardSwish: y = x * clip(x + 3, 0, 6) / 6

                    # x + 3
                    add_node = helper.make_node(
                        'Add',
                        inputs=[x_input, f"{prefix}_const_3"],
                        outputs=[f"{prefix}_add"],
                        name=f"{prefix}_add_node"
                    )

                    # clip(x + 3, 0, 6)
                    clip_node = helper.make_node(
                        'Clip',
                        inputs=[f"{prefix}_add", f"{prefix}_const_0", f"{prefix}_const_6"],
                        outputs=[f"{prefix}_clip"],
                        name=f"{prefix}_clip_node"
                    )

                    # x * clip(...)
                    mul_node_new = helper.make_node(
                        'Mul',
                        inputs=[x_input, f"{prefix}_clip"],
                        outputs=[f"{prefix}_mul"],
                        name=f"{prefix}_mul_node"
                    )

                    # / 6 -> output same as original SiLU output
                    div_node = helper.make_node(
                        'Div',
                        inputs=[f"{prefix}_mul", f"{prefix}_const_6"],
                        outputs=[output_name],
                        name=f"{prefix}_div_node"
                    )

                    new_nodes.extend([add_node, clip_node, mul_node_new, div_node])
                    break
        else:
            new_nodes.append(node)

    # Replace graph nodes
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Add initializers for constants
    graph.initializer.extend(initializers_to_add)

    # Validate and clean up the model
    try:
        import onnxsim
        model_simp, check = onnxsim.simplify(model)
        if check:
            model = model_simp
            if verbose:
                print("      ✅ Model simplified after HardSwish replacement")
    except Exception:
        pass  # Continue without simplification if onnxsim fails

    # Save modified model
    onnx.save(model, onnx_path)

    if verbose:
        print(f"      ✅ Replaced {len(replacements_info)} SiLU patterns with HardSwish")

    return len(replacements_info)


def _replace_dfl_conv3d_gather_with_conv2d(onnx_path: str, verbose: bool = True) -> int:
    """
    Replace DFL anc2vec Conv3D + Gather with Conv2D + Reshape for full XNNPACK delegation.

    In YOLOv9 DFL (Distribution Focal Loss) decoding, the expected value is computed via a 3D Conv:
      x: [N, C, D, H, W]  (C = reg_max+1, D = 4 coords)
      y = Conv3D(x, w:[1,C,1,1,1]) -> [N, 1, D, H, W]
      z = Gather(y, idx=0, axis=1) -> [N, D, H, W]

    TFLite conversion maps this to CONV_3D + GATHER, which XNNPACK cannot delegate.
    We rewrite it to an equivalent Conv2D by folding D into H:
      x1 = Reshape(x) -> [N, C, D*H, W]
      y1 = Conv2D(x1, w2d:[1,C,1,1]) -> [N, 1, D*H, W]
      z  = Reshape(y1) -> [N, D, H, W]

    This is mathematically identical (1x1 kernel) and removes CONV_3D/GATHER without retraining.

    Args:
        onnx_path: Path to the ONNX model (modified in-place)
        verbose: Whether to print progress messages

    Returns:
        Number of DFL patterns replaced
    """
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper, shape_inference
    except ImportError:
        if verbose:
            print("   ⚠️  ONNX/numpy not available, skipping DFL Conv3D→Conv2D replacement")
        return 0

    if verbose:
        print("   🔄 Replacing DFL Conv3D+Gather with Conv2D+Reshape for XNNPACK...")

    model = onnx.load(onnx_path)

    # Shapes are needed to compute D*H; export_onnx(simplify=True) should make them static.
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    graph = model.graph

    # Build shape map
    shape_by_name = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        t = vi.type.tensor_type
        if not t.HasField("shape"):
            continue
        dims = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                dims.append(None)
        shape_by_name[vi.name] = dims

    init_by_name = {i.name: i for i in graph.initializer}

    # Producer map: tensor_name -> node
    producer = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node

    def _const_array(name: str, max_hops: int = 8):  # noqa: ANN001
        """Resolve constant tensor value from initializer/Identity/Constant nodes."""
        cur = name
        for _ in range(max_hops):
            if cur in init_by_name:
                return numpy_helper.to_array(init_by_name[cur])
            node = producer.get(cur)
            if node is None:
                return None
            if node.op_type == "Identity" and node.input:
                cur = node.input[0]
                continue
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        tp = helper.get_attribute_value(attr)
                        return numpy_helper.to_array(tp)
                return None
            return None
        return None

    weight2d_name = "xnnpack_anc2vec_weight_2d"
    has_weight2d = weight2d_name in init_by_name

    nodes_to_remove = set()
    new_nodes = []
    replacements = 0

    for gather_node in graph.node:
        if gather_node.op_type != "Gather":
            continue
        if "/anc2vec/Gather" not in (gather_node.name or ""):
            continue

        axis = 0
        for attr in gather_node.attribute:
            if attr.name == "axis":
                axis = int(helper.get_attribute_value(attr))
                break
        if axis != 1:
            continue
        if len(gather_node.input) < 2:
            continue

        conv_out = gather_node.input[0]
        conv_node = producer.get(conv_out)
        if conv_node is None or conv_node.op_type != "Conv":
            continue
        if "/anc2vec/anc2vec/Conv" not in (conv_node.name or ""):
            continue

        x_name = conv_node.input[0]
        x_shape = shape_by_name.get(x_name)
        if not x_shape or len(x_shape) != 5:
            continue
        n, c, d, h, w = x_shape
        if any(v is None for v in (n, c, d, h, w)):
            continue

        idx_arr = _const_array(gather_node.input[1])
        if idx_arr is None or getattr(idx_arr, "shape", None) != ():
            continue
        if int(idx_arr) != 0:
            continue

        if not has_weight2d:
            w_arr = _const_array(conv_node.input[1])
            if w_arr is None or getattr(w_arr, "ndim", None) != 5:
                continue
            # (1, C, 1, 1, 1) -> (1, C, 1, 1)
            w2d_arr = np.squeeze(w_arr, axis=2).astype(np.float32, copy=False)
            graph.initializer.append(numpy_helper.from_array(w2d_arr, name=weight2d_name))
            init_by_name[weight2d_name] = graph.initializer[-1]
            has_weight2d = True

        replacements += 1
        prefix = f"xnnpack_dfl_{replacements}"

        reshape_in_shape_name = f"{prefix}_reshape_in_shape"
        reshape_out_shape_name = f"{prefix}_reshape_out_shape"
        graph.initializer.append(
            helper.make_tensor(
                reshape_in_shape_name,
                TensorProto.INT64,
                [4],
                [int(n), int(c), int(d) * int(h), int(w)],
            )
        )
        graph.initializer.append(
            helper.make_tensor(
                reshape_out_shape_name,
                TensorProto.INT64,
                [4],
                [int(n), int(d), int(h), int(w)],
            )
        )

        reshape_in_out = f"{prefix}_reshape_in"
        conv2d_out = f"{prefix}_conv2d_out"

        reshape_in_node = helper.make_node(
            "Reshape",
            inputs=[x_name, reshape_in_shape_name],
            outputs=[reshape_in_out],
            name=f"{prefix}_reshape_in_node",
        )
        conv2d_node = helper.make_node(
            "Conv",
            inputs=[reshape_in_out, weight2d_name],
            outputs=[conv2d_out],
            name=f"{prefix}_conv2d_node",
            dilations=[1, 1],
            group=1,
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        )
        reshape_out_node = helper.make_node(
            "Reshape",
            inputs=[conv2d_out, reshape_out_shape_name],
            outputs=[gather_node.output[0]],
            name=f"{prefix}_reshape_out_node",
        )

        new_nodes.extend([reshape_in_node, conv2d_node, reshape_out_node])
        nodes_to_remove.add(gather_node.name)
        nodes_to_remove.add(conv_node.name)

    if replacements == 0:
        if verbose:
            print("      ℹ️  No DFL Conv3D+Gather patterns found")
        return 0

    # Replace graph nodes (append new nodes at the end; outputs are not consumed internally).
    kept_nodes = [n for n in graph.node if n.name not in nodes_to_remove]
    del graph.node[:]
    graph.node.extend(kept_nodes)
    graph.node.extend(new_nodes)

    try:
        onnx.checker.check_model(model)
    except Exception as e:
        if verbose:
            print(f"      ⚠️  ONNX validation warning after DFL rewrite: {str(e)[:200]}")

    onnx.save(model, onnx_path)

    if verbose:
        print(f"      ✅ Replaced {replacements} DFL Conv3D+Gather patterns")

    return replacements


def _add_tflite_metadata(
    tflite_path: str,
    metadata: Dict[str, Any],
) -> None:
    """
    Add metadata to TFLite model file.

    TFLite files are ZIP archives, so we can append metadata as a JSON file.

    Args:
        tflite_path: Path to TFLite model file
        metadata: Dictionary containing metadata to embed
    """
    tflite_path = Path(tflite_path)

    # Read original TFLite content
    with open(tflite_path, "rb") as f:
        tflite_content = f.read()

    # Create new TFLite file with metadata
    with zipfile.ZipFile(str(tflite_path), "w", zipfile.ZIP_DEFLATED) as zf:
        # Write original model
        zf.writestr("model.tflite", tflite_content)
        # Write metadata
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))


def _convert_onnx_to_tflite_via_cli(
    onnx_path: str,
    output_dir: str,
    image_size: Tuple[int, int] = (640, 640),
    quantization: str = "fp32",
    calibration_path: Optional[str] = None,
) -> bool:
    """
    Convert ONNX to TFLite using onnx2tf CLI as fallback.

    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory for TFLite
        image_size: Input image size (height, width)
        quantization: Quantization mode
        calibration_path: Path to calibration data (for INT8)

    Returns:
        True if successful, False otherwise
    """
    # Configure onnx2tf CLI with optimized parameters for YOLO models
    # Full XNNPACK delegation optimization
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", output_dir,
        "-osd",  # output_signaturedefs - fix Attention block issues
        "-cotof",  # copy_onnx_input_output_names_to_tflite
        "-nuo",  # not_use_onnxsim - avoid shape inference issues
        "-b", "1",  # Set static batch size to help resolve dynamic shape issues
        "-ois", f"images:1,3,{image_size[0]},{image_size[1]}",  # Explicit input shape
        "-ofgd",  # optimization_for_gpu_delegate - improves XNNPACK/GPU compatibility
        "-dgc",  # disable_group_convolution - convert to standard conv for better compatibility
    ]

    # Add batchmatmul_unfold for non-INT8 exports (improves GPU delegate detection)
    if quantization != "int8":
        cmd.append("-ebu")  # enable_batchmatmul_unfold

    if quantization == "int8" and calibration_path:
        cmd.extend(["-oiqt", "-qt", "per-channel"])
        # IMPORTANT: Use [0,1] range since calibration images are normalized to [0,1]
        cmd.extend(["-cind", "images", calibration_path, "[[[0,0,0]]]", "[[[1,1,1]]]"])

    try:
        print(f"   Running: {' '.join(cmd[:6])}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            return True
        else:
            print(f"⚠️  CLI conversion failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"⚠️  CLI conversion error: {e}")
        return False


def export_tflite(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640),
    quantization: str = "fp32",
    calibration_images: Optional[str] = None,
    num_calibration_images: int = 100,
    device: Optional[str] = None,
    add_metadata: bool = True,
    nms: bool = False,
    xnnpack_optimize: bool = True,
) -> str:
    """
    Export a YOLO model from Lightning checkpoint to TFLite format.

    Pipeline: PyTorch -> ONNX -> TensorFlow SavedModel -> TFLite

    Args:
        checkpoint_path: Path to the .ckpt file
        output_path: Path for output .tflite file (default: same as checkpoint with .tflite extension)
        image_size: Input image size (height, width)
        quantization: Quantization mode - "fp32" (default), "fp16", or "int8"
        calibration_images: Directory with calibration images (required for int8)
        num_calibration_images: Number of calibration images to use (default: 100)
        device: Device to use for initial ONNX export (auto-detected if None)
        add_metadata: Whether to embed metadata in the TFLite file
        nms: Whether to include NMS in the model (experimental)
        xnnpack_optimize: Whether to apply XNNPACK graph rewrites (SiLU→HardSwish, DFL Conv3D→Conv2D) (default: True)

    Returns:
        Path to the exported TFLite file

    Raises:
        ImportError: If required dependencies are not installed
        ValueError: If int8 quantization requested without calibration images
    """
    # Validate quantization option
    valid_quantizations = {"fp32", "fp16", "int8"}
    if quantization not in valid_quantizations:
        raise ValueError(f"Invalid quantization: {quantization}. Must be one of {valid_quantizations}")

    # INT8 requires calibration images
    if quantization == "int8" and calibration_images is None:
        raise ValueError("INT8 quantization requires --calibration-images directory")

    # Check dependencies with visual feedback
    print("\n" + "=" * 60)
    print("🔍 Checking TFLite Export Dependencies")
    print("=" * 60)

    deps_ok = True
    onnx2tf_available = False

    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"  ✅ tensorflow: {tf.__version__}")
    except ImportError:
        print("  ❌ tensorflow: NOT INSTALLED")
        print("     Install with: pip install tensorflow")
        deps_ok = False

    # Check onnx2tf
    try:
        import onnx2tf
        print(f"  ✅ onnx2tf: available")
        onnx2tf_available = True
    except (ImportError, AttributeError) as e:
        error_msg = str(e)
        if "float32_to_bfloat16" in error_msg:
            print("  ⚠️  onnx2tf: version conflict (onnx-graphsurgeon incompatible)")
            print("     Will use CLI fallback mode")
        else:
            print(f"  ❌ onnx2tf: {error_msg[:50]}")
            print("     Install with: pip install onnx2tf")

    # Check onnx
    try:
        import onnx
        print(f"  ✅ onnx: {onnx.__version__}")
    except ImportError:
        print("  ❌ onnx: NOT INSTALLED")
        deps_ok = False

    print("=" * 60)

    if not deps_ok:
        raise ImportError(
            "\n❌ Missing required dependencies for TFLite export.\n"
            "Install with:\n"
            "  pip install tensorflow onnx2tf onnx\n"
        )

    # Determine output path
    checkpoint_path = Path(checkpoint_path)
    if output_path is None:
        output_path = checkpoint_path.with_suffix(".tflite")
    else:
        output_path = Path(output_path)

    print(f"\n📊 Export Configuration:")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Output: {output_path}")
    print(f"   Image size: {image_size}")
    print(f"   Quantization: {quantization.upper()}")
    if calibration_images:
        print(f"   Calibration images: {calibration_images}")

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Step 1: Export to ONNX first
        print("\n" + "-" * 40)
        print("📦 Step 1/3: Exporting to ONNX...")
        print("-" * 40)
        onnx_path = tmp_dir / "model.onnx"
        export_onnx(
            checkpoint_path=str(checkpoint_path),
            output_path=str(onnx_path),
            image_size=image_size,
            opset_version=13,  # Use opset 13 for better onnx2tf compatibility
            simplify=True,  # Simplify BEFORE onnx2tf to propagate static shapes
            dynamic_batch=False,
            half=False,
            device=device,
            verbose=False,
        )

        # Apply XNNPACK optimizations (SiLU → HardSwish, DFL Conv3D/Gather → Conv2D/Reshape)
        if xnnpack_optimize:
            _replace_silu_with_hardswish(str(onnx_path), verbose=True)
            _replace_dfl_conv3d_gather_with_conv2d(str(onnx_path), verbose=True)

        # Step 2: Convert ONNX to TFLite
        print("\n" + "-" * 40)
        print("🔄 Step 2/3: Converting ONNX to TFLite...")
        print("-" * 40)
        saved_model_dir = tmp_dir / "saved_model"

        # Prepare calibration data if needed
        calib_path = None
        if quantization == "int8":
            print(f"   Preparing INT8 calibration from: {calibration_images}")
            calib_data = list(_get_calibration_images(
                calibration_images,
                image_size,
                num_calibration_images,
            ))

            if len(calib_data) == 0:
                raise ValueError("No valid calibration images found")

            calib_array = np.concatenate(calib_data, axis=0)
            print(f"   Calibration data shape: {calib_array.shape}")

            calib_path = tmp_dir / "calibration_data.npy"
            np.save(str(calib_path), calib_array)

        conversion_success = False

        # Try Python API first if available
        if onnx2tf_available:
            try:
                import onnx2tf

                # Configure onnx2tf with optimized parameters for YOLO models
                # Full XNNPACK delegation optimization
                h, w = image_size
                convert_kwargs = {
                    "input_onnx_file_path": str(onnx_path),
                    "output_folder_path": str(saved_model_dir),
                    "not_use_onnxsim": True,  # Avoid shape inference issues
                    "verbosity": "info",  # Show info for debugging
                    "output_signaturedefs": True,  # Fix Attention block group convolution issues
                    "copy_onnx_input_output_names_to_tflite": True,
                    "enable_batchmatmul_unfold": quantization != "int8",  # Fix detected objects on GPU delegate
                    "optimization_for_gpu_delegate": True,  # Improves XNNPACK/GPU compatibility
                    "disable_group_convolution": True,  # Convert group conv to standard conv for better compatibility
                    # Overwrite input shape to help resolve dynamic dimension issues
                    "batch_size": 1,  # Static batch size
                    "overwrite_input_shape": [f"images:1,3,{h},{w}"],  # Explicit input shape
                }

                if quantization == "int8" and calib_path:
                    convert_kwargs["output_integer_quantized_tflite"] = True
                    convert_kwargs["quant_type"] = "per-channel"
                    # IMPORTANT: Use [0,1] range since calibration images are normalized to [0,1]
                    convert_kwargs["custom_input_op_name_np_data_path"] = [[
                        "images",
                        str(calib_path),
                        [[[[0, 0, 0]]]],  # min values
                        [[[[1, 1, 1]]]],  # max values (normalized range, NOT 255!)
                    ]]

                print("   Running onnx2tf conversion (Python API)...")
                onnx2tf.convert(**convert_kwargs)
                conversion_success = True
                print("   ✅ Conversion successful")

            except Exception as e:
                print(f"   ⚠️  Python API failed: {str(e)[:200]}")
                print("   Trying CLI fallback...")

        # Fallback to CLI
        if not conversion_success:
            print("   Running onnx2tf conversion (CLI mode)...")
            conversion_success = _convert_onnx_to_tflite_via_cli(
                str(onnx_path),
                str(saved_model_dir),
                image_size=image_size,
                quantization=quantization,
                calibration_path=str(calib_path) if calib_path else None,
            )

        if not conversion_success:
            raise RuntimeError(
                "❌ TFLite conversion failed.\n"
                "Try updating dependencies:\n"
                "  pip install --upgrade onnx2tf tensorflow onnx\n"
            )

        # Step 3: Locate and copy the TFLite file
        print("\n" + "-" * 40)
        print("📁 Step 3/3: Finalizing TFLite model...")
        print("-" * 40)

        # Find the generated TFLite file
        tflite_candidates = list(saved_model_dir.glob("*.tflite"))

        if not tflite_candidates:
            raise RuntimeError("No TFLite file generated by onnx2tf")

        # Select appropriate TFLite file based on quantization
        if quantization == "int8":
            int8_files = [f for f in tflite_candidates if "integer_quant" in f.name or "int8" in f.name.lower()]
            src_tflite = int8_files[0] if int8_files else tflite_candidates[0]
        elif quantization == "fp16":
            fp16_files = [f for f in tflite_candidates if "float16" in f.name]
            src_tflite = fp16_files[0] if fp16_files else tflite_candidates[0]
        else:
            fp32_files = [f for f in tflite_candidates if "float32" in f.name]
            src_tflite = fp32_files[0] if fp32_files else tflite_candidates[0]

        print(f"   Found TFLite: {src_tflite.name}")

        # Copy to final destination
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_tflite), str(output_path))

    # Verify the model
    print("\n" + "-" * 40)
    print("🔍 Verifying TFLite model...")
    print("-" * 40)

    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(output_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"   ✅ Model loaded successfully")
        print(f"   Input: {input_details[0]['name']}")
        print(f"      Shape: {input_details[0]['shape']}")
        print(f"      Dtype: {input_details[0]['dtype']}")
        print(f"   Output: {output_details[0]['name']}")
        print(f"      Shape: {output_details[0]['shape']}")
        print(f"      Dtype: {output_details[0]['dtype']}")

    except Exception as e:
        print(f"   ⚠️  Verification warning: {e}")

    # Print summary
    file_size = output_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print("✅ TFLite Export Complete!")
    print("=" * 60)
    print(f"   📁 Output: {output_path}")
    print(f"   📊 Size: {file_size:.2f} MB")
    print(f"   🔢 Quantization: {quantization.upper()}")
    print("=" * 60 + "\n")

    return str(output_path)


def export_qat_tflite(
    qat_checkpoint_path: str,
    output_path: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640),
    calibration_images: Optional[str] = None,
    num_calibration_images: int = 100,
    device: Optional[str] = None,
) -> str:
    """
    Export a QAT-trained model to INT8 TFLite format.

    This function handles models trained with Quantization-Aware Training (QAT),
    which should produce higher accuracy INT8 models than post-training quantization.

    Pipeline: QAT Checkpoint -> Convert to Quantized -> ONNX -> TFLite INT8

    Args:
        qat_checkpoint_path: Path to QAT-trained checkpoint (.ckpt file)
        output_path: Path for output .tflite file
        image_size: Input image size (height, width)
        calibration_images: Directory with calibration images
        num_calibration_images: Number of calibration images
        device: Device to use for export

    Returns:
        Path to the exported TFLite file
    """
    from yolo.training.qat_module import QATModule

    print("\n" + "=" * 60)
    print("🔧 QAT TFLite Export")
    print("=" * 60)
    print(f"   Checkpoint: {qat_checkpoint_path}")
    print(f"   Image size: {image_size}")
    print("=" * 60 + "\n")

    # Determine output path
    checkpoint_path = Path(qat_checkpoint_path)
    if output_path is None:
        output_path = checkpoint_path.with_suffix("").with_suffix(".qat_int8.tflite")
    else:
        output_path = Path(output_path)

    # Load QAT module
    print("📂 Loading QAT checkpoint...")
    qat_module = QATModule.load_from_checkpoint(
        qat_checkpoint_path,
        map_location="cpu" if device is None else device,
    )

    # Convert to quantized model
    print("🔄 Converting QAT model to quantized format...")
    qat_module.convert_to_quantized()

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Export quantized model to ONNX
        print("📦 Exporting quantized model to ONNX...")
        onnx_path = tmp_dir / "qat_model.onnx"

        # Create example input
        example_input = torch.randn(1, 3, image_size[0], image_size[1])

        # Export to ONNX
        qat_module.model.eval()
        torch.onnx.export(
            qat_module.model,
            example_input,
            str(onnx_path),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output"],
        )

        # Convert ONNX to TFLite
        print("🔄 Converting to TFLite INT8...")
        saved_model_dir = tmp_dir / "saved_model"

        # Prepare calibration data if provided
        calib_path = None
        if calibration_images:
            print(f"   Preparing INT8 calibration from: {calibration_images}")
            calib_data = list(_get_calibration_images(
                calibration_images,
                image_size,
                num_calibration_images,
            ))

            if len(calib_data) > 0:
                calib_array = np.concatenate(calib_data, axis=0)
                calib_path = tmp_dir / "calibration_data.npy"
                np.save(str(calib_path), calib_array)

        # Convert using onnx2tf
        conversion_success = _convert_onnx_to_tflite_via_cli(
            str(onnx_path),
            str(saved_model_dir),
            image_size=image_size,
            quantization="int8" if calib_path else "fp32",
            calibration_path=str(calib_path) if calib_path else None,
        )

        if not conversion_success:
            raise RuntimeError("TFLite conversion failed")

        # Find and copy TFLite file
        tflite_candidates = list(saved_model_dir.glob("*.tflite"))
        if not tflite_candidates:
            raise RuntimeError("No TFLite file generated")

        # Select INT8 file if available
        int8_files = [f for f in tflite_candidates if "integer_quant" in f.name or "int8" in f.name.lower()]
        src_tflite = int8_files[0] if int8_files else tflite_candidates[0]

        # Copy to final destination
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_tflite), str(output_path))

    # Verify and report
    file_size = output_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print("✅ QAT TFLite Export Complete!")
    print("=" * 60)
    print(f"   📁 Output: {output_path}")
    print(f"   📊 Size: {file_size:.2f} MB")
    print("=" * 60 + "\n")

    return str(output_path)


def export_saved_model(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640),
    device: Optional[str] = None,
) -> str:
    """
    Export a YOLO model from Lightning checkpoint to TensorFlow SavedModel format.

    Pipeline: PyTorch -> ONNX -> TensorFlow SavedModel

    Args:
        checkpoint_path: Path to the .ckpt file
        output_path: Path for output SavedModel directory
        image_size: Input image size (height, width)
        device: Device to use for initial ONNX export (auto-detected if None)

    Returns:
        Path to the exported SavedModel directory
    """
    # Check dependencies
    print("\n" + "=" * 60)
    print("🔍 Checking SavedModel Export Dependencies")
    print("=" * 60)

    try:
        import tensorflow as tf
        print(f"  ✅ tensorflow: {tf.__version__}")
    except ImportError:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    onnx2tf_available = False
    try:
        import onnx2tf
        print(f"  ✅ onnx2tf: available")
        onnx2tf_available = True
    except (ImportError, AttributeError) as e:
        print(f"  ⚠️  onnx2tf: {str(e)[:50]}")

    print("=" * 60)

    # Determine output path
    checkpoint_path = Path(checkpoint_path)
    if output_path is None:
        output_path = checkpoint_path.with_suffix("")
        output_path = Path(str(output_path) + "_saved_model")
    else:
        output_path = Path(output_path)

    # Create temporary directory for ONNX file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Step 1: Export to ONNX
        print("\n📦 Step 1/2: Exporting to ONNX...")
        onnx_path = tmp_dir / "model.onnx"
        export_onnx(
            checkpoint_path=str(checkpoint_path),
            output_path=str(onnx_path),
            image_size=image_size,
            opset_version=17,
            simplify=True,
            dynamic_batch=False,
            half=False,
            device=device,
            verbose=False,
        )

        # Step 2: Convert to SavedModel
        print("\n🔄 Step 2/2: Converting to TensorFlow SavedModel...")

        if onnx2tf_available:
            try:
                import onnx2tf
                onnx2tf.convert(
                    input_onnx_file_path=str(onnx_path),
                    output_folder_path=str(output_path),
                    not_use_onnxsim=False,
                    verbosity="info",
                    output_signaturedefs=True,
                    copy_onnx_input_output_names_to_tflite=True,
                    overwrite_input_shape=["images:1,3,{},{}".format(image_size[0], image_size[1])],
                )
            except Exception as e:
                print(f"⚠️  Python API failed, trying CLI: {e}")
                _convert_onnx_to_tflite_via_cli(str(onnx_path), str(output_path), image_size=image_size)
        else:
            _convert_onnx_to_tflite_via_cli(str(onnx_path), str(output_path), image_size=image_size)

    print(f"\n✅ SavedModel exported to: {output_path}")
    return str(output_path)
