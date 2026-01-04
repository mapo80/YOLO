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

from yolo.training.module import YOLOModule


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
    print(f"🔄 Exporting to ONNX (opset {opset_version})...")

    # Run a forward pass first to capture all tensor shapes (helps with dynamic dims)
    with torch.no_grad():
        _ = model(dummy_input)

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
        print(f"\n📊 Model Info:")
        print(f"   Input shape: {[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
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
    # Use Ultralytics-style parameters for better compatibility
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", output_dir,
        "-osd",  # output_signaturedefs - fix Attention block issues
        "-cotof",  # copy_onnx_input_output_names_to_tflite
        "-nuo",  # not_use_onnxsim - avoid shape inference issues
        "-b", "1",  # Set static batch size to help resolve dynamic shape issues
        "-ois", f"images:1,3,{image_size[0]},{image_size[1]}",  # Explicit input shape
    ]

    # Add batchmatmul_unfold for non-INT8 exports (improves GPU delegate detection)
    if quantization != "int8":
        cmd.append("-ebu")  # enable_batchmatmul_unfold

    if quantization == "int8" and calibration_path:
        cmd.extend(["-oiqt", "-qt", "per-channel"])
        cmd.extend(["-cind", "images", calibration_path, "[[[0,0,0]]]", "[[[255,255,255]]]"])

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

                # Use Ultralytics-style parameters for better compatibility
                h, w = image_size
                convert_kwargs = {
                    "input_onnx_file_path": str(onnx_path),
                    "output_folder_path": str(saved_model_dir),
                    "not_use_onnxsim": True,  # Avoid shape inference issues
                    "verbosity": "info",  # Show info for debugging
                    "output_signaturedefs": True,  # Fix Attention block group convolution issues
                    "copy_onnx_input_output_names_to_tflite": True,
                    "enable_batchmatmul_unfold": quantization != "int8",  # Fix detected objects on GPU delegate
                    # Overwrite input shape to help resolve dynamic dimension issues
                    "batch_size": 1,  # Static batch size
                    "overwrite_input_shape": [f"images:1,3,{h},{w}"],  # Explicit input shape
                }

                if quantization == "int8" and calib_path:
                    convert_kwargs["output_integer_quantized_tflite"] = True
                    convert_kwargs["quant_type"] = "per-channel"
                    convert_kwargs["custom_input_op_name_np_data_path"] = [[
                        "images",
                        str(calib_path),
                        [[[[0, 0, 0]]]],  # min values
                        [[[[255, 255, 255]]]],  # max values
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
