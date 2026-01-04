"""
Benchmark module for YOLO models.

This module provides functions for benchmarking YOLO model performance
across different formats (PyTorch, ONNX, TFLite) and configurations.

Usage:
    python -m yolo.cli benchmark --checkpoint best.ckpt
    python -m yolo.cli benchmark --model model.onnx
    python -m yolo.cli benchmark --checkpoint best.ckpt --formats pytorch,onnx --batch-sizes 1,8
"""

import gc
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

console = Console()


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the device to use for benchmarking.

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


def get_gpu_memory() -> Optional[float]:
    """
    Get current GPU memory usage in GB.

    Returns:
        GPU memory usage in GB or None if not available
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return None


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_time(ms: float) -> str:
    """Format time in appropriate units."""
    if ms < 1:
        return f"{ms * 1000:.1f} µs"
    elif ms < 1000:
        return f"{ms:.2f} ms"
    else:
        return f"{ms / 1000:.2f} s"


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(
        self,
        format_name: str,
        device: str,
        batch_size: int,
        latency_mean: float,
        latency_std: float,
        fps: float,
        memory_mb: Optional[float] = None,
        model_size_mb: Optional[float] = None,
    ):
        self.format_name = format_name
        self.device = device
        self.batch_size = batch_size
        self.latency_mean = latency_mean
        self.latency_std = latency_std
        self.fps = fps
        self.memory_mb = memory_mb
        self.model_size_mb = model_size_mb

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "latency_mean_ms": self.latency_mean,
            "latency_std_ms": self.latency_std,
            "fps": self.fps,
            "memory_mb": self.memory_mb,
            "model_size_mb": self.model_size_mb,
        }


def benchmark_pytorch(
    checkpoint_path: str,
    batch_size: int = 1,
    image_size: Tuple[int, int] = (640, 640),
    warmup: int = 10,
    runs: int = 100,
    device: Optional[str] = None,
) -> BenchmarkResult:
    """
    Benchmark PyTorch model inference.

    Args:
        checkpoint_path: Path to model checkpoint
        batch_size: Batch size for inference
        image_size: Input image size (width, height)
        warmup: Number of warmup iterations
        runs: Number of benchmark runs
        device: Device to use

    Returns:
        BenchmarkResult with timing statistics
    """
    from yolo.training.module import YOLOModule

    device = get_device(device)
    device_str = str(device)

    # Load model
    model = YOLOModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

    # Get model size
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
    memory_mb = get_gpu_memory() * 1000 if get_gpu_memory() else None

    return BenchmarkResult(
        format_name="PyTorch",
        device=device_str,
        batch_size=batch_size,
        latency_mean=mean_latency,
        latency_std=std_latency,
        fps=fps,
        memory_mb=memory_mb,
        model_size_mb=model_size_mb,
    )


def benchmark_onnx(
    model_path: str,
    batch_size: int = 1,
    image_size: Tuple[int, int] = (640, 640),
    warmup: int = 10,
    runs: int = 100,
    device: Optional[str] = None,
) -> BenchmarkResult:
    """
    Benchmark ONNX model inference using ONNX Runtime.

    Args:
        model_path: Path to ONNX model
        batch_size: Batch size for inference
        image_size: Input image size (width, height)
        warmup: Number of warmup iterations
        runs: Number of benchmark runs
        device: Device to use ('cuda' or 'cpu')

    Returns:
        BenchmarkResult with timing statistics
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required for ONNX benchmarking. Install with: pip install onnxruntime-gpu")

    # Determine providers based on device
    device = get_device(device)
    device_str = str(device)

    if device.type == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    # Create session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options, providers=providers)

    # Get model size
    model_size_mb = os.path.getsize(model_path) / 1e6

    # Get input name
    input_name = session.get_inputs()[0].name

    # Create dummy input
    dummy_input = np.random.randn(batch_size, 3, image_size[1], image_size[0]).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    # Benchmark
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)
    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    fps = (batch_size * 1000) / mean_latency

    return BenchmarkResult(
        format_name="ONNX",
        device=device_str,
        batch_size=batch_size,
        latency_mean=mean_latency,
        latency_std=std_latency,
        fps=fps,
        model_size_mb=model_size_mb,
    )


def benchmark_tflite(
    model_path: str,
    batch_size: int = 1,
    image_size: Tuple[int, int] = (640, 640),
    warmup: int = 10,
    runs: int = 100,
) -> BenchmarkResult:
    """
    Benchmark TFLite model inference.

    Args:
        model_path: Path to TFLite model
        batch_size: Batch size for inference (usually 1 for TFLite)
        image_size: Input image size (width, height)
        warmup: Number of warmup iterations
        runs: Number of benchmark runs

    Returns:
        BenchmarkResult with timing statistics
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow is required for TFLite benchmarking. Install with: pip install tensorflow")

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model size
    model_size_mb = os.path.getsize(model_path) / 1e6

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create dummy input matching expected shape and type
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    # Handle batch size
    if input_shape[0] != batch_size:
        try:
            new_shape = list(input_shape)
            new_shape[0] = batch_size
            interpreter.resize_tensor_input(input_details[0]["index"], new_shape)
            interpreter.allocate_tensors()
            input_shape = new_shape
        except Exception:
            batch_size = input_shape[0]

    dummy_input = np.random.randn(*input_shape).astype(input_dtype)

    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()

    # Benchmark
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)
    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    fps = (batch_size * 1000) / mean_latency

    return BenchmarkResult(
        format_name="TFLite",
        device="cpu",
        batch_size=batch_size,
        latency_mean=mean_latency,
        latency_std=std_latency,
        fps=fps,
        model_size_mb=model_size_mb,
    )


def benchmark(
    checkpoint_path: Optional[str] = None,
    model_path: Optional[str] = None,
    formats: List[str] = ["pytorch"],
    batch_sizes: List[int] = [1],
    image_size: Tuple[int, int] = (640, 640),
    warmup: int = 10,
    runs: int = 100,
    device: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> List[BenchmarkResult]:
    """
    Run comprehensive benchmarks on YOLO models.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.ckpt)
        model_path: Path to exported model (ONNX, TFLite)
        formats: List of formats to benchmark ('pytorch', 'onnx', 'tflite')
        batch_sizes: List of batch sizes to test
        image_size: Input image size (width, height)
        warmup: Number of warmup iterations
        runs: Number of benchmark runs
        device: Device to use
        output_path: Path to save results JSON
        verbose: Whether to print progress

    Returns:
        List of BenchmarkResult objects
    """
    results = []

    if verbose:
        console.print("\n" + "=" * 60)
        console.print("[bold blue]YOLO Benchmark[/]")
        console.print("=" * 60)
        console.print(f"[cyan]Image size:[/] {image_size[0]}x{image_size[1]}")
        console.print(f"[cyan]Warmup iterations:[/] {warmup}")
        console.print(f"[cyan]Benchmark runs:[/] {runs}")
        console.print(f"[cyan]Formats:[/] {', '.join(formats)}")
        console.print(f"[cyan]Batch sizes:[/] {', '.join(map(str, batch_sizes))}")
        console.print()

    for format_name in formats:
        for batch_size in batch_sizes:
            try:
                if verbose:
                    console.print(f"[yellow]Benchmarking {format_name} (batch={batch_size})...[/]")

                if format_name.lower() == "pytorch":
                    if checkpoint_path is None:
                        raise ValueError("checkpoint_path is required for PyTorch benchmark")
                    result = benchmark_pytorch(
                        checkpoint_path=checkpoint_path,
                        batch_size=batch_size,
                        image_size=image_size,
                        warmup=warmup,
                        runs=runs,
                        device=device,
                    )

                elif format_name.lower() == "onnx":
                    # Export to ONNX if needed
                    if model_path and model_path.endswith(".onnx"):
                        onnx_path = model_path
                    elif checkpoint_path:
                        from yolo.tools.export import export_onnx
                        with tempfile.TemporaryDirectory() as tmpdir:
                            onnx_path = export_onnx(
                                checkpoint_path=checkpoint_path,
                                output_path=str(Path(tmpdir) / "model.onnx"),
                                image_size=image_size,
                                verbose=False,
                            )
                            result = benchmark_onnx(
                                model_path=onnx_path,
                                batch_size=batch_size,
                                image_size=image_size,
                                warmup=warmup,
                                runs=runs,
                                device=device,
                            )
                            results.append(result)
                            continue
                    else:
                        raise ValueError("checkpoint_path or model_path is required for ONNX benchmark")

                    result = benchmark_onnx(
                        model_path=onnx_path,
                        batch_size=batch_size,
                        image_size=image_size,
                        warmup=warmup,
                        runs=runs,
                        device=device,
                    )

                elif format_name.lower() == "tflite":
                    if model_path and model_path.endswith(".tflite"):
                        tflite_path = model_path
                    elif checkpoint_path:
                        from yolo.tools.export import export_tflite
                        with tempfile.TemporaryDirectory() as tmpdir:
                            tflite_path = export_tflite(
                                checkpoint_path=checkpoint_path,
                                output_path=str(Path(tmpdir) / "model.tflite"),
                                image_size=image_size,
                                verbose=False,
                            )
                            result = benchmark_tflite(
                                model_path=tflite_path,
                                batch_size=batch_size,
                                image_size=image_size,
                                warmup=warmup,
                                runs=runs,
                            )
                            results.append(result)
                            continue
                    else:
                        raise ValueError("checkpoint_path or model_path is required for TFLite benchmark")

                    result = benchmark_tflite(
                        model_path=tflite_path,
                        batch_size=batch_size,
                        image_size=image_size,
                        warmup=warmup,
                        runs=runs,
                    )

                else:
                    if verbose:
                        console.print(f"[red]Unknown format: {format_name}[/]")
                    continue

                results.append(result)

            except Exception as e:
                if verbose:
                    console.print(f"[red]Error benchmarking {format_name}: {e}[/]")

            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Print results table
    if verbose and results:
        print_benchmark_results(results)

    # Save results
    if output_path and results:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({
                "config": {
                    "image_size": list(image_size),
                    "warmup": warmup,
                    "runs": runs,
                },
                "results": [r.to_dict() for r in results],
            }, f, indent=2)
        if verbose:
            console.print(f"\n[green]Results saved to:[/] {output_file}")

    return results


def print_benchmark_results(results: List[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    table = Table(title="Benchmark Results", show_header=True, header_style="bold cyan")

    table.add_column("Format", style="cyan")
    table.add_column("Device", style="green")
    table.add_column("Batch", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("FPS", justify="right", style="yellow")
    table.add_column("Memory", justify="right")
    table.add_column("Model Size", justify="right")

    for r in results:
        memory_str = f"{r.memory_mb:.0f} MB" if r.memory_mb else "-"
        size_str = f"{r.model_size_mb:.1f} MB" if r.model_size_mb else "-"

        table.add_row(
            r.format_name,
            r.device,
            str(r.batch_size),
            f"{r.latency_mean:.2f} ± {r.latency_std:.2f}",
            f"{r.fps:.1f}",
            memory_str,
            size_str,
        )

    console.print()
    console.print(table)
    console.print()
