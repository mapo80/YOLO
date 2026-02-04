# YOLOv9-MIT: Training Framework

![GitHub License](https://img.shields.io/github/license/WongKinYiu/YOLO)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-792ee5.svg)](https://lightning.ai/)
[![Tests](https://img.shields.io/badge/tests-405%20passed-brightgreen.svg)](tests/)

> A complete training framework for YOLOv9 built on PyTorch Lightning,
> with comprehensive metrics and export to ONNX/TFLite.

**Key Features:**
- **Task-Aligned Learning** - Soft target assignment aligned with yolov9-official
- **Full COCO Metrics** - mAP, AR, size-based metrics + EvalDashboard
- **Export** - ONNX, TFLite (FP32/FP16/INT8), Quantization-Aware Training
- **Data Pipeline** - Encrypted datasets, LMDB caching, BBox Mosaic augmentation

> **Fork Notice**: This is a fork of [WongKinYiu/YOLO](https://github.com/WongKinYiu/YOLO) with extensive additions for production training.

This repository extends the official YOLOv7[^1], YOLOv9[^2], and YOLO-RD[^3] implementation with a **robust CLI for training, validation, and export**.

---

## Table of Contents

- [What's Different From the Original?](#whats-different-from-the-original)
- [Features](#features)
- [Papers](#papers)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Training](#training)
  - [Pretrained Weights](#pretrained-weights)
  - [Validation](#validation)
  - [Inference](#inference)
  - [Export](#export)
  - [Quantization-Aware Training (QAT)](#quantization-aware-training-qat)
- [Image Loaders](#image-loaders)
- [Data Loading Performance](#data-loading-performance)
- [Advanced Training Techniques](#advanced-training-techniques)
  - [Data Augmentation](#data-augmentation)
  - [Dataset Caching](#dataset-caching)
    - [Cache Management CLI](#cache-management-cli)
    - [Use Cases and Scenarios](#use-cases-and-scenarios)
    - [Encrypted Images vs Encrypted Cache](#encrypted-images-vs-encrypted-cache)
    - [Common Errors and Troubleshooting](#common-errors-and-troubleshooting)
  - [Data Fraction](#data-fraction-quick-testing)
  - [EMA](#exponential-moving-average-ema)
  - [Optimizer Selection](#optimizer-selection)
- [Configuration](#configuration)
  - [Dataset Formats](#dataset-formats)
- [Testing](#testing)
- [Metrics](#metrics)
- [Learning Rate Schedulers](#learning-rate-schedulers)
- [Layer Freezing](#layer-freezing-transfer-learning)
- [Checkpoints](#checkpoints)
- [Early Stopping](#early-stopping)
- [Performance](#performance)
- [Task-Aligned Learning (TAL)](#task-aligned-learning-tal)
- [Loss Functions](#loss-functions)
- [Custom Callbacks](#custom-callbacks)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Citations](#citations)
- [License](#license)

---

## What's Different From the Original?

The original repository focuses on model architecture and research contributions. This fork adds:

- **Complete training CLI** with YAML configuration and command-line overrides
- **Data augmentation suite** - Mosaic 4/9, BBox Mosaic (document-optimized), MixUp, CutMix, RandomPerspective, HSV
- **Multiple dataset formats** - COCO JSON and YOLO TXT with caching
- **Comprehensive metrics** - Full COCO evaluation, confusion matrix, PR curves
- **Rich monitoring** - Progress bars, eval dashboard, TensorBoard integration
- **Model export** - ONNX, TFLite (FP32/FP16/INT8), SavedModel

All additions are built on **PyTorch Lightning** for clean, scalable training.

### Comparison with yolov9-official

| Feature | yolov9-official | **yolo-mit** |
|---------|-----------------|--------------|
| Training CLI | ❌ Manual scripts | ✅ YAML config + CLI overrides |
| PyTorch Lightning | ❌ | ✅ Multi-GPU, checkpoints, callbacks |
| EMA | ❌ | ✅ Configurable with tau warmup |
| COCO Metrics | Basic print | ✅ Full 12 metrics + EvalDashboard |
| TFLite Export | ❌ | ✅ FP32/FP16/INT8 + QAT |
| BBox Mosaic | ❌ | ✅ Document-optimized augmentation |
| Encrypted Datasets | ❌ | ✅ AES-256 image encryption |
| Image Caching | ❌ | ✅ LMDB RAM/disk with encryption |
| Early Stopping | ❌ | ✅ Configurable patience |
| LR Schedulers | Step only | ✅ Cosine, OneCycle, Linear, Step |

## Features

### Training & Optimization
| Feature | Description |
|---------|-------------|
| **Multi-GPU Training** | Scale training across multiple GPUs with automatic Distributed Data Parallel (DDP). No code changes needed - just set `--trainer.devices=N` |
| **Mixed Precision** | Train 2x faster with half the memory using FP16/BF16. Automatic loss scaling prevents underflow |
| **Optimizers** | SGD with Nesterov momentum (recommended for detection) or AdamW for faster convergence on small datasets |
| **LR Schedulers** | Cosine annealing for smooth decay, OneCycle for super-convergence, Step for classic training, Linear for simplicity |
| **Warmup** | Gradually ramp learning rate and momentum during initial epochs to stabilize training and prevent early divergence |
| **EMA** | Maintain a smoothed copy of model weights that often achieves better accuracy than final training weights |
| **Layer Freezing** | Freeze backbone layers when fine-tuning on small datasets. Train only the detection head, then unfreeze for full fine-tuning |
| **Early Stopping** | Automatically stop training when validation mAP stops improving, saving compute and preventing overfitting |
| **Checkpointing** | Save best model by mAP, keep last checkpoint for resume, and maintain top-K checkpoints with metrics in filenames |

### Data Augmentation
| Feature | Description |
|---------|-------------|
| **Mosaic 4/9** | Combine 4 or 9 images into one training sample. Improves small object detection and increases effective batch diversity |
| **BBox Mosaic** | Document-optimized mosaic: crops to bounding box, applies per-document transforms, places in 2x2 grid without clipping. Ideal for ID documents, logos, products |
| **MixUp** | Blend two mosaic images with random weights. Regularizes the model and improves generalization |
| **CutMix** | Cut a patch from one image and paste onto another. Combines benefits of cutout and mixup augmentation |
| **RandomPerspective** | Apply geometric transforms: rotation, translation, scaling, shear, and perspective distortion with box adjustment |
| **RandomHSV** | Randomly shift hue, saturation, and brightness. Makes model robust to lighting and color variations |
| **RandomFlip** | Horizontal and vertical flips with automatic bounding box coordinate updates |
| **Close Mosaic** | Disable heavy augmentations for final N epochs. Lets model fine-tune on clean images for better convergence |

### Data Loading & Caching
| Feature | Description |
|---------|-------------|
| **Dataset Formats** | Native support for COCO JSON annotations and YOLO TXT format. Switch with `--data.format=yolo` |
| **Label Caching** | Parse label files once and cache to disk. Subsequent runs load instantly with automatic invalidation on file changes |
| **Image Caching** | Memory-mapped RAM cache (`ram`) shared between workers, or disk cache (`disk`) with optional encryption |
| **Cache Resize** | Resize images to `image_size` during caching for reduced RAM usage (default: enabled) |
| **Encrypted Cache** | Encrypt disk cache files (`.npy.enc`) for secure storage of sensitive datasets |
| **Data Fraction** | Stratified sampling to use a fraction of data for quick testing (e.g., `data_fraction: 0.1` for 10%) |
| **Custom Loaders** | Plug in custom image loaders for encrypted datasets, cloud storage, or proprietary formats |
| **Pin Memory** | Pre-load batches to pinned (page-locked) memory for faster CPU-to-GPU transfer |

### Inference & NMS
| Feature | Description |
|---------|-------------|
| **Configurable NMS** | Tune confidence threshold, IoU threshold, and max detections per image to balance precision vs recall |
| **Separate Val Threshold** | Use lower confidence threshold during validation (`nms_val_conf_threshold: 0.001`) to capture all predictions for accurate mAP calculation, while keeping higher threshold for inference |
| **Batch Inference** | Process entire directories of images with automatic output organization and optional JSON export |
| **Class Names** | Automatically load class names from dataset (data.yaml or COCO JSON) for human-readable predictions |

### Metrics & Evaluation
| Feature | Description |
|---------|-------------|
| **COCO Metrics** | Full COCO evaluation: mAP@0.5, mAP@0.5:0.95, AP75, AR@100, and size-specific APs/APm/APl |
| **Precision/Recall/F1** | Track detection quality at your production confidence threshold. Per-class and aggregate scores |
| **Confusion Matrix** | Visualize which classes get confused with each other. Essential for debugging detection errors |
| **PR/F1 Curves** | Auto-generated plots showing precision-recall tradeoffs and optimal confidence thresholds |
| **Eval Dashboard** | Rich terminal dashboard with sparkline trends, top/worst classes, error analysis, and threshold sweep |
| **Benchmark** | Measure inference latency (mean, p95, p99), memory usage, and throughput with `--benchmark` |

### Model Export
| Feature | Description |
|---------|-------------|
| **ONNX** | Export for cross-platform deployment. Supports simplification, dynamic batching, and FP16 precision |
| **TFLite** | Deploy on mobile/edge devices with FP32, FP16, or INT8 quantization. INT8 requires calibration images |
| **SavedModel** | TensorFlow SavedModel format for TF Serving, TensorFlow.js, and cloud deployment |

### Logging & Monitoring
| Feature | Description |
|---------|-------------|
| **TensorBoard** | Visualize training curves, validation metrics, learning rate, and hyperparameters in real-time |
| **WandB** | Weights & Biases integration for experiment tracking, team collaboration, and hyperparameter sweeps |
| **Progress Bar** | Custom progress bar showing epoch, loss components (box/cls/dfl), learning rate, and ETA |
| **Training Summary** | End-of-epoch summary with current vs best mAP, improvement indicators, and checkpoint status |

## Papers

- [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- [**YOLOv7**: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)
- [**YOLO-RD**: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary](https://arxiv.org/abs/2410.15346)

## Installation

```shell
git clone https://github.com/mapo80/YOLO.git
cd YOLO
pip install -r requirements.txt

# (Recommended) install as a package to enable the `yolo` CLI entrypoint
pip install -e .  # add --no-deps if you already ran pip install -r requirements.txt
```

## Quick Start

All CLI commands can be run either with `python -m yolo.cli ...` (from this repo) or with the installed `yolo ...` entrypoint.

### Training

```shell
# Train with default config (from scratch)
python -m yolo.cli fit --config yolo/config/experiment/default.yaml

# Train with pretrained weights (transfer learning)
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.weight_path=weights/v9-c.pt

# Custom parameters
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.model_config=v9-c \
    --data.batch_size=16 \
    --trainer.max_epochs=300

# Debug run (small dataset, few epochs)
python -m yolo.cli fit --config yolo/config/experiment/debug.yaml
```

📖 **For complete documentation**, see [HOWTO.md](HOWTO.md) - covers configuration, custom models, dataset formats, metrics, LR schedulers, layer freezing, and model export.

### Training Example

For a complete end-to-end training example, see the [Simpsons Character Detection Training Guide](training-experiment/TRAINING_GUIDE.md).

This example demonstrates:
- Dataset preparation (COCO format conversion)
- Configuration file setup
- Training with pretrained weights (transfer learning)
- Checkpoint management and resume training
- Inference on custom images

**Quick start with the example:**

```shell
# Train on Simpsons dataset with pretrained weights
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml

# Resume training from checkpoint
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml \
    --ckpt_path=training-experiment/checkpoints/last.ckpt

# Run inference on trained model
python -m yolo.cli predict \
    --checkpoint training-experiment/checkpoints/best.ckpt \
    --source path/to/image.jpg
```

### Pretrained Weights

Use official YOLOv9 pretrained weights (trained on COCO, 80 classes) for transfer learning on custom datasets.

#### Usage

```shell
# Auto-download weights based on model_config (recommended)
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.weight_path=true

# Or specify a path explicitly
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.weight_path=weights/v9-c.pt

# Train from scratch (no pretrained weights)
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.weight_path=null
```

#### YAML Configuration

```yaml
model:
  model_config: v9-c      # Model architecture
  num_classes: 7          # Your custom classes

  # Pretrained weights options:
  weight_path: null       # Train from scratch
  weight_path: true       # Auto-download v9-c.pt (based on model_config)
  weight_path: "weights/custom.pt"  # Use specific file
```

#### Available Pretrained Weights

| Model | Config Name | Weights |
|:------|:------------|:-------:|
| YOLOv9-T | `v9-t` | [v9-t.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt) |
| YOLOv9-S | `v9-s` | [v9-s.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s-converted.pt) |
| YOLOv9-M | `v9-m` | [v9-m.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m-converted.pt) |
| YOLOv9-C | `v9-c` | [v9-c.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt) |
| YOLOv9-E | `v9-e` | [v9-e.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt) |

#### Transfer Learning Behavior

When loading pretrained weights with a different number of classes, the system automatically:

1. **Loads compatible layers** - Backbone, neck, and box regression layers are loaded from COCO weights
2. **Skips incompatible layers** - Classification head layers (`class_conv`) are initialized randomly
3. **Logs warnings** - Shows which layers couldn't be loaded due to shape mismatch

Example output when training 7-class model with COCO (80-class) weights:
```
INFO     Building model: v9-t (7 classes)
WARNING  Weight Mismatch for Layer 22: heads.0.class_conv, heads.1.class_conv, heads.2.class_conv
WARNING  Weight Mismatch for Layer 30: heads.0.class_conv, heads.1.class_conv, heads.2.class_conv
INFO     Loaded pretrained weights: weights/v9-t.pt
```

This is **expected behavior** - the classification layers will be trained for your custom classes while the feature extraction layers benefit from COCO pretraining.

### Validation

Standalone validation to evaluate a trained model on a dataset:

```shell
# Validate with config file
python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml

# Validate with direct parameters (YOLO format)
python -m yolo.cli validate --checkpoint best.ckpt \
    --data.root dataset/ --data.format yolo \
    --data.val_images valid/images --data.val_labels valid/labels

# Validate with direct parameters (COCO format)
python -m yolo.cli validate --checkpoint best.ckpt \
    --data.root dataset/ --data.format coco \
    --data.val_images val2017 --data.val_ann annotations/instances_val.json

# Save plots and JSON results
python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml \
    --output results/ --save-plots --save-json

# Validate on specific split (e.g., test set)
python -m yolo.cli validate --checkpoint best.ckpt \
    --data.root dataset/ --data.format yolo \
    --data.val_images images --data.val_labels labels \
    --data.val_split test.txt

# Or use split file via config
python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml
# With config containing: data.val_split: test.txt
```

#### Validation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint, -c` | required | Path to model checkpoint (.ckpt) |
| `--config` | - | Path to config YAML file |
| `--data.root` | - | Root directory of the dataset |
| `--data.format` | coco | Dataset format (coco or yolo) |
| `--data.val_images` | - | Path to validation images |
| `--data.val_labels` | - | Path to validation labels (YOLO format) |
| `--data.train_split` | - | Path to train split file (YOLO format) |
| `--data.val_split` | - | Path to split file (filters images by filename list) |
| `--data.val_ann` | - | Path to validation annotations (COCO format) |
| `--batch-size` | 16 | Batch size for validation |
| `--conf` | 0.001 | Confidence threshold |
| `--iou` | 0.6 | IoU threshold for NMS |
| `--size` | 640 | Input image size |
| `--device` | auto | Device (cuda/mps/cpu) |
| `--output, -o` | validation_results | Output directory |
| `--save-plots` | true | Save metric plots (PR curve, confusion matrix) |
| `--no-plots` | - | Disable metric plots |
| `--save-json` | true | Save results as JSON |
| `--no-json` | - | Disable JSON output |
| `--workers` | 4 | Number of data loading workers |

**Output**: Eval dashboard with comprehensive metrics, per-class analysis, confusion matrix, PR/F1 curves.

#### Validation with Benchmark

Run validation with integrated benchmark to measure inference performance:

```shell
# Validate + benchmark (latency, memory, model size)
python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml --benchmark

# Custom benchmark settings
python -m yolo.cli validate --checkpoint best.ckpt --config config.yaml \
    --benchmark --benchmark-warmup 20 --benchmark-runs 200
```

#### Extended Validation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--conf-prod` | 0.25 | Production confidence for operative metrics |
| `--benchmark` | false | Run latency/memory benchmark |
| `--benchmark-warmup` | 10 | Warmup iterations for benchmark |
| `--benchmark-runs` | 100 | Number of benchmark runs |

### Inference

```shell
# Single image
python -m yolo.cli predict --checkpoint runs/best.ckpt --source image.jpg

# Directory of images
python -m yolo.cli predict --checkpoint runs/best.ckpt --source images/ --output results/

# Custom thresholds
python -m yolo.cli predict --checkpoint runs/best.ckpt --source image.jpg --conf 0.5 --iou 0.5

# Without drawing boxes (JSON output only)
python -m yolo.cli predict --checkpoint runs/best.ckpt --source image.jpg --no-draw --save-json
```

#### Inference Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint, -c` | required | Path to model checkpoint (.ckpt) |
| `--source, -s` | required | Image file or directory |
| `--output, -o` | auto | Output path for results |
| `--conf` | 0.25 | Confidence threshold |
| `--iou` | 0.65 | IoU threshold for NMS |
| `--max-det` | 300 | Maximum detections per image |
| `--draw` | true | Draw bounding boxes on images |
| `--no-draw` | - | Disable bounding box drawing |
| `--save-json` | false | Save predictions to JSON |
| `--class-names` | - | Path to JSON file with class names |
| `--device` | auto | Device (cuda/mps/cpu) |
| `--size` | 640 | Input image size |

### Export

Export trained models to ONNX, TFLite, or TensorFlow SavedModel format for deployment on various platforms including mobile devices, edge TPUs, and cloud services.

#### Step-by-Step Export Guide (Recommended)

This guide shows the complete workflow to export YOLOv9 models with validated accuracy.

##### 1. Export to ONNX

```shell
# Export with simplification (required for TFLite conversion)
python -m yolo.cli export --checkpoint runs/best.ckpt --format onnx --simplify --opset 17

# Output: runs/best.onnx (~11 MB for YOLOv9-T)
```

##### 2. Export to TFLite (via Docker)

**Build Docker image (one-time):**
```shell
docker build --platform linux/amd64 -t yolo-tflite-export -f docker/Dockerfile.tflite-export .
```

**Export FP32 (full precision):**
```shell
docker run --platform linux/amd64 -v $(pwd):/workspace yolo-tflite-export \
    --checkpoint /workspace/runs/best.ckpt \
    --format tflite \
    --output /workspace/exports/model_fp32.tflite
```

**Export FP16 (recommended for mobile GPU):**
```shell
docker run --platform linux/amd64 -v $(pwd):/workspace yolo-tflite-export \
    --checkpoint /workspace/runs/best.ckpt \
    --format tflite \
    --quantization fp16 \
    --output /workspace/exports/model_fp16.tflite
```

**Export INT8 (smallest size, requires calibration):**
```shell
# Prepare calibration images (100-200 representative images)
mkdir -p calibration_data
# Copy images from your training/validation set

# Export with INT8 quantization
docker run --platform linux/amd64 -v $(pwd):/workspace yolo-tflite-export \
    --checkpoint /workspace/runs/best.ckpt \
    --format tflite \
    --quantization int8 \
    --calibration-images /workspace/calibration_data/ \
    --num-calibration 100 \
    --output /workspace/exports/model_int8.tflite
```

##### 3. Validate Exported Models

**Validate ONNX:**
```shell
python -m yolo.cli validate --checkpoint exports/model.onnx \
    --data.root path/to/dataset \
    --data.format coco
```

**Validate TFLite (using provided script):**

The `validate_tflite.py` script computes mAP metrics for TFLite models (FP32, FP16, INT8). It supports both **COCO** and **YOLO** format datasets and automatically detects the model's input size.

**COCO format examples:**

```shell
# Standard COCO dataset structure
# Expects: coco_root/annotations/instances_val2017.json and coco_root/val2017/
python scripts/validate_tflite.py \
    --model model.tflite \
    --coco-root /path/to/coco

# Custom COCO paths (specify annotation file and images separately)
python scripts/validate_tflite.py \
    --model model.tflite \
    --ann-file /path/to/annotations.json \
    --images-dir /path/to/images \
    --num-classes 10
```

**YOLO format examples:**

```shell
# YOLO format dataset
# Expects: images/ directory + labels/ directory with .txt files
python scripts/validate_tflite.py \
    --model model.tflite \
    --images-dir /path/to/images \
    --labels-dir /path/to/labels \
    --data-format yolo \
    --num-classes 5

# Validate subset of images with custom thresholds
python scripts/validate_tflite.py \
    --model model.tflite \
    --images-dir /path/to/images \
    --labels-dir /path/to/labels \
    --data-format yolo \
    --num-classes 5 \
    --num-images 100 \
    --conf-threshold 0.25
```

**Available options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | required | Path to TFLite model |
| `--coco-root` | - | Path to COCO dataset root (expects `annotations/instances_val2017.json` and `val2017/`) |
| `--ann-file` | - | Path to COCO-format annotations JSON (overrides `--coco-root`) |
| `--images-dir` | - | Path to images directory |
| `--labels-dir` | - | Path to YOLO labels directory (enables YOLO format) |
| `--data-format` | auto | Dataset format: `auto`, `coco`, or `yolo` |
| `--num-images` | 500 | Number of images to validate |
| `--num-classes` | 80 | Number of classes in the model (use for custom models) |
| `--conf-threshold` | 0.001 | Confidence threshold for detections |
| `--iou-threshold` | 0.65 | IoU threshold for NMS |

**Dataset formats:**

| Format | Structure | Annotation format |
|--------|-----------|-------------------|
| **COCO** | `annotations/*.json` + `images/` | JSON with `[x, y, width, height]` in pixels |
| **YOLO** | `images/` + `labels/` | `.txt` files with `class x_center y_center width height` (normalized 0-1) |

**Notes:**
- The script auto-detects the model's input size from the TFLite file
- For custom models with `num_classes != 80`, you **must** specify `--num-classes`
- Format is auto-detected: `--labels-dir` triggers YOLO, `--ann-file` or `--coco-root` triggers COCO
- Results are saved to `<model_name>_metrics.json` in the model's directory

##### Expected Output Files

After completing the export workflow:

```
exports/
├── model.onnx         # 11 MB - ONNX format
├── model_fp32.tflite  # 11 MB - TFLite full precision
├── model_fp16.tflite  # 5.8 MB - TFLite half precision
└── model_int8.tflite  # 3.3 MB - TFLite INT8 quantized
```

#### Supported Export Formats

| Format | Extension | Use Case | Dependencies |
|--------|-----------|----------|--------------|
| **ONNX** | `.onnx` | Cross-platform inference, TensorRT, OpenVINO | `onnx`, `onnxsim` |
| **TFLite** | `.tflite` | Mobile (Android/iOS), Edge TPU, Coral | `tensorflow`, `onnx2tf` |
| **SavedModel** | directory | TensorFlow Serving, TF.js | `tensorflow`, `onnx2tf` |

#### Export to ONNX

ONNX export is the fastest and simplest option, with no additional dependencies beyond the base installation.

```shell
# Basic ONNX export
python -m yolo.cli export --checkpoint runs/best.ckpt --format onnx

# Export with custom output path and size
python -m yolo.cli export --checkpoint runs/best.ckpt --output model.onnx --size 640

# Export with FP16 precision (CUDA only, smaller model)
python -m yolo.cli export --checkpoint runs/best.ckpt --half

# Export with dynamic batch size (for variable batch inference)
python -m yolo.cli export --checkpoint runs/best.ckpt --dynamic-batch

# Export with ONNX simplification (recommended for deployment)
python -m yolo.cli export --checkpoint runs/best.ckpt --simplify

# Full example with all optimizations
python -m yolo.cli export --checkpoint runs/best.ckpt \
    --format onnx \
    --output model_optimized.onnx \
    --simplify \
    --opset 17
```

##### Export and Verification Workflow

After exporting, verify that the ONNX model produces the same results as the original checkpoint:

```bash
# Step 1: Export to ONNX
python -m yolo export --checkpoint runs/best.ckpt --output model.onnx --size 320

# Step 2: Run inference with checkpoint (reference)
python -m yolo predict --checkpoint runs/best.ckpt --source test.jpg --output result_ckpt.jpg --size 320

# Step 3: Run inference with ONNX (verification)
python tools/onnx_inference.py --model model.onnx --image test.jpg --output result_onnx.jpg

# Step 4: Compare results
# Both should show the same classes, similar confidences, and matching bounding boxes
```

**Expected behavior:**
- Same classes detected
- Confidence scores within ~2% (minor differences due to floating point precision)
- Bounding box coordinates should match (after accounting for coordinate scaling)

##### ONNX Model Format Specification

The exported ONNX model uses the [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) output format, ensuring compatibility with existing inference pipelines (including .NET, C++, and mobile).

###### Input Specification

| Property | Value | Description |
|----------|-------|-------------|
| **Name** | `images` | Input tensor name |
| **Shape** | `[batch, 3, height, width]` | NCHW format |
| **Type** | `float32` | Normalized to [0, 1] |
| **Color space** | RGB | Not BGR |

Example for 320x320 model: `images=[1, 3, 320, 320]`

###### Output Specification

| Property | Value | Description |
|----------|-------|-------------|
| **Name** | `output0` | Output tensor name |
| **Shape** | `[batch, 4+num_classes, num_anchors]` | Transposed format |
| **Type** | `float32` | Box coords + probabilities |

Example for 320x320 with 13 classes: `output0=[1, 17, 2100]`
- `17` = 4 (box) + 13 (class scores)
- `2100` = 40×40 + 20×20 + 10×10 anchors (for strides 8, 16, 32)

Example for 640x640 with 80 classes: `output0=[1, 84, 8400]`
- `84` = 4 (box) + 80 (class scores)
- `8400` = 80×80 + 40×40 + 20×20 anchors

###### Output Tensor Layout

The output tensor has shape `[batch, 4+num_classes, num_anchors]`:

```
output[0, 0, :]  → center_x (cx) for all anchors
output[0, 1, :]  → center_y (cy) for all anchors
output[0, 2, :]  → width (w) for all anchors
output[0, 3, :]  → height (h) for all anchors
output[0, 4:, :] → class scores (post-sigmoid, 0-1) for all anchors
```

| Channel | Content | Range | Units |
|---------|---------|-------|-------|
| `[0]` | center_x | [0, input_width] | pixels |
| `[1]` | center_y | [0, input_height] | pixels |
| `[2]` | width | [0, input_width] | pixels |
| `[3]` | height | [0, input_height] | pixels |
| `[4:]` | class scores | [0, 1] | probability (post-sigmoid) |

###### Box Format: XYWH (Center)

Bounding boxes are in **XYWH center format** with absolute pixel coordinates:

```
┌─────────────────────────────┐
│                             │
│      ┌───────────┐          │
│      │           │          │
│      │    (cx,cy)●          │  cx = center x
│      │           │          │  cy = center y
│      │     w     │          │  w  = width
│      └─────┬─────┘          │  h  = height
│            h                │
│                             │
└─────────────────────────────┘
```

To convert to XYXY (top-left, bottom-right):
```python
x1 = cx - w / 2
y1 = cy - h / 2
x2 = cx + w / 2
y2 = cy + h / 2
```

##### Preprocessing Requirements

**IMPORTANT: The model requires letterbox preprocessing to maintain aspect ratio.**

The model was trained with letterbox resizing (padding to maintain aspect ratio). Using simple resize (stretching) will produce incorrect bounding boxes.

###### Letterbox Preprocessing Steps

1. **Calculate scale** to fit image in target size while maintaining aspect ratio
2. **Resize** image using the calculated scale
3. **Pad** with gray (114, 114, 114) to reach target size
4. **Convert** BGR to RGB
5. **Normalize** to [0, 1] by dividing by 255
6. **Transpose** from HWC to CHW format

###### Python Letterbox Implementation

```python
import numpy as np
import cv2

def letterbox(image, target_size=(640, 640), color=(114, 114, 114)):
    """
    Resize image with letterbox (maintain aspect ratio with padding).

    Args:
        image: BGR image as numpy array (H, W, C)
        target_size: (width, height) tuple
        color: padding color (B, G, R)

    Returns:
        letterboxed: padded image (target_height, target_width, 3)
        scale: scale factor used
        pad: (pad_x, pad_y) padding added
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale (fit in target while maintaining aspect ratio)
    scale = min(target_w / w, target_h / h)

    # New size after scaling
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    # Create padded image
    letterboxed = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    letterboxed[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

    return letterboxed, scale, (pad_w, pad_h)


def preprocess(image_path, target_size=(640, 640)):
    """
    Full preprocessing pipeline for ONNX inference.

    Returns:
        tensor: (1, 3, H, W) float32 tensor normalized to [0, 1]
        scale: scale factor for coordinate conversion
        pad: (pad_x, pad_y) for coordinate conversion
    """
    # Load image (OpenCV loads as BGR)
    image = cv2.imread(image_path)

    # Letterbox resize
    letterboxed, scale, pad = letterbox(image, target_size)

    # BGR to RGB
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0

    # HWC to CHW
    chw = np.transpose(normalized, (2, 0, 1))

    # Add batch dimension
    tensor = np.expand_dims(chw, axis=0)

    return tensor, scale, pad
```

##### Postprocessing

###### Decode Predictions

```python
import numpy as np

def decode_predictions(output, conf_threshold=0.25):
    """
    Decode ONNX output to detections.

    Args:
        output: ONNX output tensor [1, 4+num_classes, num_anchors]
        conf_threshold: confidence threshold

    Returns:
        boxes: [N, 4] array of XYXY boxes
        scores: [N] array of confidence scores
        class_ids: [N] array of class indices
    """
    # Transpose to [num_anchors, 4+num_classes]
    predictions = output[0].T

    # Split box coordinates and class scores
    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    # Get best class for each anchor
    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    # Filter by confidence
    mask = scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert XYWH to XYXY
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

    return boxes_xyxy, scores, class_ids
```

###### Scale Boxes to Original Image

```python
def scale_boxes(boxes, scale, pad, original_size):
    """
    Scale boxes from letterboxed coordinates to original image coordinates.

    Args:
        boxes: [N, 4] XYXY boxes in letterboxed coordinates
        scale: scale factor used in letterbox
        pad: (pad_x, pad_y) padding used in letterbox
        original_size: (width, height) of original image

    Returns:
        boxes: [N, 4] XYXY boxes in original image coordinates
    """
    pad_x, pad_y = pad

    # Remove padding offset
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y

    # Scale back to original size
    boxes /= scale

    # Clip to image bounds
    orig_w, orig_h = original_size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    return boxes
```

###### Non-Maximum Suppression (NMS)

```python
def nms(boxes, scores, iou_threshold=0.45):
    """
    Apply Non-Maximum Suppression.

    Args:
        boxes: [N, 4] XYXY boxes
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        keep: indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)
```

##### Complete Inference Example

```python
import numpy as np
import cv2
import onnxruntime as ort

# Load model
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_size = (input_shape[3], input_shape[2])  # (width, height)

# Load and preprocess image
image = cv2.imread("test.jpg")
original_size = (image.shape[1], image.shape[0])  # (width, height)
tensor, scale, pad = preprocess("test.jpg", input_size)

# Run inference
output = session.run(None, {input_name: tensor})[0]

# Decode predictions
boxes, scores, class_ids = decode_predictions(output, conf_threshold=0.25)

# Scale boxes to original image
boxes = scale_boxes(boxes, scale, pad, original_size)

# Apply NMS (per class)
final_boxes, final_scores, final_classes = [], [], []
for cls_id in np.unique(class_ids):
    cls_mask = class_ids == cls_id
    cls_boxes = boxes[cls_mask]
    cls_scores = scores[cls_mask]
    keep = nms(cls_boxes, cls_scores, iou_threshold=0.45)
    final_boxes.extend(cls_boxes[keep])
    final_scores.extend(cls_scores[keep])
    final_classes.extend([cls_id] * len(keep))

# Print results
for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
    print(f"Class {cls_id}: {score:.2%} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
```

##### Anchor Grid Reference

The model uses 3 detection scales with strides 8, 16, and 32:

| Input Size | Stride 8 | Stride 16 | Stride 32 | Total Anchors |
|------------|----------|-----------|-----------|---------------|
| 320×320 | 40×40=1600 | 20×20=400 | 10×10=100 | **2100** |
| 416×416 | 52×52=2704 | 26×26=676 | 13×13=169 | **3549** |
| 640×640 | 80×80=6400 | 40×40=1600 | 20×20=400 | **8400** |

Anchors are ordered by scale (stride 8 first, then 16, then 32), and within each scale by row-major order (left-to-right, top-to-bottom).

##### ONNX Inference Script

A standalone inference script is provided at `tools/onnx_inference.py` that performs complete inference on ONNX models without any SDK dependencies (only requires `onnxruntime`, `numpy`, and `opencv-python`).

###### Features

- **Standalone**: No YOLO SDK required, just standard ML libraries
- **Complete pipeline**: Letterbox preprocessing, inference, NMS, coordinate scaling
- **Auto-detection**: Automatically reads input size and number of classes from ONNX model
- **Visualization**: Draws bounding boxes and saves annotated images
- **Flexible output**: Console output with detection details, optional image saving

###### Usage

```bash
# Basic inference
python tools/onnx_inference.py --model model.onnx --image test.jpg

# Save annotated output
python tools/onnx_inference.py --model model.onnx --image test.jpg --output result.jpg

# Adjust thresholds
python tools/onnx_inference.py --model model.onnx --image test.jpg \
    --conf-thresh 0.5 --iou-thresh 0.45

# With class names file
python tools/onnx_inference.py --model model.onnx --image test.jpg \
    --class-names classes.txt --output result.jpg
```

###### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | Path to ONNX model file |
| `--image` | (required) | Path to input image |
| `--output` | None | Path to save annotated image (displays if not set) |
| `--conf-thresh` | 0.25 | Confidence threshold for detections |
| `--iou-thresh` | 0.45 | IoU threshold for NMS |
| `--num-classes` | auto | Number of classes (auto-detected from model) |
| `--input-size` | auto | Input size as "H W" (auto-detected from model) |
| `--class-names` | None | Path to class names file (one name per line) |

###### Output Format

The script prints detection results to console:

```
Loading model: model.onnx
Input: images [1, 3, 320, 320]
Output: output0 [1, 17, 2100]
Using model input size: (320, 320)
Processing image: test.jpg
Running inference...
Output shape: (1, 17, 2100)
Auto-detected 13 classes
Found 2 detections
  [0] Class 7: 0.986 [370.5, 301.7, 1119.5, 742.1]
  [1] Class 9: 0.989 [374.3, 1181.0, 1136.1, 1679.5]
Saved result to: result.jpg
```

Each detection shows:
- **Class ID**: Integer class index
- **Confidence**: Detection confidence (0-1)
- **Bounding box**: `[x1, y1, x2, y2]` in original image coordinates (pixels)

###### Coordinate System

The script automatically handles the coordinate transformation pipeline:

```
Original Image (e.g., 2448x3264)
        │
        ▼ letterbox resize
Letterboxed Image (320x320)
        │
        ▼ model inference
Model Output (XYWH in 320x320 space)
        │
        ▼ decode & scale_boxes
Final Detections (XYXY in original image coordinates)
```

The output bounding boxes are in the **original image coordinate system**, so they can be drawn directly on the original image without any additional transformation.

###### Class Names File Format

Create a plain text file with one class name per line:

```
class_0_name
class_1_name
class_2_name
...
```

The line number (0-indexed) corresponds to the class ID.

#### Export to TFLite

TFLite export enables deployment on mobile devices (Android, iOS), microcontrollers, and Edge TPU devices like Google Coral.

##### TFLite Dependencies

TFLite export requires additional dependencies and uses a specific export pipeline:

```
PyTorch (.ckpt) → ONNX → onnx2tf → TFLite (.tflite)
```

**Important:** The `onnxsim` (ONNX Simplifier) step is **critical** for TFLite export. It propagates static shapes through the ONNX graph, which resolves shape inference issues that would otherwise cause `onnx2tf` conversion to fail.

##### Option 1: Native Installation (Linux x86_64)

For Linux x86_64 systems, you can install dependencies directly:

```shell
# Install TFLite export dependencies
pip install tensorflow>=2.15.0 onnx2tf>=1.25.0 onnx>=1.14.0 onnxsim>=0.4.0

# Or use the setup script
chmod +x scripts/setup_tflite_export.sh
./scripts/setup_tflite_export.sh
```

##### Option 2: Docker (macOS, Windows, ARM)

**Why Docker?** TFLite export using `onnx2tf` requires specific dependencies that are challenging to install on:
- **macOS** (especially Apple Silicon M1/M2/M3)
- **Windows**
- **ARM-based Linux systems**

The main issues are:
1. `onnx2tf` and `tensorflow` have complex dependency chains
2. Some dependencies are only available for x86_64 architecture
3. Conflicting numpy versions between tensorflow and onnxruntime

We provide a Docker image based on [PINTO0309/onnx2tf](https://github.com/PINTO0309/onnx2tf) that handles all these complexities.

**Build the Docker image:**

```shell
# Build the TFLite export Docker image (one-time setup)
docker build --platform linux/amd64 -t yolo-tflite-export -f docker/Dockerfile.tflite-export .
```

**Export using Docker:**

```shell
# Export to TFLite (FP32)
docker run --platform linux/amd64 -v $(pwd):/workspace yolo-tflite-export \
    --checkpoint /workspace/runs/best.ckpt \
    --format tflite \
    --output /workspace/model.tflite

# Export with FP16 quantization
docker run --platform linux/amd64 -v $(pwd):/workspace yolo-tflite-export \
    --checkpoint /workspace/runs/best.ckpt \
    --format tflite \
    --quantization fp16 \
    --output /workspace/model_fp16.tflite

# Export with INT8 quantization (requires calibration images)
docker run --platform linux/amd64 -v $(pwd):/workspace yolo-tflite-export \
    --checkpoint /workspace/runs/best.ckpt \
    --format tflite \
    --quantization int8 \
    --calibration-images /workspace/path/to/calibration/images/ \
    --num-calibration 100 \
    --output /workspace/model_int8.tflite
```

##### TFLite Export Commands (Native)

If you have dependencies installed natively:

```shell
# Export to TFLite (FP32 - full precision)
python -m yolo.cli export --checkpoint runs/best.ckpt --format tflite

# Export with FP16 quantization (half size, minimal accuracy loss)
python -m yolo.cli export --checkpoint runs/best.ckpt --format tflite --quantization fp16

# Export with INT8 quantization (smallest size, requires calibration)
python -m yolo.cli export --checkpoint runs/best.ckpt --format tflite \
    --quantization int8 \
    --calibration-images /path/to/train/images/ \
    --num-calibration 100
```

#### Export to TensorFlow SavedModel

SavedModel format is useful for TensorFlow Serving and TensorFlow.js deployments.

```shell
# Export to SavedModel format
python -m yolo.cli export --checkpoint runs/best.ckpt --format saved_model

# With Docker (if dependencies are not available)
docker run --platform linux/amd64 -v $(pwd):/workspace yolo-tflite-export \
    --checkpoint /workspace/runs/best.ckpt \
    --format saved_model \
    --output /workspace/saved_model/
```

#### Export Options Reference

**Common options (all formats):**

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint, -c` | required | Path to model checkpoint (.ckpt) |
| `--output, -o` | auto | Output path (auto-generated if not specified) |
| `--format, -f` | onnx | Export format: `onnx`, `tflite`, or `saved_model` |
| `--size` | 640 | Input image size (height and width) |
| `--device` | auto | Device for export (cuda/cpu) |

**ONNX-specific options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--opset` | 17 | ONNX opset version (13 recommended for TFLite conversion) |
| `--simplify` | false | Simplify ONNX model using onnxsim |
| `--dynamic-batch` | false | Enable dynamic batch size dimension |
| `--half` | false | Export in FP16 precision (CUDA only) |

**TFLite-specific options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--quantization, -q` | fp32 | Quantization mode: `fp32`, `fp16`, or `int8` |
| `--calibration-images` | - | Directory with representative images (required for INT8) |
| `--num-calibration` | 100 | Number of calibration images for INT8 quantization |
| `--xnnpack-optimize` | true | Apply XNNPACK graph rewrites (SiLU→HardSwish, DFL Conv3D→Conv2D) to maximize CPU delegation (disable with `--no-xnnpack-optimize`) |

#### Quantization Comparison

| Mode | Model Size | Inference Speed | Accuracy | Use Case |
|------|------------|-----------------|----------|----------|
| **FP32** | 1x (baseline) | 1x | Best | Development, cloud GPU |
| **FP16** | ~0.5x | ~1.5-2x | Near FP32 | GPU inference, mobile GPU |
| **INT8** | ~0.25x | ~2-4x | Slight loss | Mobile CPU, Edge TPU, Coral |

**Example model sizes (YOLOv9-T):**
- FP32: ~10.8 MB
- FP16: ~5.6 MB
- INT8: ~2.8 MB (approximate)

#### INT8 Full Quantization

INT8 quantization provides the smallest model size and fastest inference, ideal for Edge TPU (Google Coral), mobile CPUs, and microcontrollers. It requires a **calibration dataset** to determine optimal quantization ranges for each layer.

##### How INT8 Calibration Works

1. **Representative Dataset**: You provide a set of images representative of your inference data
2. **Forward Pass**: The model runs inference on these images to collect activation statistics
3. **Range Calculation**: Min/max ranges are computed for each tensor in the network
4. **Quantization**: Weights and activations are mapped from FP32 to INT8 using these ranges

##### INT8 Export Commands

**Native (Linux x86_64):**

```shell
# INT8 with calibration images from training set
python -m yolo.cli export --checkpoint runs/best.ckpt --format tflite \
    --quantization int8 \
    --calibration-images data/train/images/ \
    --num-calibration 200

# INT8 with custom calibration dataset
python -m yolo.cli export --checkpoint runs/best.ckpt --format tflite \
    --quantization int8 \
    --calibration-images /path/to/calibration/images/ \
    --num-calibration 100 \
    --output model_int8.tflite
```

**Docker (macOS, Windows, ARM):**

```shell
# Build Docker image first (one-time)
docker build --platform linux/amd64 -t yolo-tflite-export -f docker/Dockerfile.tflite-export .

# INT8 export with calibration
docker run --platform linux/amd64 -v $(pwd):/workspace yolo-tflite-export \
    --checkpoint /workspace/runs/best.ckpt \
    --format tflite \
    --quantization int8 \
    --calibration-images /workspace/data/train/images/ \
    --num-calibration 200 \
    --output /workspace/model_int8.tflite
```

##### Preparing Calibration Images

The calibration dataset should be **representative** of your actual inference data:

```shell
# Option 1: Use a subset of training images (recommended)
# The export command will automatically sample from the directory

# Option 2: Create a dedicated calibration folder
mkdir -p data/calibration
# Copy diverse images that represent your use case
cp data/train/images/image_001.jpg data/calibration/
cp data/train/images/image_050.jpg data/calibration/
# ... select 100-300 diverse images
```

##### Calibration Best Practices

| Factor | Recommendation | Impact |
|--------|----------------|--------|
| **Number of images** | 100-300 | More = better accuracy, slower export |
| **Image diversity** | Various scenes, lighting, object sizes | Better generalization |
| **Image source** | From your training/validation set | Matches deployment distribution |
| **Image format** | JPG, PNG, BMP supported | Automatically detected |

**Tips for best INT8 accuracy:**

1. **Include edge cases**: Images with small objects, crowded scenes, different lighting
2. **Balance classes**: Include images with all object classes you want to detect
3. **Match deployment**: Use images similar to what the model will see in production
4. **Avoid outliers**: Don't include corrupted or unrepresentative images

##### INT8 Quantization Types

The export uses **per-channel quantization** by default, which provides better accuracy than per-tensor quantization:

| Quantization Type | Accuracy | Speed | Compatibility |
|-------------------|----------|-------|---------------|
| Per-tensor | Lower | Fastest | All hardware |
| **Per-channel** (default) | Higher | Fast | Most modern hardware |

##### Expected Model Sizes

| Model | FP32 | FP16 | INT8 |
|-------|------|------|------|
| YOLOv9-T | 10.8 MB | 5.6 MB | ~2.8 MB |
| YOLOv9-S | 28 MB | 14 MB | ~7 MB |
| YOLOv9-M | 80 MB | 40 MB | ~20 MB |
| YOLOv9-C | 100 MB | 50 MB | ~25 MB |

##### Troubleshooting INT8 Export

**Issue: Poor detection accuracy after INT8 quantization**
- Increase calibration images (try 300+)
- Ensure calibration images are representative
- Some models may not quantize well - try FP16 instead

**Issue: Export fails with memory error**
- Reduce `--num-calibration` to 50-100
- Ensure Docker has enough memory allocated (8GB+ recommended)

**Issue: Calibration takes too long**
- Reduce `--num-calibration` (100 is usually sufficient)
- Use smaller input size `--size 416`

#### Quantization-Aware Training (QAT)

If post-training INT8 quantization results in unacceptable accuracy loss, **Quantization-Aware Training (QAT)** can recover most of the accuracy by simulating INT8 quantization during training.

##### Why QAT?

YOLOv9's **Distribution Focal Loss (DFL)** decoder uses softmax over 16 values for precise coordinate prediction. This layer is particularly sensitive to INT8 quantization. Post-training quantization (PTQ) can cause significant accuracy degradation (~20-40% mAP loss). QAT fine-tunes the model with fake quantization operators, allowing it to learn to be robust to quantization errors.

**Expected results:**
- **PTQ INT8**: ~56% of original mAP (significant loss)
- **QAT INT8**: ~95-98% of original mAP (minimal loss)

##### QAT Fine-tuning Command

```shell
# Basic QAT fine-tuning (20 epochs recommended)
python -m yolo.cli qat-finetune \
    --checkpoint runs/best.ckpt \
    --config config.yaml \
    --epochs 20 \
    --lr 0.0001

# QAT with custom settings
python -m yolo.cli qat-finetune \
    --checkpoint runs/best.ckpt \
    --config config.yaml \
    --epochs 30 \
    --lr 0.00005 \
    --backend qnnpack \
    --freeze-bn-after 10 \
    --output runs/qat

# QAT with automatic TFLite export
python -m yolo.cli qat-finetune \
    --checkpoint runs/best.ckpt \
    --config config.yaml \
    --epochs 20 \
    --export-tflite \
    --calibration-images data/train/images/
```

##### QAT CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint, -c` | required | Path to pre-trained model checkpoint |
| `--config` | required | Path to config YAML (for data configuration) |
| `--epochs` | 20 | Number of QAT fine-tuning epochs |
| `--lr` | 0.0001 | Learning rate (lower than training) |
| `--batch-size` | 16 | Batch size for training |
| `--backend` | qnnpack | Quantization backend: `qnnpack` (mobile), `x86`, `fbgemm` |
| `--freeze-bn-after` | 5 | Freeze batch norm statistics after N epochs |
| `--val-every` | 1 | Validation frequency (epochs) |
| `--output, -o` | runs/qat | Output directory |
| `--export-tflite` | false | Export to INT8 TFLite after training |
| `--calibration-images` | - | Directory for TFLite INT8 calibration |

##### QAT Best Practices

1. **Start from a well-trained model**: QAT fine-tunes an existing checkpoint, not from scratch
2. **Use low learning rate**: 0.0001 or lower (10-100x smaller than original training)
3. **Short training**: 10-20 epochs is usually sufficient
4. **Freeze batch norm early**: After ~5 epochs, freeze BN statistics for stability
5. **Backend selection**:
   - `qnnpack`: Best for ARM/mobile deployment (Android, iOS, Raspberry Pi)
   - `x86`: Best for x86 CPU deployment (Intel, AMD servers)
   - `fbgemm`: Facebook's optimized backend for x86

##### QAT Workflow Example

```shell
# Step 1: Train your model normally
python -m yolo.cli fit --config yolo/config/experiment/default.yaml

# Step 2: Fine-tune with QAT (10-20 epochs)
python -m yolo.cli qat-finetune \
    --checkpoint runs/yolo/version_0/checkpoints/best.ckpt \
    --config config.yaml \
    --epochs 20 \
    --output runs/qat

# Step 3: Export the QAT model to INT8 TFLite
python -m yolo.cli export \
    --checkpoint runs/qat/qat_finetune/last.ckpt \
    --format tflite \
    --quantization int8 \
    --calibration-images data/train/images/

# Step 4: Validate accuracy
python -m yolo.cli validate \
    --checkpoint runs/qat/model_int8.tflite \
    --config config.yaml
```

##### QAT vs PTQ Comparison

| Method | Training Time | Accuracy Recovery | Complexity |
|--------|--------------|-------------------|------------|
| **PTQ (Post-Training)** | None | ~56-80% of FP32 | Low |
| **QAT (Quantization-Aware)** | 10-20 epochs | ~95-98% of FP32 | Medium |

**When to use QAT:**
- INT8 accuracy from PTQ is unacceptable
- Deploying to edge devices where INT8 is required
- Model has precision-sensitive layers (DFL, attention, etc.)

**When PTQ is sufficient:**
- Small accuracy loss is acceptable
- Quick deployment without retraining
- Model architecture is quantization-friendly

#### TFLite Technical Notes

**Input/Output format differences:**
- PyTorch/ONNX: NCHW format (batch, channels, height, width)
- TFLite: NHWC format (batch, height, width, channels)
- The conversion handles this automatically

**Why onnxsim is critical:**
The YOLO model contains `AveragePool` operations with shapes that depend on input dimensions. Without `onnxsim`, these shapes remain dynamic (`None`) and cause `onnx2tf` to fail with:
```
TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
```
The `onnxsim` step propagates concrete shapes through the graph, resolving this issue.

##### Full XNNPACK Delegation (No Retraining)

TFLite uses the **XNNPACK delegate** for CPU acceleration, but some ops prevent delegation and force fallback to the default interpreter.

In YOLOv9, the main blockers were:
- **SiLU activations** (`x * sigmoid(x)`) → `Sigmoid` becomes **`LOGISTIC`** in TFLite (often not delegated)
- **DFL decoding** → produces **`CONV_3D`** + **`GATHER`** in TFLite (not delegated)

To avoid retraining, the export pipeline applies **graph-level rewrites on the ONNX model** (before `onnx2tf`):

- **SiLU → HardSwish** (`yolo/tools/export.py::_replace_silu_with_hardswish`)
  - Replaces `x * sigmoid(x)` with `x * clip(x + 3, 0, 6) / 6`
  - Eliminates `LOGISTIC` while keeping an activation with similar behavior
- **DFL Conv3D+Gather → Conv2D+Reshape** (`yolo/tools/export.py::_replace_dfl_conv3d_gather_with_conv2d`)
  - Rewrites the 5D `Conv3D` expected-value step to an equivalent `Conv2D` by reshaping `[N,C,D,H,W] → [N,C,D·H,W]`
  - Removes both `CONV_3D` and the follow-up `GATHER`, without changing weights

This is enabled by default via `--xnnpack-optimize` (disable with `--no-xnnpack-optimize`).

**Result (example YOLOv9-T, 256x256):**
- `LOGISTIC`: **237 → 0**
- `CONV_3D`: **6 → 0**
- `GATHER`: **6 → 0**

**Conversion pipeline:**
```
1. PyTorch checkpoint → Load model
2. Model → ONNX (opset 13)
3. ONNX → onnxsim (simplify and propagate shapes)
4. Simplified ONNX → onnx2tf → TensorFlow SavedModel
5. SavedModel → TFLite Converter → .tflite file
```

#### Docker Image Details

The Docker image (`docker/Dockerfile.tflite-export`) is based on `pinto0309/onnx2tf:1.26.3` and includes:

- TensorFlow 2.18.0
- onnx2tf 1.26.3
- PyTorch 2.2.0 (CPU)
- All YOLO dependencies

**Building the image:**
```shell
docker build --platform linux/amd64 -t yolo-tflite-export -f docker/Dockerfile.tflite-export .
```

**Image size:** ~8 GB (due to TensorFlow and PyTorch)

**Note:** On Apple Silicon Macs (M1/M2/M3), Docker runs the amd64 image through Rosetta emulation. This is slower but works correctly.

#### Benchmark Results

Actual export results for YOLOv9-T model (pretrained on COCO, 80 classes), evaluated on COCO val2017 (500 images):

| Format | File Size | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | AR@100 |
|--------|-----------|--------------|---------|----------|--------|
| **PyTorch** | 12 MB | 0.412 | 0.558 | 0.447 | 0.631 |
| **ONNX** | 11 MB | 0.412 | 0.558 | 0.447 | 0.631 |
| **TFLite FP32** | 11 MB | 0.421 | 0.568 | 0.454 | 0.590 |
| **TFLite FP16** | 5.8 MB | 0.421 | 0.568 | 0.454 | 0.590 |
| **TFLite INT8** | 3.3 MB | 0.337 | 0.469 | 0.353 | 0.559 |

**Key observations:**

1. **PyTorch ↔ ONNX**: Identical metrics, confirming correct ONNX export
2. **ONNX ↔ TFLite FP32/FP16**: Slight improvement (~2%) due to XNNPACK optimizations (SiLU→HardSwish)
3. **FP32 ↔ FP16**: Virtually identical metrics, FP16 recommended for half the size
4. **FP32 → INT8**: ~20% mAP degradation (0.421 → 0.337), typical for per-channel INT8 quantization

**Note on INT8 accuracy:** The ~20% accuracy loss is expected for INT8 quantization on detection models. For applications requiring higher accuracy, use FP16. For edge deployment where model size is critical, INT8 provides 3x size reduction with acceptable accuracy for many use cases.

**Export configuration:**
- Input size: 640x640
- ONNX opset: 17
- INT8 calibration: 100 images from COCO val2017 (per-channel quantization)

## Features

| Feature | Description |
|---------|-------------|
| **Multi-GPU Training** | Automatic DDP with `--trainer.devices=N` |
| **Mixed Precision** | FP16/BF16 training with `--trainer.precision=16-mixed` |
| **Advanced Metrics** | mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, Confusion Matrix |
| **Metrics Plots** | Auto-generated PR, F1, P, R curves per epoch |
| **Multiple LR Schedulers** | Cosine, Linear, Step, OneCycle |
| **Layer Freezing** | Transfer learning with backbone freezing |
| **Checkpointing** | Automatic best/last model saving (see [Checkpoints](#checkpoints)) |
| **Early Stopping** | Stop on validation plateau |
| **Logging** | TensorBoard, WandB support |
| **Custom Image Loader** | Support for encrypted or custom image formats |
| **Mosaic Augmentation** | 4-way and 9-way mosaic (see [Advanced Training](#advanced-training-techniques)) |
| **MixUp / CutMix** | Image blending augmentations |
| **EMA** | Exponential Moving Average of model weights |
| **Close Mosaic** | Disable augmentation for final N epochs |
| **Optimizer Selection** | SGD (default) or AdamW |
| **Model Export** | ONNX, TFLite (FP32/FP16/INT8), SavedModel |
| **Standalone Validation** | Eval dashboard with full COCO metrics, plots, JSON export |
| **Integrated Benchmark** | Measure latency/memory during validation with `--benchmark` |

## Image Loaders

The training pipeline includes multiple high-performance image loaders optimized for different use cases.

### Available Loaders

| Loader | Description | Dependencies |
|--------|-------------|--------------|
| **DefaultImageLoader** | Standard PIL loader with proper file handle management | None (included) |
| **FastImageLoader** | OpenCV-based with memory-mapped I/O for large files | `opencv-python` |
| **TurboJPEGLoader** | Ultra-fast JPEG loading (2-4x speedup) | `PyTurboJPEG`, libjpeg-turbo |
| **EncryptedImageLoader** | AES-256 encrypted images (.enc files) | `cryptography` |

All loaders are optimized for:
- Large batch sizes (128+)
- Many DataLoader workers
- Proper file handle management (no "Too many open files" errors)

### Built-in Encrypted Image Loader

For datasets with AES-256 encrypted images (.enc files), use the built-in `EncryptedImageLoader`:

```shell
# Install cryptography dependency
pip install cryptography

# Set encryption key (64 hex characters = 32 bytes for AES-256)
export YOLO_ENCRYPTION_KEY=<your-64-char-hex-key>
```

**Configuration via YAML:**

```yaml
data:
  root: data/encrypted-dataset
  image_loader:
    class_path: yolo.data.encrypted_loader.EncryptedImageLoader
    init_args:
      use_opencv: true  # Use OpenCV for faster decoding (default: true)
```

**Configuration via CLI:**

```shell
python -m yolo.cli fit --config config.yaml \
    --data.image_loader.class_path=yolo.data.encrypted_loader.EncryptedImageLoader
```

The loader automatically handles:
- Encrypted files (`.enc` extension): decrypts using AES-256-CBC
- Regular files: loads normally with optimal performance

### High-Performance Loaders

For maximum throughput on standard datasets:

```yaml
# OpenCV-based loader (good for large batch sizes)
data:
  image_loader:
    class_path: yolo.data.loaders.FastImageLoader
    init_args:
      use_mmap: true  # Memory-mapped I/O for files >1MB

# TurboJPEG loader (fastest for JPEG datasets)
data:
  image_loader:
    class_path: yolo.data.loaders.TurboJPEGLoader
```

### Creating a Custom Loader

For special formats (cloud storage, proprietary formats), create a custom loader:

```python
# my_loaders.py
import io
from PIL import Image
from yolo.data.loaders import ImageLoader

class CloudStorageLoader(ImageLoader):
    """Loader for images from cloud storage."""

    def __init__(self, bucket: str):
        self.bucket = bucket
        self._client = None  # Lazy init

    def __call__(self, path: str) -> Image.Image:
        # Download from cloud
        data = self._get_client().download(self.bucket, path)

        # IMPORTANT: Use context manager and load() for proper handle management
        with io.BytesIO(data) as buf:
            with Image.open(buf) as img:
                img.load()  # Force load into memory
                return img.convert("RGB")
```

**Important for custom loaders:**
- Always use context managers (`with` statements) for file operations
- Call `img.load()` before the context manager closes to force data into memory
- This prevents "Too many open files" errors with large batch sizes

### Configuration via YAML

```yaml
data:
  root: data/cloud-dataset
  image_loader:
    class_path: my_loaders.CloudStorageLoader
    init_args:
      bucket: "my-training-bucket"
```

### Notes

- Custom loaders must return a PIL Image in RGB mode
- The loader must be picklable for multi-worker data loading (`num_workers > 0`)
- When using a custom loader, a log message will confirm: `📷 Using custom image loader: CloudStorageLoader`

## Data Loading Performance

Best practices for optimizing data loading performance during training.

### DataLoader Settings

```yaml
data:
  num_workers: 8        # Parallel data loading workers (default: 8)
  pin_memory: true      # Faster GPU transfer (default: true)
  batch_size: 16        # Adjust based on GPU memory
  prefetch_factor: 4    # Batches to prefetch per worker (default: 4)
```

**Guidelines:**
- `num_workers`: Set to number of CPU cores, or 2x for I/O-bound workloads
- `pin_memory`: Keep `true` for GPU training
- `batch_size`: Larger batches improve throughput but require more VRAM
- `prefetch_factor`: Number of batches each worker loads in advance. Higher values use more RAM but reduce GPU stalls. With `num_workers=8` and `prefetch_factor=4`, there are 32 batches ready in queue.

### File Descriptor Limits (num_workers)

When using many workers (`num_workers > 64`), you may hit the system's file descriptor limit. The training CLI **automatically detects** this and reduces workers with a warning:

```
⚠️ Reducing num_workers: 128 → 40 (ulimit -n = 1024).
   To use 128 workers, run: ulimit -n 2920
```

**Recommended ulimit values:**

| num_workers | ulimit min | ulimit recommended |
|-------------|------------|-------------------|
| 8 | 1,120 | 2,000 |
| 16 | 1,240 | 4,000 |
| 32 | 1,480 | 8,000 |
| 64 | 1,960 | 16,000 |
| 128 | 2,920 | 32,000 |
| 256 | 4,840 | **65,536** |

**Formula:** `ulimit = num_workers × 15 + 1000`

**To increase the limit:**

```shell
# Check current limits
ulimit -Sn  # soft limit (current)
ulimit -Hn  # hard limit (maximum)

# Increase for current session (Linux/macOS)
ulimit -n 65536
```

> **Note:** If you get `cannot modify limit: Operation not permitted`, the hard limit is too low. You need root access to increase it.

**Permanent increase (Linux):**

```shell
# Add to /etc/security/limits.conf (requires root)
sudo bash -c 'echo "* soft nofile 65536" >> /etc/security/limits.conf'
sudo bash -c 'echo "* hard nofile 65536" >> /etc/security/limits.conf'

# Logout and login again for changes to take effect
```

**Permanent increase (macOS):**

```shell
# Add to /etc/launchd.conf (requires root)
sudo bash -c 'echo "limit maxfiles 65536 200000" >> /etc/launchd.conf'

# Reboot for changes to take effect
```

**Docker:**

```shell
# Via docker run
docker run --ulimit nofile=65536:65536 ...

# Via docker-compose.yml
services:
  training:
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
```

**Kubernetes:**

```yaml
# In pod spec
securityContext:
  sysctls:
    - name: fs.file-max
      value: "65536"
```

### Storage Optimization

| Method | Use Case | Speedup |
|--------|----------|---------|
| **SSD** | Standard training | Baseline |
| **NVMe SSD** | Large datasets | 2-3x vs HDD |
| **RAM disk** | Small datasets (<32GB) | 5-10x |

**RAM disk setup (Linux):**
```shell
# Copy dataset to RAM disk
cp -r data/coco /dev/shm/coco

# Update config
python -m yolo.cli fit --config config.yaml --data.root=/dev/shm/coco
```

### Caching for Custom Loaders

When using expensive custom loaders (e.g., decryption, cloud storage), use the built-in image caching instead of implementing your own:

```yaml
data:
  cache_images: ram            # Use memory-mapped RAM cache
  cache_resize_images: true    # Resize to image_size for memory efficiency
  image_loader:
    class_path: yolo.data.encrypted_loader.EncryptedImageLoader
```

**How it works:**
- First run: Images are loaded via your custom loader and cached to a memory-mapped file
- All DataLoader workers share the same cache without serialization overhead
- Subsequent batches read from cache (no decryption/download needed)

**For disk caching with encryption:**

```yaml
data:
  cache_images: disk           # Save to disk as .npy files
  cache_encrypt: true          # Encrypt cache files (.npy.enc)
  encryption_key: "your-64-char-hex-key"
```

This ensures cached images are encrypted on disk, maintaining security for sensitive datasets.

## Advanced Training Techniques

This implementation includes advanced training techniques that improve model accuracy and robustness. All features are configurable via YAML/CLI and can be individually enabled or disabled.

### Data Augmentation

Augmentations are applied in two stages:

1. **Multi-image** (dataset wrapper): Mosaic (4-way/9-way), optional MixUp on top of Mosaic, optional CutMix on single images.
2. **Single-image** (transform pipeline): `LetterBox` → `RandomPerspective` → `RandomHSV` → `RandomFlip`.

All parameters live under `data:` and can be overridden via CLI (`--data.*=...`).

#### Mosaic Augmentation

Mosaic combines multiple images into a single training sample, improving detection of small objects and increasing batch diversity.

| Variant | Description |
|---------|-------------|
| **4-way Mosaic** | Combines 4 images in a 2x2 grid |
| **9-way Mosaic** | Combines 9 images in a 3x3 grid |

**Configuration:**

```yaml
data:
  mosaic_prob: 1.0      # 1.0 = always apply, 0.0 = disable
  mosaic_9_prob: 0.0    # Probability of 9-way vs 4-way (0.0 = always 4-way)
```

**CLI override:**

```shell
# Disable mosaic entirely
python -m yolo.cli fit --config config.yaml --data.mosaic_prob=0.0

# Use 50% mosaic with 30% chance of 9-way
python -m yolo.cli fit --config config.yaml \
    --data.mosaic_prob=0.5 \
    --data.mosaic_9_prob=0.3
```

#### BBox Mosaic (Document-Optimized)

BBox Mosaic is a specialized mosaic augmentation designed for **documents, logos, products, and objects that must always appear complete** (never partially cropped).

Unlike standard mosaic which can crop images at quadrant boundaries, bbox_mosaic:

1. **Crops each image to its bounding box** - Extracts only the object/document
2. **Applies individual transforms per crop** - Each document gets its own rotation, scale, HSV
3. **Scales to fit in quadrants** - Documents are sized to fit without overlapping
4. **Places in a 2x2 grid** - With optional position jitter for variety
5. **Tracks bounding boxes through transforms** - Accurate labels even after rotation

**When to use BBox Mosaic:**

| Use Case | Why BBox Mosaic |
|----------|-----------------|
| **ID Documents** | Documents must be fully visible, flipped documents are invalid |
| **Logos/Icons** | Brand assets should never be cropped |
| **Product Images** | E-commerce products need complete visibility |
| **Medical Images** | Complete specimen visibility is critical |
| **Single-object datasets** | When each image contains one main object |

**Configuration:**

```yaml
data:
  # Disable standard mosaic when using bbox_mosaic
  mosaic_prob: 0.0

  # BBox Mosaic configuration (nested structure)
  bbox_mosaic:
    prob:        0.6      # Probability of applying (0.0 = disable, 1.0 = always)
    degrees:     20.0     # Max rotation per document (±degrees)
    translate:   0.05     # Translation in affine transform (fraction)
    scale:       0.15     # Scale variation (0.85x to 1.15x)
    shear:       2.0      # Max shear per document (±degrees)
    perspective: 0.0      # Perspective distortion (0.0 = disable)
    hsv_h:       0.015    # Hue variation per document
    hsv_s:       0.4      # Saturation variation per document
    hsv_v:       0.3      # Value/brightness variation per document
    jitter:      0.3      # Position jitter in quadrant (0=centered, 1=full range)
```

**Parameter Details:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `prob` | 0.0-1.0 | Probability of applying bbox_mosaic. Set to 0.0 to disable. |
| `degrees` | 0.0-360.0 | Maximum rotation angle (±degrees). The canvas auto-expands to fit rotated images without clipping. |
| `translate` | 0.0-1.0 | Translation as fraction of canvas size in the affine transform. |
| `scale` | 0.0-1.0 | Scale variation range. `scale=0.15` means documents can be 0.85x to 1.15x their original size. |
| `shear` | 0.0-90.0 | Maximum shear angle (±degrees). Adds perspective-like distortion. |
| `perspective` | 0.0-0.01 | Perspective distortion factor. Keep at 0.0 for documents. |
| `hsv_h` | 0.0-1.0 | Hue shift gain. Applied individually to each document. |
| `hsv_s` | 0.0-1.0 | Saturation shift gain. Applied individually to each document. |
| `hsv_v` | 0.0-1.0 | Value/brightness shift gain. Applied individually to each document. |
| `jitter` | 0.0-1.0 | Position variability within quadrant. `0.0`=always centered, `0.5`=up to 50% offset, `1.0`=can touch quadrant edges. |

**How Rotation Works:**

Standard mosaic clips images when rotated. BBox Mosaic uses an **expanded canvas** approach:

```
Original 100x100 image rotated 20°:
┌─────────────────────┐
│   ┌─────────────┐   │  ← Expanded canvas (calculated from rotation formula)
│  /│             │\  │     new_W = W·|cos(θ)| + H·|sin(θ)|
│ / │  Original   │ \ │     new_H = W·|sin(θ)| + H·|cos(θ)|
│/  │   Image     │  \│
│\  │  (rotated)  │  /│  ← Document is fully visible, no clipping
│ \ │             │ / │
│  \│_____________│/  │
│                     │
└─────────────────────┘
        ↓
Scaled to fit quadrant (with 15% margin)
```

**Bounding Box Tracking:**

BBox Mosaic tracks the 4 corners of each document through all transforms:

1. Original corners: `[(0,0), (W,0), (W,H), (0,H)]`
2. Apply transformation matrix M to corners
3. Find min/max of transformed corners → new bounding box
4. Scale and offset to final canvas position

This ensures **tight, accurate bounding boxes** even after rotation and shear.

**Example Output:**

```
┌──────────────┬──────────────┐
│              │              │
│   ┌────┐     │     ┌──────┐ │
│   │Doc1│     │     │ Doc2 │ │  ← Each document is:
│   │ 15°│     │     │ -10° │ │    - Cropped to bbox
│   └────┘     │     └──────┘ │    - Individually rotated
│              │              │    - HSV adjusted
├──────────────┼──────────────┤    - Scaled to fit
│    ┌───────┐ │ ┌────┐       │    - Positioned with jitter
│    │ Doc3  │ │ │Doc4│       │
│    │  5°   │ │ │-20°│       │
│    └───────┘ │ └────┘       │
│              │              │
└──────────────┴──────────────┘
        320x320 output canvas
```

**Comparison: Standard Mosaic vs BBox Mosaic:**

| Feature | Standard Mosaic | BBox Mosaic |
|---------|-----------------|-------------|
| **Object visibility** | May be cropped at boundaries | Always fully visible |
| **Per-image transforms** | No (global only) | Yes (individual) |
| **Canvas size** | 2x image_size (then cropped) | 1x image_size (final) |
| **Best for** | General detection | Documents, single objects |
| **Rotation handling** | Can clip objects | Expanded canvas, no clipping |
| **Background** | Original image backgrounds | Gray fill (114) |

**CLI Override:**

```shell
# Enable bbox_mosaic with custom parameters
python -m yolo fit --config config.yaml \
    --data.mosaic_prob=0.0 \
    --data.bbox_mosaic.prob=0.8 \
    --data.bbox_mosaic.degrees=15.0 \
    --data.bbox_mosaic.jitter=0.5

# Disable bbox_mosaic
python -m yolo fit --config config.yaml --data.bbox_mosaic.prob=0.0
```

**Visualization:**

```shell
# Visualize bbox_mosaic augmentations
python -m yolo visualize-aug --config config.yaml --num-samples 10

# Test with high rotation to verify no clipping
python -m yolo visualize-aug --config config.yaml --num-samples 5 \
    --data.bbox_mosaic.degrees=30.0
```

**Integration with close_mosaic_epochs:**

BBox Mosaic respects `close_mosaic_epochs`. When enabled, bbox_mosaic is disabled for the final N epochs along with standard mosaic, mixup, and cutmix.

```yaml
data:
  bbox_mosaic:
    prob: 0.6
  close_mosaic_epochs: 5  # Disable bbox_mosaic for last 5 epochs
```

#### MixUp Augmentation

MixUp blends two images together with a random weight; for detection, boxes/labels from both images are kept. In this implementation, MixUp is applied only when Mosaic is selected for the sample (it blends two mosaic images).

**Configuration:**

```yaml
data:
  mixup_prob: 0.15      # Probability of applying mixup (0.0 = disable)
  mixup_alpha: 32.0     # Beta distribution parameter (higher = more uniform blending)
```

**CLI override:**

```shell
# Disable mixup
python -m yolo.cli fit --config config.yaml --data.mixup_prob=0.0

# Increase mixup probability
python -m yolo.cli fit --config config.yaml --data.mixup_prob=0.3
```

#### CutMix Augmentation

CutMix cuts a rectangular region from one image and pastes it onto another, combining their labels. In this implementation, CutMix is applied only when Mosaic is not selected for the sample.

**Configuration:**

```yaml
data:
  cutmix_prob: 0.0      # Probability of applying cutmix (0.0 = disable)
```

**CLI override:**

```shell
# Enable cutmix with 10% probability
python -m yolo.cli fit --config config.yaml --data.cutmix_prob=0.1
```

#### RandomPerspective

Applies geometric transformations including rotation, translation, scale, shear, and perspective distortion.

Internally it builds a 3×3 homography by composing center shift → (optional) perspective → rotation/scale → (optional) shear → translation, then warps the image with OpenCV. Boxes are filtered by minimum area/visibility after the transform.

**Configuration:**

```yaml
data:
  degrees: 0.0          # Max rotation degrees (+/-), 0.0 = no rotation
  translate: 0.1        # Max translation as fraction of image size
  scale: 0.9            # Scale range (1-scale to 1+scale)
  shear: 0.0            # Max shear degrees (+/-), 0.0 = no shear
  perspective: 0.0      # Perspective distortion, 0.0 = no perspective
```

**CLI override:**

```shell
# Enable rotation and shear
python -m yolo.cli fit --config config.yaml \
    --data.degrees=10.0 \
    --data.shear=5.0

# Disable all geometric augmentation
python -m yolo.cli fit --config config.yaml \
    --data.degrees=0 --data.translate=0 --data.scale=0 \
    --data.shear=0 --data.perspective=0
```

#### RandomHSV

Applies random hue/saturation/value shifts (color augmentation).

**Configuration:**

```yaml
data:
  hsv_h: 0.015          # Hue shift (0.0 = disable)
  hsv_s: 0.7            # Saturation gain (0.0 = disable)
  hsv_v: 0.4            # Value/brightness gain (0.0 = disable)
```

**CLI override:**

```shell
# Disable HSV augmentation
python -m yolo.cli fit --config config.yaml --data.hsv_h=0 --data.hsv_s=0 --data.hsv_v=0
```

#### RandomFlip

Applies random horizontal/vertical flips and updates bounding boxes accordingly.

**Configuration:**

```yaml
data:
  flip_lr: 0.5          # Horizontal flip probability
  flip_ud: 0.0          # Vertical flip probability
```

**CLI override:**

```shell
# Disable flips
python -m yolo.cli fit --config config.yaml --data.flip_lr=0 --data.flip_ud=0
```

### Close Mosaic

Disables mosaic, mixup, and cutmix augmentations for the final N epochs of training. This allows the model to fine-tune on "clean" single images, improving convergence.

**Configuration:**

```yaml
data:
  close_mosaic_epochs: 15   # Disable augmentations for last 15 epochs
```

**CLI override:**

```shell
# Disable close_mosaic (use augmentation until the end)
python -m yolo.cli fit --config config.yaml --data.close_mosaic_epochs=0

# Use close_mosaic for last 20 epochs
python -m yolo.cli fit --config config.yaml --data.close_mosaic_epochs=20
```

### Augmentation Visualization

Debug augmentations visually by generating images with bounding boxes. This command uses **exactly the same data pipeline as training**, ensuring what you see matches training behavior.

**Command:**

```shell
# Visualize 10 random augmented samples
python -m yolo visualize-aug --config config.yaml --num-samples 10

# Visualize specific indices with original comparison
python -m yolo visualize-aug --config config.yaml --indices 0,5,10 --show-original

# Create grid visualization (4 columns)
python -m yolo visualize-aug --config config.yaml --grid --num-samples 16

# Reproducible output with seed
python -m yolo visualize-aug --config config.yaml --num-samples 10 --seed 42
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config`, `-c` | str | required | Path to training config YAML |
| `--num-samples`, `-n` | int | 10 | Number of random samples to visualize |
| `--indices` | str | - | Specific indices (comma-separated, e.g., "0,5,10") |
| `--output`, `-o` | str | `<dataset_root>/aug_viz/` | Output directory for images |
| `--seed` | int | - | Random seed for reproducibility |
| `--show-original` | flag | false | Also save original (non-augmented) images |
| `--grid` | flag | false | Create grid image instead of separate files |

**Output:**

- Individual images: `<dataset_root>/aug_viz/aug_0001.jpg`, `<dataset_root>/aug_viz/aug_0005.jpg`, ...
- With `--show-original`: also `<dataset_root>/aug_viz/orig_0001.jpg`, ...
- With `--grid`: single `<dataset_root>/aug_viz/grid.jpg` file

**Use Cases:**

1. **Debug mosaic**: Verify boxes are correctly transformed after mosaic 4/9
2. **Verify mixup**: Check that blended images have valid combined boxes
3. **Test flip**: Confirm box coordinates flip correctly with images
4. **Inspect augmentation pipeline**: See exactly what the model sees during training

> **Note**: This command supports both YOLO and COCO dataset formats - it automatically uses the format specified in your config.

### Dataset Caching

Accelerate data loading with label caching and optional image caching. Labels are parsed once and cached; subsequent runs load instantly. The cache auto-invalidates when source files change.

**Cache System v4.0.0** - Unified caching with JPEG and RAW format support.

#### Quick Start

```bash
# 1. Create cache (one time)
cd yolo-mit
export YOLO_ENCRYPTION_KEY="$(python -c 'import os; print(os.urandom(32).hex())')"
python -m yolo cache-create --config config.yaml

# 2. Train using cache
python -m yolo fit --config config.yaml
```

#### Cache Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `none` | No image caching (default) | Small datasets, debugging |
| `ram` | Pre-fault all pages into memory | Fast SSD, enough RAM |
| `disk` | OS-managed memory-mapping, lazy loading | Large datasets, limited RAM |

Both `ram` and `disk` modes use the same unified **LMDB** (Lightning Memory-Mapped Database) backend.

#### Cache Formats

| Format | Compression | Size | Quality | Best For |
|--------|-------------|------|---------|----------|
| `jpeg` | TurboJPEG ~10x | ~5-6 GB (32K images) | Near-lossless (quality 95) | **Recommended for disk cache** |
| `raw` | LZ4 automatic | ~40% reduction | Lossless | RAM cache, exact reproduction |

**JPEG format** (default) uses TurboJPEG for fast encoding/decoding with minimal quality loss at quality=95.

**RAW format** stores uncompressed numpy arrays with automatic LZ4 compression.

#### Full Configuration Reference

```yaml
data:
  # Label caching
  cache_labels: true           # Parse labels once, cache to .cache file (default: true)

  # Image caching mode
  cache_images: none           # "none", "ram", or "disk"

  # Cache format (v4.0.0+)
  cache_format: jpeg           # "jpeg" (~10x smaller, fast) or "raw" (lossless with LZ4)
  jpeg_quality: 95             # JPEG quality 1-100 (only for cache_format: jpeg)

  # Image resize during caching
  cache_resize_images: true    # Resize to image_size when caching (default: true, saves space)

  # Storage options
  cache_dir: null              # Custom cache directory (null = alongside images)
  cache_max_memory_gb: 8.0     # Max memory for RAM mode

  # Security
  cache_encrypt: true          # AES-256 encryption (requires YOLO_ENCRYPTION_KEY env var)
  encryption_key: null         # Or set key directly (less secure than env var)

  # Performance
  cache_workers: null          # Parallel workers for caching (null = all CPU threads)
  cache_sync: false            # LMDB fsync - disable for external volumes on macOS

  # Cache management
  cache_refresh: false         # Force cache regeneration (delete and rebuild)
  cache_only: false            # Train using ONLY cache (no original images needed)
```

#### Cache Architecture

```
                         +----------------------------------------------------------+
                         |                   ImageCache v4.0.0                      |
                         |              (Unified LMDB + JPEG/RAW formats)           |
                         +----------------------------------------------------------+
                                                  |
                    +-----------------------------+-----------------------------+
                    |                                                           |
                    v                                                           v
         +-------------------+                                    +-------------------+
         |   JPEG Format     |                                    |    RAW Format     |
         | (cache_format:    |                                    | (cache_format:    |
         |      jpeg)        |                                    |       raw)        |
         +-------------------+                                    +-------------------+
         | TurboJPEG encode  |                                    | LZ4 compression   |
         | ~10x compression  |                                    | ~40% reduction    |
         | Quality: 95       |                                    | Lossless          |
         +-------------------+                                    +-------------------+
                    |                                                           |
                    +-----------------------------+-----------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
              +---------------------+                        +---------------------+
              |     RAM Mode        |                        |     Disk Mode       |
              |  (cache_images:ram) |                        | (cache_images:disk) |
              +---------------------+                        +---------------------+
              | Pre-fault pages     |                        | Lazy OS-managed     |
              | into RAM at init    |                        | memory-mapping      |
              +---------------------+                        +---------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
                         +----------------------------------------------------------+
                         |                      LMDB Storage                        |
                         |  .yolo_cache_{WxH}_f{fraction}/cache.lmdb/               |
                         |  Example: .yolo_cache_320x320_f1.0/cache.lmdb/           |
                         |           .yolo_cache_orig_f1.0/cache.lmdb/ (no resize)  |
                         +----------------------------------------------------------+
                         |  Metadata (__metadata__ key):                            |
                         |    - version: "4.0.0"                                    |
                         |    - cache_format: "jpeg" | "raw"                        |
                         |    - encrypted: true | false                             |
                         |    - target_size: (W, H) | null                          |
                         |    - image_paths: [...]                                  |
                         |    - labels: [...] (for cache-only mode)                 |
                         |    - train_indices: [...], val_indices: [...]            |
                         +----------------------------------------------------------+
```

**RAM Mode:**

The RAM cache pre-faults all pages into memory during initialization, ensuring fastest possible access. All DataLoader workers share the same memory-mapped LMDB database without serialization overhead.

- **Parallel loading**: Images are pre-loaded using multiple threads
  - By default, uses all CPU threads (`os.cpu_count()`)
  - Control with `cache_workers` parameter
- All workers access the same LMDB database
- No serialization delays when spawning workers
- Data persists between training runs

**Disk Mode:**

The disk cache uses OS-managed memory-mapping with lazy loading. Pages are loaded into memory only when accessed, making it suitable for datasets larger than available RAM.

- **Lazy loading**: Only accessed pages are loaded into memory
- **OS-managed**: The operating system handles page caching
- **Persistent**: Cache survives restarts and can be reused

**Cache and DataLoader Workers:**

The `num_workers` parameter is **not modified** based on cache type. Both RAM and Disk caches use LMDB which supports concurrent reads from multiple processes:

| Cache Mode | DataLoader Behavior |
|------------|---------------------|
| **RAM** | All workers share the same memory-mapped LMDB. Data is already in RAM, so workers read concurrently with minimal I/O. Recommended: keep default `num_workers`. |
| **Disk** | Workers read from disk via memory-mapping. OS caches frequently accessed pages. On **fast SSD**: keep default `num_workers`. On **slow HDD**: consider reducing `num_workers` to 2-4 to avoid I/O contention. |
| **None** | Workers load images from disk on each access. Higher `num_workers` helps parallelize I/O. |

> **Tip**: If using disk cache on external/slow storage, consider setting `cache_dir` to a faster drive (SSD or NVMe) for better performance.

**Switching Between RAM and Disk Modes:**

Both `ram` and `disk` modes use the same LMDB format, so you can switch between them without rebuilding the cache:

| First Run | Second Run | Result |
|-----------|------------|--------|
| `disk` | `ram` | Reuses cache, pre-faults pages into RAM |
| `ram` | `disk` | Reuses cache, lazy OS-managed loading |
| `ram` | `ram` | Reuses cache, pre-faults pages into RAM |
| `disk` | `disk` | Reuses cache, lazy OS-managed loading |

This allows you to:
1. Create the cache once with `disk` mode (lower memory during creation)
2. Reuse it with `ram` mode for fastest training (instant page loading)

**Image Resize During Caching:**

When `cache_resize_images: true` (default), images are resized to `image_size` during caching. This significantly reduces memory usage - a 4K image (4000x3000) takes ~36MB in RAM, but resized to 640x640 only ~1.2MB.

*Resize Method - Letterboxing:*

The cache uses **letterbox resize** which preserves the original aspect ratio:

1. Calculate scale factor: `scale = min(target_w/orig_w, target_h/orig_h)`
2. Resize image maintaining aspect ratio
3. Center on gray (114, 114, 114) background canvas

Example: 1920×1080 image resized to 320×320:
- Scale: `min(320/1920, 320/1080) = 0.167`
- New size: 320×180
- Padding: 70px top and bottom (gray)

*Interpolation Methods:*

| Method | Quality | Speed | Use Case |
|--------|---------|-------|----------|
| NEAREST | ⭐ | ⭐⭐⭐⭐⭐ | Masks, segmentation labels |
| **BILINEAR** | ⭐⭐⭐ | ⭐⭐⭐⭐ | **YOLO standard (cache & training)** |
| BICUBIC | ⭐⭐⭐⭐ | ⭐⭐⭐ | High-quality photo processing |
| LANCZOS | ⭐⭐⭐⭐⭐ | ⭐⭐ | Critical downscaling |

The cache uses **PIL BILINEAR** interpolation, which matches the training pipeline:

| Component | Interpolation | Consistency |
|-----------|---------------|-------------|
| Cache (`_resize_for_cache`) | PIL BILINEAR | ✓ |
| LetterBox Transform (training) | PIL BILINEAR | ✓ |
| MixUp/CutMix | PIL BILINEAR | ✓ |
| Mosaic | OpenCV INTER_LINEAR | ≈ (equivalent) |

> **Note:** PIL BILINEAR and OpenCV INTER_LINEAR are both bilinear interpolations with minimal practical differences. The choice of BILINEAR provides the optimal balance between quality and speed for object detection tasks.

*Why Letterbox with Gray Padding?*

- **Aspect ratio preserved**: Objects maintain correct proportions (critical for bounding box accuracy)
- **Gray (114, 114, 114)**: Standard YOLO padding color, reduces edge artifacts
- **Centered image**: Consistent positioning across dataset
- **No distortion**: Unlike stretch-to-fit which can degrade detection accuracy

**Encrypted Cache:**

When using encrypted images, enable `cache_encrypt` to encrypt the cached values:

```yaml
data:
  cache_images: disk
  cache_encrypt: true          # Encrypt cached array values
  encryption_key: "your-64-char-hex-key"  # Or use YOLO_ENCRYPTION_KEY env var
```

This ensures cached images are encrypted in the LMDB database, maintaining security for sensitive datasets. Note: LMDB metadata (keys, sizes) remains visible; only values are encrypted.

**Cache Format Details:**

**JPEG Format** (Recommended for most use cases):

Uses TurboJPEG for fast hardware-accelerated encoding/decoding:
- **Compression**: ~10x smaller than RAW (e.g., 5-6 GB vs 60+ GB for 32K images)
- **Quality**: Configurable (default 95 = near-lossless for object detection)
- **Speed**: Faster than LZ4 due to smaller I/O
- **Encryption**: Applied after JPEG compression

```yaml
data:
  cache_format: jpeg     # Default
  jpeg_quality: 95       # 1-100 (higher = better quality, larger files)
```

**RAW Format** (For lossless requirements):

Stores numpy arrays with automatic LZ4 compression:
- **Compression**: ~40% reduction with LZ4 (automatic, not configurable)
- **Quality**: Lossless - exact array reconstruction
- **Use case**: When exact pixel values matter (e.g., medical imaging)

```yaml
data:
  cache_format: raw      # Lossless with automatic LZ4
```

**Processing Order (with encryption):**

```
JPEG: Image → JPEG encode → AES-256 encrypt → LMDB
RAW:  Image → LZ4 compress → AES-256 encrypt → LMDB
```

**Storage Comparison (32K images at original size ~1240x1754):**

| Format | Encryption | Approximate Size |
|--------|------------|------------------|
| JPEG (q=95) | No | ~5-6 GB |
| JPEG (q=95) | Yes | ~5-6 GB |
| RAW + LZ4 | No | ~35-40 GB |
| RAW + LZ4 | Yes | ~35-40 GB |

> **Note**: `cache_compress` option is deprecated. Use `cache_format: raw` for automatic LZ4 compression, or `cache_format: jpeg` for JPEG compression.

**Performance Characteristics:**

The LMDB-based cache implementation provides excellent performance for deep learning workloads:

| Aspect | Implementation | Benefit |
|--------|----------------|---------|
| **Storage** | LMDB with memory-mapping | Zero-copy reads, no syscall per access |
| **Concurrency** | Multi-reader single-writer | Perfect for DataLoader workers (spawn mode) |
| **Serialization** | Compact header (2 + 4×ndim bytes) + raw bytes | Minimal overhead, fast encode/decode |
| **Compression** | LZ4 frame compression (optional) | ~40% size reduction, ~4 GB/s decompress |
| **Encryption** | AES-256-GCM on values only | Strong security, key never serialized |
| **Multiprocessing** | Automatic reconnection in workers | Transparent pickling support |

*Estimated storage for 137K images at 256×256×3 (uint8):*
- Uncompressed: ~27 GB
- With resize from original: typically 3-5× smaller cache than raw images

**Comparison with Alternatives:**

| Solution | Pros | Cons |
|----------|------|------|
| **LMDB** (current) | Zero-copy, multi-reader, battle-tested | Requires map_size estimation |
| HDF5 | Good compression, scientific standard | GIL contention in multiprocess |
| SQLite | Single-file, portable | Slower for binary blobs |
| Individual files | Simple, no dependencies | Slow on many small files |

**CLI override:**

```shell
# Disable label caching
python -m yolo.cli fit --config config.yaml --data.cache_labels=false

# Enable RAM image caching (LMDB, fastest access)
python -m yolo.cli fit --config config.yaml --data.cache_images=ram

# Enable disk image caching (LMDB, lazy loading)
python -m yolo.cli fit --config config.yaml --data.cache_images=disk

# Disable image resize during caching (use original resolution)
python -m yolo.cli fit --config config.yaml --data.cache_resize_images=false

# Enable encrypted cache
python -m yolo.cli fit --config config.yaml --data.cache_images=disk --data.cache_encrypt=true

# Use custom directory for cache (useful for faster storage)
python -m yolo.cli fit --config config.yaml --data.cache_images=disk --data.cache_dir=/tmp/yolo_cache

# Control number of parallel workers for caching (null = all CPU threads)
python -m yolo.cli fit --config config.yaml --data.cache_workers=16

# Force cache regeneration (delete and rebuild)
python -m yolo.cli fit --config config.yaml --data.cache_refresh=true
```

**Force Cache Refresh:**

Use `--data.cache_refresh=true` to force deletion and regeneration of the cache. Useful when:
- Dataset files were modified but timestamps didn't change
- Cache file is corrupted
- Switching between different preprocessing configurations

**Automatic Cache Invalidation:**

The cache is automatically invalidated (and rebuilt) when any of these settings change:

| Change | Message |
|--------|---------|
| Cache version upgrade | `version changed (3.2.0 → 4.0.0)` |
| Image count changed | `image count changed (100,000 → 117,266)` |
| Image files changed | `image files changed` |
| Size or fraction changed | `settings changed (size/fraction)` |
| Encryption setting changed | `encryption changed (unencrypted → encrypted)` |
| Format changed | `format changed (jpeg → raw)` |

Settings that do **not** invalidate the cache:
- `batch_size` - only affects DataLoader batching
- `num_workers` - only affects parallel loading
- `cache_max_memory_gb` - only affects RAM mode limits
- `jpeg_quality` - only affects new cache creation (not validation)

#### Cache Management CLI

The CLI provides commands to create, export, import, and inspect dataset caches. This enables secure transfer of preprocessed datasets to remote machines without exposing original images.

**Commands Overview:**

| Command | Description |
|---------|-------------|
| `cache-create` | Create LMDB cache from dataset (without training) |
| `cache-export` | Export cache directory to compressed archive |
| `cache-import` | Import cache from archive to target directory |
| `cache-info` | Display cache statistics and metadata |
| `visualize-aug` | Visualize augmentations with bounding boxes for debugging |

##### Creating Cache Without Training

Create a cache independently from training, useful for preparing datasets before deployment:

```shell
# Create cache from config file (reads image_size from model.image_size in config)
yolo cache-create --config config.yaml --encrypt

# Or specify size explicitly (overrides config)
yolo cache-create --config config.yaml --size 640 --encrypt

# Create cache with direct parameters
yolo cache-create \
    --data.root /path/to/dataset \
    --data.format yolo \
    --data.train_images train/images \
    --data.train_labels train/labels \
    --size 640 \
    --encrypt
```

The cache-create command automatically reads split files (`train_split`, `val_split`) from the config to store correct train/val indices in the cache metadata. This enables true `cache_only` mode where no original files are needed.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config` | str | - | Path to YAML config file (reads all settings from `data:` section) |
| `--data.root` | str | required | Dataset root directory |
| `--data.format` | str | yolo | Dataset format (yolo/coco) |
| `--size` | int | from config | Image size for cache (reads `model.image_size` if not specified) |
| `--no-resize` | flag | false | Store images at original size (no letterbox resize) |
| `--encrypt` | flag | from config | Encrypt cache with AES-256 (reads `cache_encrypt` from config) |
| `--cache-format` | str | from config | Cache format: `jpeg` (smaller) or `raw` (lossless) |
| `--jpeg-quality` | int | from config | JPEG quality 1-100 (only for jpeg format, default: 95) |
| `--workers` | int | auto | Parallel workers for caching |
| `--split` | str | both | Which split to cache (train/val/both) |
| `--sync` | flag | false | Enable fsync for crash safety (disable for external volumes) |
| `--output-dir` | str | data.root | Custom output directory for cache |

> **Note**: `--compress` is deprecated. Use `--cache-format raw` for LZ4 compression or `--cache-format jpeg` for JPEG compression.

##### Exporting Cache to Archive

Create a compressed archive of the cache for transfer:

```shell
# Export cache to tar.gz
yolo cache-export \
    --cache-dir dataset/.yolo_cache_640x640_f1.0 \
    --output cache_archive.tar.gz

# Export without compression (faster, larger file)
yolo cache-export \
    --cache-dir dataset/.yolo_cache_640x640_f1.0 \
    --output cache_archive.tar \
    --compression none
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--cache-dir` | str | required | Path to cache directory |
| `--output`, `-o` | str | auto | Output archive path |
| `--compression` | str | gzip | Compression type (gzip/none) |

##### Importing Cache from Archive

Extract cache archive to target directory:

```shell
# Import cache
yolo cache-import \
    --archive cache_archive.tar.gz \
    --output /data/dataset/
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--archive` | str | required | Path to cache archive |
| `--output`, `-o` | str | . | Target directory |

##### Inspecting Cache

Display cache statistics and metadata:

```shell
yolo cache-info --cache-dir dataset/.yolo_cache_320x320_f1.0
```

**Example output:**

```
Cache Information
─────────────────────────────────────
  Path:        dataset/.yolo_cache_320x320_f1.0
  Version:     4.0.0
  Format:      yolo
  Images:      32,513
  Size:        320x320
  Cache Format: jpeg (quality: 95)
  Encrypted:   Yes
  Labels:      32,513
  Train:       25,995 images
  Val:         4,858 images
─────────────────────────────────────
```

##### Cache-Only Training Mode

Train using only the cache without requiring original images on disk. This is essential when deploying to remote machines where you only transfer the encrypted cache.

```shell
# Training with cache-only mode (no original images needed)
export YOLO_ENCRYPTION_KEY="your-64-char-hex-key"
yolo fit --config config.yaml \
    --data.cache_images disk \
    --data.cache_only true
```

**When to use `cache_only`:**
- Training on remote VM where only the cache was transferred
- Protecting original images from exposure
- Reducing storage requirements on training machines

**Requirements for cache-only mode:**
- Cache must be created with `yolo cache-create` command
- Cache is self-contained (images, labels, and split indices included)
- Error raised if any image is not found in cache

##### What's in the Cache (v4.0.0+)

The `cache-create` command stores everything needed for training:

| Item | In Cache | Notes |
|------|----------|-------|
| Pre-processed images | YES | JPEG or RAW format |
| Image paths | YES | For logging/debugging |
| Labels (YOLO format) | YES | Stored in metadata |
| COCO annotations | YES | For COCO format datasets |
| Train/Val split indices | YES | From split files |
| Cache metadata | YES | Version, format, encryption, etc. |

```
# Files required on remote server (cache-only mode):
dataset/
└── .yolo_cache_320x320_f1.0/    # Complete cache (everything included)
    └── cache.lmdb/
        ├── data.mdb             # Images + metadata
        └── lock.mdb             # LMDB lock file

# No separate labels/, train.txt, val.txt needed!
```

> **Note**: For backward compatibility, if labels are not in cache, the system falls back to loading from `labels/` directory.

##### Complete Workflow: Secure Remote Training

This workflow demonstrates how to train on a remote VM without ever exposing original images:

```shell
# ============================================
# LOCAL MACHINE (has original images)
# ============================================

# 1. Generate encryption key (save this securely!)
export YOLO_ENCRYPTION_KEY=$(python -c "import os; print(os.urandom(32).hex())")
echo "Save this key: $YOLO_ENCRYPTION_KEY"

# 2. Create encrypted cache (includes images, labels, split indices)
cd yolo-mit
yolo cache-create --config config.yaml
# Creates: dataset/.yolo_cache_320x320_f1.0/ (encrypted LMDB with everything)

# 3. Export to archive
yolo cache-export \
    --cache-dir ../dataset/.yolo_cache_320x320_f1.0 \
    --output cache_encrypted.tar.gz

# 4. Transfer to remote (cache is encrypted + self-contained)
scp cache_encrypted.tar.gz user@remote:/data/

# ============================================
# REMOTE VM (no original images needed!)
# ============================================

# 5. Import cache
yolo cache-import \
    --archive cache_encrypted.tar.gz \
    --output /data/dataset/

# 6. Verify cache is present
ls /data/dataset/
# Should show: .yolo_cache_320x320_f1.0/

# 7. Train using only cache (decryption happens only in memory)
export YOLO_ENCRYPTION_KEY="your-64-char-hex-key"
yolo fit --config config.yaml \
    --data.root /data/dataset \
    --data.cache_images disk \
    --data.cache_only true

# ✓ Original images are NEVER on the remote VM
# ✓ Encrypted data on disk is NEVER decrypted to files
# ✓ Decryption happens ONLY in memory during training
# ✓ Labels are stored in encrypted cache metadata
```

##### Security Model

```
┌─────────────────────────────────────────────────────────────────┐
│                     SECURITY ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LOCAL MACHINE                    REMOTE VM                     │
│  ─────────────                    ─────────                     │
│  ┌─────────────┐                  ┌─────────────┐              │
│  │  Original   │                  │  Encrypted  │              │
│  │   Images    │ ──(encrypt)──►  │   Cache     │              │
│  └─────────────┘                  │  (on disk)  │              │
│                                   └──────┬──────┘              │
│                                          │                      │
│                                          ▼                      │
│                                   ┌─────────────┐              │
│                                   │  Decrypted  │              │
│                                   │   Images    │              │
│                                   │ (in memory) │              │
│                                   └──────┬──────┘              │
│                                          │                      │
│                                          ▼                      │
│                                   ┌─────────────┐              │
│                                   │   Training  │              │
│                                   │   Process   │              │
│                                   └─────────────┘              │
│                                                                 │
│  ✓ Original images never leave local machine                   │
│  ✓ Cache on remote is always encrypted on disk                 │
│  ✓ Decryption only in RAM during data loading                  │
│  ✓ Key required only at training time                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

##### Encryption Key Management

The encryption key is a 256-bit (32 bytes) AES key encoded as 64 hexadecimal characters.

**Setting the key:**

```shell
# Option 1: Environment variable (recommended)
export YOLO_ENCRYPTION_KEY="your-64-char-hex-key"

# Option 2: YAML configuration (less secure - key in file)
data:
  encryption_key: "your-64-char-hex-key"
```

**Generating a secure key:**

```python
import os
key = os.urandom(32).hex()
print(f"YOLO_ENCRYPTION_KEY={key}")
```

**Security best practices:**
- Never commit keys to version control
- Use environment variables or secret management systems
- Rotate keys periodically for long-running projects
- Keep a secure backup of keys (loss = data loss)

##### Use Cases and Scenarios

###### Scenario 1: Training on Sensitive Data (Medical, Financial, Personal)

When working with sensitive datasets that cannot be stored unencrypted:

```shell
# Setup: Generate and save key securely
export YOLO_ENCRYPTION_KEY=$(python -c "import os; print(os.urandom(32).hex())")
echo "$YOLO_ENCRYPTION_KEY" > ~/.yolo_key  # Save securely!
chmod 600 ~/.yolo_key

# Create encrypted cache (images remain safe on disk)
yolo cache-create --config config.yaml --size 640 --encrypt

# Train (decryption happens only in RAM)
yolo fit --config config.yaml \
    --data.cache_images disk \
    --data.cache_encrypt true
```

**Benefits:**
- Original images can be deleted after cache creation
- Disk never contains unencrypted image data
- Safe for shared/cloud storage

###### Scenario 2: Pre-Encrypted Dataset (Images Already Encrypted)

When your source images are already encrypted (e.g., from a secure data provider):

```yaml
# config.yaml
data:
  root: /data/encrypted-images  # Contains .enc files
  image_loader:
    class_path: yolo.data.encrypted_loader.EncryptedImageLoader
  cache_images: ram              # Cache decrypted images in RAM
  cache_resize_images: true      # Reduce memory footprint
```

```shell
export YOLO_ENCRYPTION_KEY="key-from-data-provider"
yolo fit --config config.yaml
```

**Flow:**
1. Encrypted images (.enc) loaded from disk
2. Decrypted in memory by EncryptedImageLoader
3. Cached in RAM (no encryption needed - RAM is volatile)

###### Scenario 3: Secure Remote Training (Cloud/VM)

Train on AWS/GCP/Azure without exposing original images:

```shell
# === LOCAL (trusted machine with original images) ===
export YOLO_ENCRYPTION_KEY=$(python -c "import os; print(os.urandom(32).hex())")

# Create and export encrypted cache
yolo cache-create --config config.yaml --size 640 --encrypt
yolo cache-export --cache-dir dataset/.yolo_cache_640x640_f1.0 -o cache.tar.gz

# Upload to cloud storage
aws s3 cp cache.tar.gz s3://my-bucket/

# === REMOTE VM ===
# Download and import
aws s3 cp s3://my-bucket/cache.tar.gz .
yolo cache-import --archive cache.tar.gz --output /data/

# Train with cache-only (no original images needed!)
export YOLO_ENCRYPTION_KEY="same-key-from-local"
yolo fit --config config.yaml \
    --data.cache_images disk \
    --data.cache_only true \
    --data.cache_encrypt true
```

###### Scenario 4: Multi-User Shared Dataset

Multiple users training on the same encrypted dataset:

```shell
# Admin creates encrypted cache once
yolo cache-create --config config.yaml --size 640 --encrypt

# Share the encryption key securely (e.g., via secret manager)
# Each user sets their environment variable
export YOLO_ENCRYPTION_KEY="shared-team-key"

# All users can train using the same cache
yolo fit --config config.yaml \
    --data.cache_images disk \
    --data.cache_encrypt true
```

##### Encrypted Images vs Encrypted Cache

| Feature | Encrypted Images (.enc) | Encrypted Cache |
|---------|------------------------|-----------------|
| **What's encrypted** | Source image files | LMDB cache values |
| **File extension** | `.enc` | Standard LMDB |
| **Encryption point** | Before training | During cache creation |
| **Requires** | `EncryptedImageLoader` | `cache_encrypt: true` |
| **Use case** | Data provider encrypts | You encrypt for security |
| **Key setting** | `YOLO_ENCRYPTION_KEY` | `YOLO_ENCRYPTION_KEY` |
| **Decryption** | On load | On cache read |

**Can be combined:** Use `EncryptedImageLoader` with encrypted source images, then cache them with `cache_encrypt: true` for double protection.

##### Common Errors and Troubleshooting

###### Error: "ENCRYPTION MISMATCH"

```
╔══════════════════════════════════════════════════════════════════════╗
║  ENCRYPTION MISMATCH                                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  The cache is ENCRYPTED but cache_encrypt: false in your config.    ║
╚══════════════════════════════════════════════════════════════════════╝
```

**Cause:** Cache was created with `--encrypt` but YAML has `cache_encrypt: false`.

**Solution:**
```yaml
data:
  cache_encrypt: true  # Must match how cache was created
```
```shell
export YOLO_ENCRYPTION_KEY="your-key"
```

###### Error: "Image not found in cache"

When using encrypted cache without the key:

**Cause:** Key not set or incorrect key.

**Solution:**
```shell
# Verify key is set
echo $YOLO_ENCRYPTION_KEY

# Set correct key
export YOLO_ENCRYPTION_KEY="correct-64-char-hex-key"
```

###### Error: "FORMAT MISMATCH"

```
╔══════════════════════════════════════════════════════════════════════╗
║  FORMAT MISMATCH                                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  YAML config:  format: coco                                          ║
║  Cache format: yolo                                                  ║
╚══════════════════════════════════════════════════════════════════════╝
```

**Cause:** Cache was created with different format than YAML config.

**Solution:** Match YAML format to how cache was created:
```yaml
data:
  format: yolo  # or coco - must match cache creation
```

###### Error: "Encryption key must be 64 hex characters"

**Cause:** Invalid key format.

**Solution:** Generate proper key:
```python
import os
print(os.urandom(32).hex())  # Produces exactly 64 hex chars
```

##### Quick Reference: YAML Parameters

```yaml
data:
  # === Source Image Encryption ===
  image_loader:
    class_path: yolo.data.encrypted_loader.EncryptedImageLoader
    init_args:
      use_opencv: true

  # === Cache Encryption ===
  cache_images: disk        # Must be 'disk' for encryption
  cache_encrypt: true       # Encrypt cache values
  cache_only: true          # Use cache without original images

  # === Key (prefer env var) ===
  # encryption_key: "64-char-hex"  # Only if env var not used
```

```shell
# Preferred: Set key via environment variable
export YOLO_ENCRYPTION_KEY="your-64-char-hex-key"
```

##### LMDB Sync Option

By default, LMDB filesystem sync (`fsync`) is **disabled** for cache creation. This ensures compatibility with external volumes on macOS which may return `mdb_env_sync: Permission denied`.

**Default behavior (sync disabled):**
- Works on all volumes including external drives (USB, network mounts)
- Faster cache creation
- If system crashes during creation, cache may be incomplete (just delete and recreate)

**Enable sync for crash safety:**

```shell
# Enable fsync for internal drives where crash safety is important
yolo cache-create --data.root dataset/ --data.format yolo --size 640 --sync
```

| Flag | Behavior | Use Case |
|------|----------|----------|
| (default) | `sync=False` | External volumes, faster creation |
| `--sync` | `sync=True` | Internal drives, crash safety important |

**Via YAML (for training):**

```yaml
data:
  cache_images: disk
  cache_sync: true  # Enable fsync during training cache creation
```

```shell
# Or via CLI during training
yolo fit --config config.yaml --data.cache_sync=true
```

**Note:** Disabling sync is safe for cache data because it's derived from original images and can always be regenerated.

### Data Fraction (Quick Testing)

Use `data_fraction` to train/validate on a subset of your dataset. This is useful for:
- Quick experiments to find optimal `batch_size` and `num_workers`
- Debugging training pipelines
- Rapid prototyping of augmentation strategies

**Stratified Sampling:** The parameter uses stratified sampling by primary class (the first label in each annotation file), ensuring each class is proportionally represented in the subset.

**Configuration:**

```yaml
data:
  data_fraction: 1.0           # 1.0 = all data (default), 0.1 = 10% per class
```

**CLI override:**

```shell
# Use 10% of data for quick testing
python -m yolo.cli fit --config config.yaml --data.data_fraction=0.1

# Use 50% of data
python -m yolo.cli fit --config config.yaml --data.data_fraction=0.5
```

**Example: Finding Optimal DataLoader Settings**

```shell
# Quick 10% test to find best num_workers
for workers in 0 2 4 8 12; do
    python -m yolo.cli fit --config config.yaml \
        --data.data_fraction=0.1 \
        --data.num_workers=$workers \
        --trainer.max_epochs=1
done
```

### Exponential Moving Average (EMA)

EMA maintains a shadow copy of model weights that is updated with exponential moving average at each training step. The EMA model typically achieves better accuracy than the final training weights.

**How it works:**
- Shadow weights are updated each step: `ema = decay * ema + (1 - decay) * model`
- Decay ramps up during warmup: `effective_decay = decay * (1 - exp(-updates / tau))`
- EMA weights are used for validation and saved checkpoints

**Configuration:**

```yaml
trainer:
  callbacks:
    - class_path: yolo.training.callbacks.EMACallback
      init_args:
        decay: 0.9999     # EMA decay rate (higher = slower update)
        tau: 2000         # Warmup steps for decay
        enabled: true     # Set to false to disable EMA
```

**CLI override:**

```shell
# Disable EMA
python -m yolo.cli fit --config config.yaml \
    --trainer.callbacks.6.init_args.enabled=false

# Adjust decay rate
python -m yolo.cli fit --config config.yaml \
    --trainer.callbacks.6.init_args.decay=0.999
```

### Optimizer Selection

Choose between SGD (default, recommended for detection) and AdamW optimizers.

**Configuration:**

```yaml
model:
  optimizer: sgd          # "sgd" or "adamw"

  # SGD parameters
  learning_rate: 0.01
  momentum: 0.937
  weight_decay: 0.0005

  # AdamW parameters (only used if optimizer: adamw)
  adamw_betas: [0.9, 0.999]
```

**CLI override:**

```shell
# Use AdamW optimizer
python -m yolo.cli fit --config config.yaml --model.optimizer=adamw

# Use AdamW with custom betas
python -m yolo.cli fit --config config.yaml \
    --model.optimizer=adamw \
    --model.adamw_betas="[0.9, 0.99]"
```

### Recommended Configurations

#### Standard Training (COCO-like datasets)

```yaml
data:
  mosaic_prob: 1.0
  mixup_prob: 0.15
  cutmix_prob: 0.0
  close_mosaic_epochs: 15
model:
  optimizer: sgd
trainer:
  callbacks:
    - class_path: yolo.training.callbacks.EMACallback
      init_args:
        enabled: true
```

#### Small Dataset (< 1000 images)

```yaml
data:
  mosaic_prob: 1.0
  mixup_prob: 0.3         # Higher mixup for regularization
  cutmix_prob: 0.1        # Add cutmix
  degrees: 10.0           # Add rotation
  close_mosaic_epochs: 10
```

#### Fast Training (reduced augmentation)

```yaml
data:
  mosaic_prob: 0.5        # 50% mosaic
  mixup_prob: 0.0         # No mixup
  close_mosaic_epochs: 5
trainer:
  callbacks:
    - class_path: yolo.training.callbacks.EMACallback
      init_args:
        enabled: false    # Disable EMA for speed
```

#### Fine-tuning (minimal augmentation)

```yaml
data:
  mosaic_prob: 0.0        # No mosaic
  mixup_prob: 0.0         # No mixup
  flip_lr: 0.5            # Keep basic flip
  close_mosaic_epochs: 0
```

## Configuration

All configuration via YAML files in `yolo/config/experiment/`:

```yaml
trainer:
  max_epochs: 500
  accelerator: auto
  devices: auto
  precision: 16-mixed

model:
  model_config: v9-c
  num_classes: 80
  learning_rate: 0.01

  # NMS settings
  nms_conf_threshold: 0.25       # For inference
  nms_val_conf_threshold: 0.001  # For validation (low to capture all predictions for mAP)
  nms_iou_threshold: 0.65
  nms_max_detections: 300

data:
  root: data/coco
  batch_size: 16
  image_size: [640, 640]
```

See [HOWTO](docs/HOWTO.md) for detailed documentation and [Training Guide](training-experiment/TRAINING_GUIDE.md) for a complete training example.

### Dataset Formats

The training pipeline supports two dataset formats: **COCO** (default) and **YOLO**.

The format is configured via the `data.format` parameter in your YAML config file:

```yaml
data:
  format: coco   # or 'yolo'
```

You can also override via CLI:
```shell
python -m yolo.cli fit --config config.yaml --data.format=yolo
```

#### COCO Format (Default)

Standard COCO JSON annotation format. Used by default when `format: coco` or not specified.

**Directory structure:**
```
dataset/
├── images/
│   ├── train/
│   │   └── *.jpg
│   └── val/
│       └── *.jpg
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

**Configuration:**
```yaml
data:
  format: coco
  root: path/to/dataset
  train_images: images/train
  val_images: images/val
  train_ann: annotations/instances_train.json
  val_ann: annotations/instances_val.json
  batch_size: 16
  image_size: [640, 640]
```

#### YOLO Format

Standard YOLO `.txt` annotation format with normalized coordinates. Use `format: yolo` in your config.

**Directory structure:**
```
dataset/
├── train/
│   ├── images/
│   │   └── *.jpg
│   └── labels/
│       └── *.txt
└── valid/
    ├── images/
    │   └── *.jpg
    └── labels/
        └── *.txt
```

**Label file format** (one file per image, same name as image):
```
class_id x_center y_center width height
class_id x_center y_center width height
...
```
All coordinates are **normalized** (0-1 range).

**Configuration:**
```yaml
data:
  format: yolo
  root: path/to/dataset
  train_images: train/images
  train_labels: train/labels
  val_images: valid/images
  val_labels: valid/labels
  batch_size: 16
  image_size: [640, 640]
```

#### YOLO Format with Split Files

For datasets with a **single images/labels folder** and separate split files (e.g., `train.txt`, `val.txt`), you can use the `train_split` and `val_split` parameters.

**Directory structure:**
```
dataset/
├── images/           # All images in one folder
│   └── *.jpg
├── labels/           # All labels in one folder
│   └── *.txt
├── train.txt         # List of training image filenames
└── val.txt           # List of validation image filenames
```

**Split file format** (one filename per line):
```
img_001.jpg
img_002.jpg
img_003.jpg
...
```

**Configuration:**
```yaml
data:
  format: yolo
  root: path/to/dataset
  train_images: images      # Single images folder
  val_images: images        # Same folder
  train_labels: labels      # Single labels folder
  val_labels: labels        # Same folder
  train_split: train.txt    # Filter for training
  val_split: val.txt        # Filter for validation
  batch_size: 16
  image_size: [640, 640]
```

This is useful when:
- You have a pre-existing dataset with split files
- You want to use the same images/labels folders for both train and val
- You're using a cached dataset where all images are in a single cache

**Note:** Split files work with both regular and cache-only modes. When using `cache_only: true`, the split files filter the cached images.

#### Example Configurations

See the training experiment for complete examples:

| Format | Config File | Dataset |
|--------|-------------|---------|
| **COCO** | [simpsons-train.yaml](training-experiment/simpsons-train.yaml) | `simpsons-coco-std/` |
| **YOLO** | [simpsons-yolo-train.yaml](training-experiment/simpsons-yolo-train.yaml) | `simpsons-yolo/` |

**Quick start:**
```shell
# Train with COCO format
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml

# Train with YOLO format
python -m yolo.cli fit --config training-experiment/simpsons-yolo-train.yaml
```

## Testing

The project includes a comprehensive test suite to ensure correctness of all components.

### Running Tests

```shell
# Run all tests (excluding integration tests)
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_augmentations.py -v

# Run with coverage
python -m pytest tests/ --cov=yolo --cov-report=html

# Run integration tests (require additional setup/datasets)
python -m pytest tests/ -v --run-integration
```

### Test Coverage

| Module | Tests | Description |
|--------|-------|-------------|
| **Image Loaders** | 45 tests | DefaultImageLoader, FastImageLoader, TurboJPEGLoader, EncryptedImageLoader, file handle leaks, stress tests |
| **Augmentations** | 44 tests | Mosaic4/9, MixUp, CutMix, RandomPerspective, EMA |
| **Metrics** | 33 tests | IoU, AP computation, confusion matrix, DetMetrics, plot generation |
| **Eval Dashboard** | 26 tests | Dashboard rendering, trends, sparklines, sections |
| **Validate** | 16 tests | Benchmark, device detection, metrics integration |
| **Schedulers** | 15 tests | Cosine, linear, step, OneCycle creation and behavior |
| **Layer Freezing** | 14 tests | Backbone freezing, pattern matching, epoch-based unfreezing |
| **Model Building** | 10 tests | v9-c, v9-m, v9-s, v7 models, forward pass |
| **Bounding Box Utils** | 13 tests | IoU, transforms, NMS, anchor generation |
| **Export** | 12 tests | Letterbox, ONNX/TFLite signatures, CLI options |
| **Training Experiment** | 16 tests | Dataset loading, metrics, schedulers, freezing, export |
| **YOLO Format Dataloader** | 36 tests | Dataset loading, transforms, collate, prewarm, edge cases |
| **Cache** | 50 tests | Label caching, image caching (RAM/disk), mmap, encryption, LRU buffer |
| **Progress** | 26 tests | Spinner, progress bar, ProgressTracker, Rich console integration |
| **Integration** | 10 tests | Full pipeline tests (run with `--run-integration`) |

**Total: 405 tests** covering image loaders, data augmentation, training callbacks, metrics, eval dashboard, schedulers, layer freezing, model components, export, validate, caching, progress indicators, and utilities.

### Training Experiment Tests

All features have been validated on the [Simpsons Character Detection](training-experiment/TRAINING_GUIDE.md) dataset:

```bash
python -m pytest tests/test_training_experiment.py -v
# Result: 16 passed (dataset loading, metrics, schedulers, freezing, export)
```

| Feature | Status |
|---------|--------|
| COCO Dataset Loading | ✅ Tested |
| Metrics System (7 classes) | ✅ Tested |
| LR Schedulers | ✅ Tested |
| Layer Freezing | ✅ Tested |
| Model Export (ONNX/TFLite) | ✅ Tested |

## Metrics

The training pipeline includes a comprehensive detection metrics system with automatic plot generation.

### Available Metrics

All validation metrics are automatically logged to TensorBoard and other loggers:

| Metric | Description |
|--------|-------------|
| **val/mAP** | mAP @ IoU=0.50:0.95 (COCO primary metric) |
| **val/mAP50** | mAP @ IoU=0.50 |
| **val/mAP75** | mAP @ IoU=0.75 |
| **val/precision** | Mean precision across all classes |
| **val/recall** | Mean recall across all classes |
| **val/f1** | Mean F1 score across all classes |
| **val/AR@100** | Average Recall with max 100 detections |

### Metrics Plots

When `save_metrics_plots: true` (default), the following plots are automatically generated for each validation epoch:

- **PR_curve.png**: Precision-Recall curve per class
- **F1_curve.png**: F1 vs. Confidence threshold curve
- **P_curve.png**: Precision vs. Confidence curve
- **R_curve.png**: Recall vs. Confidence curve
- **confusion_matrix.png**: Confusion matrix with class predictions

Plots are saved to `runs/<experiment>/metrics/epoch_<N>/`.

### Configuration Example

```yaml
model:
  # Metrics plots
  save_metrics_plots: true
  metrics_plots_dir: null  # Auto: runs/<experiment>/metrics/
```

### CLI Override Examples

```shell
# Disable metrics plots for faster validation
python -m yolo.cli fit --config config.yaml \
    --model.save_metrics_plots=false

# Custom metrics plots directory
python -m yolo.cli fit --config config.yaml \
    --model.metrics_plots_dir=outputs/metrics
```

### Eval Dashboard

During training and validation, a comprehensive dashboard displays all metrics using Rich tables with clean formatting:

```
                            EVAL SUMMARY
╭─────────────────────────────┬───────────────────────┬──────────────────────────────────╮
│ Info                        │ mAP                   │ Settings                         │
├─────────────────────────────┼───────────────────────┼──────────────────────────────────┤
│ epoch: 25/100  imgs: 1200   │ 0.4523 [NEW BEST]     │ conf: 0.25  iou: 0.65  max: 300  │
│ size: 640x640               │                       │                                  │
╰─────────────────────────────┴───────────────────────┴──────────────────────────────────╯

                              KPI (QUALITY)
╭──────────┬────────┬────────┬─────────┬────────┬────────┬────────╮
│ AP50-95  │   AP50 │   AP75 │  AR@100 │    APs │    APm │    APl │
├──────────┼────────┼────────┼─────────┼────────┼────────┼────────┤
│   0.4523 │ 0.6821 │ 0.4912 │  0.5234 │ 0.2134 │ 0.4521 │ 0.5823 │
╰──────────┴────────┴────────┴─────────┴────────┴────────┴────────╯

                          KPI (OPERATIVE @ 0.25)
╭─────────┬─────────┬──────────┬─────────┬───────────╮
│  P@0.25 │  R@0.25 │  F1@0.25 │ best-F1 │ conf_best │
├─────────┼─────────┼──────────┼─────────┼───────────┤
│  0.7823 │  0.6234 │   0.6941 │  0.7123 │      0.32 │
╰─────────┴─────────┴──────────┴─────────┴───────────╯

                       TRENDS (last 10 epochs)
╭─────────┬────────────┬──────────────────────╮
│ Metric  │ Trend      │ Range                │
├─────────┼────────────┼──────────────────────┤
│ AP50-95 │ _.,~'^.~'^ │ min: 0.32  max: 0.45 │
│ AP50    │ .~'^.~'^.^ │ min: 0.58  max: 0.68 │
│ AR@100  │ _.,~'.~'^. │ min: 0.42  max: 0.52 │
╰─────────┴────────────┴──────────────────────╯

                        THRESHOLD SWEEP
╭──────┬──────┬──────┬──────╮
│ conf │    P │    R │   F1 │
├──────┼──────┼──────┼──────┤
│ 0.10 │ 0.52 │ 0.89 │ 0.66 │
│ 0.20 │ 0.68 │ 0.72 │ 0.70 │
│ 0.30 │ 0.78 │ 0.61 │ 0.69 │
│ 0.40 │ 0.85 │ 0.48 │ 0.61 │
│ 0.50 │ 0.91 │ 0.34 │ 0.50 │
╰──────┴──────┴──────┴──────╯

                    PER-CLASS: TOP by AP50-95
╭────────┬──────────┬────────┬────────┬─────╮
│ Class  │ AP50-95  │   AP50 │ R@conf │  GT │
├────────┼──────────┼────────┼────────┼─────┤
│ homer  │   0.6234 │ 0.8521 │ 0.7823 │ 342 │
│ bart   │   0.5821 │ 0.7934 │ 0.7234 │ 289 │
│ marge  │   0.5234 │ 0.7621 │ 0.6821 │ 256 │
╰────────┴──────────┴────────┴────────┴─────╯

                   PER-CLASS: WORST by AP50-95
╭─────────┬──────────┬────────┬────────┬────╮
│ Class   │ AP50-95  │   AP50 │ R@conf │ GT │
├─────────┼──────────┼────────┼────────┼────┤
│ maggie  │   0.2134 │ 0.4523 │ 0.3421 │ 45 │
│ abraham │   0.2821 │ 0.5234 │ 0.4123 │ 67 │
│ ned     │   0.3421 │ 0.5821 │ 0.4821 │ 89 │
╰─────────┴──────────┴────────┴────────┴────╯

                       ERROR HEALTH CHECK
╭──────┬──────┬──────────────┬─────────────╮
│   FP │   FN │ det/img mean │ det/img p95 │
├──────┼──────┼──────────────┼─────────────┤
│  234 │  156 │          8.3 │          15 │
╰──────┴──────┴──────────────┴─────────────╯

                   Top Confusions (pred → true)
╭───┬───────────┬───┬───────┬───────╮
│ # │ Predicted │ → │ True  │ Count │
├───┼───────────┼───┼───────┼───────┤
│ 1 │ bart      │ → │ lisa  │    23 │
│ 2 │ homer     │ → │ abraham │  12 │
│ 3 │ marge     │ → │ lisa  │     8 │
╰───┴───────────┴───┴───────┴───────╯
```

The dashboard shows a **[NEW BEST]** indicator when the model achieves a new best mAP, or displays a delta **(+0.0050)** in green for improvements or **(-0.0123)** in red when worse than the best.

#### Dashboard Metrics

**Quality Metrics (COCO Standard):**

| Metric | Description |
|--------|-------------|
| **AP50-95** | mAP @ IoU=0.50:0.95 (primary COCO metric) |
| **AP50** | mAP @ IoU=0.50 |
| **AP75** | mAP @ IoU=0.75 |
| **AR@100** | Average Recall @ 100 detections/image |
| **APs/APm/APl** | AP for small/medium/large objects |

**Operative Metrics (at production threshold):**

| Metric | Description |
|--------|-------------|
| **P@conf** | Precision at production confidence |
| **R@conf** | Recall at production confidence |
| **F1@conf** | F1 at production confidence |
| **best-F1** | Best achievable F1 score |
| **conf_best** | Confidence threshold for best F1 |

**Error Metrics:**

| Metric | Description |
|--------|-------------|
| **FP/FN** | Total False Positives and False Negatives |
| **det/img** | Detections per image (mean, p95) |
| **MeanIoU(TP)** | Mean IoU of True Positives |
| **Top confusions** | Most confused class pairs |

#### Dashboard Configuration

```yaml
trainer:
  callbacks:
    - class_path: yolo.training.callbacks.EvalDashboardCallback
      init_args:
        conf_prod: 0.25        # Production confidence threshold
        show_trends: true      # Show sparkline trends
        top_n_classes: 3       # N classes for TOP/WORST
```

## Learning Rate Schedulers

The training pipeline supports multiple learning rate schedulers for different training strategies.

### Available Schedulers

| Scheduler | Description | Best For |
|-----------|-------------|----------|
| **cosine** | Cosine annealing (default) | General training |
| **linear** | Linear decay to minimum LR | Simple baseline |
| **step** | Step decay every N epochs | Classic approach |
| **one_cycle** | One cycle policy | Fast convergence |

### Configuration

```yaml
model:
  lr_scheduler: cosine  # Options: cosine, linear, step, one_cycle

  # Common parameters
  lr_min_factor: 0.01   # final_lr = initial_lr * lr_min_factor

  # Step scheduler parameters
  step_size: 30         # Epochs between LR decay
  step_gamma: 0.1       # LR multiplication factor

  # OneCycle scheduler parameters
  one_cycle_pct_start: 0.3        # Warmup fraction
  one_cycle_div_factor: 25.0      # initial_lr = max_lr / div_factor
  one_cycle_final_div_factor: 10000.0  # final_lr = initial_lr / final_div_factor
```

### CLI Examples

```shell
# Cosine annealing (default)
python -m yolo.cli fit --config config.yaml --model.lr_scheduler=cosine

# OneCycle for faster convergence
python -m yolo.cli fit --config config.yaml \
    --model.lr_scheduler=one_cycle \
    --model.one_cycle_pct_start=0.3

# Step decay (classic approach)
python -m yolo.cli fit --config config.yaml \
    --model.lr_scheduler=step \
    --model.step_size=30 \
    --model.step_gamma=0.1
```

## Layer Freezing (Transfer Learning)

Freeze layers for efficient transfer learning on custom datasets. This is especially useful when fine-tuning on small datasets.

### Configuration

```yaml
model:
  # Load pretrained weights
  weight_path: true  # Auto-download based on model_config

  # Freeze entire backbone
  freeze_backbone: true
  freeze_until_epoch: 10  # Unfreeze after epoch 10 (0 = always frozen)

  # Or freeze specific layers by name pattern
  freeze_layers:
    - backbone_conv1
    - stem
```

### CLI Examples

```shell
# Freeze backbone for first 10 epochs, then fine-tune all layers
python -m yolo.cli fit --config config.yaml \
    --model.weight_path=true \
    --model.freeze_backbone=true \
    --model.freeze_until_epoch=10

# Freeze specific layers permanently
python -m yolo.cli fit --config config.yaml \
    --model.weight_path=true \
    --model.freeze_layers='[backbone_conv1, backbone_conv2]'
```

### Transfer Learning Workflow

1. **Load pretrained weights**: `--model.weight_path=true`
2. **Freeze backbone**: `--model.freeze_backbone=true`
3. **Train head for N epochs**: `--model.freeze_until_epoch=10`
4. **Unfreeze and fine-tune**: Automatic after `freeze_until_epoch`

This approach allows the detection head to adapt to your custom classes first, then fine-tunes the entire network with a lower learning rate.

## Checkpoints

During training, model checkpoints are automatically saved to the configured directory.

### Saved Files

| File | Description |
|------|-------------|
| `best.ckpt` | Best model (highest val/mAP) - always updated |
| `last.ckpt` | Latest model (end of last epoch) |
| `{name}-epoch=XX-mAP=X.XXXX.ckpt` | Top-K best models with metrics in filename |

### Default Location

```
training-experiment/checkpoints/
├── best.ckpt                            # Best model (use this for inference)
├── last.ckpt                            # Latest checkpoint
├── simpsons-epoch=05-mAP=0.4523.ckpt    # Top-3 checkpoints with metrics
├── simpsons-epoch=08-mAP=0.4891.ckpt
└── simpsons-epoch=12-mAP=0.5102.ckpt
```

### Configuration

```yaml
trainer:
  callbacks:
    # Save top-K models with detailed filenames
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: checkpoints/
        monitor: val/mAP
        mode: max
        save_top_k: 3
        save_last: true
        filename: "{name}-{epoch:02d}-mAP={val/mAP:.4f}"

    # Save best model with fixed name
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: checkpoints/
        monitor: val/mAP
        mode: max
        save_top_k: 1
        filename: "best"
```

### Resume Training

```shell
# Resume from last checkpoint
python -m yolo.cli fit --config config.yaml \
    --ckpt_path=checkpoints/last.ckpt

# Resume from best checkpoint
python -m yolo.cli fit --config config.yaml \
    --ckpt_path=checkpoints/best.ckpt

# Resume and extend training to more epochs
python -m yolo.cli fit --config config.yaml \
    --ckpt_path=checkpoints/last.ckpt \
    --trainer.max_epochs=500
```

When resuming, the training state is fully restored including:
- Model weights and optimizer state
- Learning rate scheduler position
- Current epoch and global step
- Best metric values for checkpointing

## Early Stopping

Early stopping automatically halts training when the validation metric stops improving, preventing overfitting and saving compute time.

### Configuration

```yaml
trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: val/mAP          # Metric to monitor
        mode: max                 # 'max' for mAP (higher is better)
        patience: 50              # Epochs to wait before stopping
        verbose: true             # Log when stopping
        min_delta: 0.0            # Minimum improvement threshold
        check_on_train_epoch_end: false  # Check after validation (required for resume)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `monitor` | `val/mAP` | Metric to monitor for improvement |
| `mode` | `max` | `max` = higher is better, `min` = lower is better |
| `patience` | 50 | Number of epochs with no improvement before stopping |
| `min_delta` | 0.0 | Minimum change to qualify as improvement |
| `verbose` | true | Print message when stopping |
| `check_on_train_epoch_end` | false | When to check (false = after validation) |

### CLI Override

```shell
# Increase patience for longer training
python -m yolo.cli fit --config config.yaml \
    --trainer.callbacks.1.init_args.patience=100

# Disable early stopping (train for full max_epochs)
# Simply remove EarlyStopping from callbacks in your YAML config
```

### Example Output

When early stopping triggers:
```
Epoch 45: val/mAP did not improve. Best: 0.5234 @ epoch 35
EarlyStopping: Stopping training at epoch 85
```

### Tips

- **For small datasets**: Use higher patience (30-50) as metrics can be noisy
- **For large datasets**: Lower patience (10-20) is usually sufficient
- **For fine-tuning**: Consider disabling early stopping to train for a fixed number of epochs
- **Resume compatibility**: Always use `check_on_train_epoch_end: false` to avoid errors when resuming from checkpoints

## Performance

MS COCO Object Detection

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | Params | FLOPs | Weights |
|:------|:---------:|:----------------:|:-----------------------------:|:-----------------------------:|:------:|:-----:|:-------:|
| YOLOv9-T | 640 | **38.3%** | 53.1% | 41.3% | 2.0M | 7.7G | [download](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt) |
| YOLOv9-S | 640 | **46.8%** | 63.4% | 50.7% | 7.1M | 26.4G | [download](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s-converted.pt) |
| YOLOv9-M | 640 | **51.4%** | 68.1% | 56.1% | 20.0M | 76.3G | [download](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m-converted.pt) |
| YOLOv9-C | 640 | **53.0%** | 70.2% | 57.8% | 25.3M | 102.1G | [download](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt) |
| YOLOv9-E | 640 | **55.6%** | 72.8% | 60.6% | 57.3M | 189.0G | [download](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt) |

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | Params | FLOPs | Weights |
|:------|:---------:|:----------------:|:-----------------------------:|:-----------------------------:|:------:|:-----:|:-------:|
| GELAN-S | 640 | **46.7%** | 63.3% | 50.8% | 7.1M | 26.4G | [download](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-s.pt) |
| GELAN-M | 640 | **51.1%** | 68.0% | 55.8% | 20.0M | 76.3G | [download](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-m.pt) |
| GELAN-C | 640 | **52.3%** | 69.8% | 57.1% | 25.3M | 102.1G | [download](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt) |
| GELAN-E | 640 | **53.7%** | 71.0% | 58.5% | 57.3M | 189.0G | [download](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt) |

## Task-Aligned Learning (TAL)

This framework implements **Task-Aligned Learning** for anchor-target assignment, matching the yolov9-official implementation for optimal detection performance.

### How It Works

1. **Alignment Metric**: For each anchor-target pair, compute alignment score:
   ```
   align = cls_score^α × IoU^β
   ```
   where `α=0.5` (classification weight) and `β=6.0` (localization weight)

2. **TopK Selection**: Select the top-K anchors with highest alignment scores for each target (default K=13)

3. **Soft Targets**: Normalize alignment scores to create soft classification targets:
   ```
   soft_target = align / max_align_per_target
   ```

4. **Loss Normalization**: Scale gradients by `cls_norm = sum(soft_targets)` for stable training

### Configuration

```yaml
model:
  # Loss weights (aligned with yolov9-official)
  box_loss_weight: 7.5
  cls_loss_weight: 0.5
  dfl_loss_weight: 1.5

# Matcher parameters (in loss.py defaults)
# matcher_topk: 13      # Anchors per target
# matcher_alpha: 0.5    # Classification weight
# matcher_beta: 6.0     # IoU weight
```

### Why Task-Aligned Learning?

| Approach | Description | mAP Impact |
|----------|-------------|------------|
| **Fixed Assignment** | One anchor per target | Lower recall |
| **IoU-based** | Assign by IoU only | Misses good classification anchors |
| **Task-Aligned** | Balance cls + IoU | Best mAP, matches detection task |

## Loss Functions

The framework uses three loss components, each optimized for detection:

### Box Loss (CIoU)

Complete IoU loss that considers overlap, distance, and aspect ratio:

```
CIoU = 1 - IoU + distance²/diagonal² + αv
```

- **Weight**: `box_loss_weight: 7.5`
- **Purpose**: Precise bounding box regression

### Classification Loss (BCE)

Binary Cross-Entropy with **soft targets** from Task-Aligned Learning:

```
BCE = -Σ (soft_target × log(pred) + (1 - soft_target) × log(1 - pred))
```

- **Weight**: `cls_loss_weight: 0.5`
- **Normalization**: Divided by `cls_norm` (sum of soft targets)

### Distribution Focal Loss (DFL)

Predicts box coordinates as a discrete probability distribution over 16 bins:

```
DFL = -((ceil - x) × log(P_floor) + (x - floor) × log(P_ceil))
```

- **Weight**: `dfl_loss_weight: 1.5`
- **Purpose**: Sub-pixel coordinate precision

### Loss Configuration

```yaml
model:
  box_loss_weight: 7.5    # Box regression importance
  cls_loss_weight: 0.5    # Classification importance
  dfl_loss_weight: 1.5    # Coordinate distribution importance

  # Alternative classification losses
  cls_loss_type: bce      # Options: bce, vfl (Varifocal)
  cls_vfl_alpha: 0.75     # VFL alpha (if using vfl)
  cls_vfl_gamma: 2.0      # VFL gamma (if using vfl)
```

## Custom Callbacks

The framework includes specialized PyTorch Lightning callbacks for YOLO training:

| Callback | Description |
|----------|-------------|
| `EMACallback` | Exponential Moving Average with configurable decay and tau warmup |
| `YOLOProgressBar` | Rich progress bar with 1-indexed epochs and loss components |
| `TrainingSummaryCallback` | End-of-epoch summary with best model tracking |
| `EvalDashboardCallback` | Comprehensive metrics dashboard with trends and threshold sweep |

### EMACallback

Maintains a smoothed copy of model weights that often achieves better accuracy:

```yaml
trainer:
  callbacks:
    - class_path: yolo.training.callbacks.EMACallback
      init_args:
        decay: 0.9999       # EMA decay factor
        tau: 2000.0         # Warmup steps (ramps decay from 0.5 to target)
        enabled: true       # Set false to disable
```

### EvalDashboardCallback

Rich terminal dashboard showing comprehensive metrics during validation:

```yaml
trainer:
  callbacks:
    - class_path: yolo.training.callbacks.EvalDashboardCallback
      init_args:
        conf_prod: 0.25     # Production confidence for operative metrics
        show_trends: true   # Show sparkline trends
        top_n_classes: 5    # Classes to show in per-class section
```

### YOLOProgressBar

Custom progress bar with 1-indexed epochs and compact metrics display:

```yaml
trainer:
  callbacks:
    - class_path: yolo.training.callbacks.YOLOProgressBar
```

## Troubleshooting

### Low mAP During Training

**Symptoms**: mAP stays below 50% even after many epochs

**Solutions**:
1. Ensure DataLoader shuffle is enabled (default: `True`)
2. Check that background/negative images are distributed across batches
3. Verify `matcher_topk >= 10` for sufficient anchor assignment
4. Use pretrained weights: `--model.weight_path=true`

### Loss Spikes

**Symptoms**: Sudden large increases in loss values

**Causes & Solutions**:
- **Early training**: Normal with soft targets, usually stabilizes
- **Batch with no targets**: Framework handles this automatically
- **Corrupted images**: Check dataset integrity

### CUDA Out of Memory

**Solutions**:
1. Reduce `batch_size`
2. Enable gradient accumulation: `--trainer.accumulate_grad_batches=2`
3. Use mixed precision: `--trainer.precision=16-mixed`
4. Reduce `image_size`

### Validation Takes Too Long

**Solutions**:
1. Reduce `nms_max_detections` (default: 300)
2. Increase `nms_val_conf_threshold` (default: 0.001)
3. Use smaller validation split for debugging

### Training Won't Resume

**Symptoms**: Resuming from checkpoint starts from epoch 0

**Solution**: Use `--ckpt_path` not `--model.weight_path`:
```shell
# Correct: Full training state restored
python -m yolo fit --config config.yaml --ckpt_path=last.ckpt

# Wrong: Only weights loaded, training restarts
python -m yolo fit --config config.yaml --model.weight_path=last.ckpt
```

## Project Structure

```
yolo/
├── cli.py                    # CLI entry point (train, predict, export)
├── config/
│   ├── experiment/           # Training configs (default.yaml, debug.yaml)
│   └── model/                # Model architectures (v9-c.yaml, v9-s.yaml, ...)
├── data/
│   ├── datamodule.py         # LightningDataModule with caching
│   ├── transforms.py         # Data augmentations (Mosaic, BBox Mosaic, etc.)
│   ├── loaders.py            # Image loaders (Default, Fast, TurboJPEG)
│   └── encrypted_loader.py   # AES-256 encrypted image loader
├── model/
│   ├── yolo.py               # Model builder from DSL
│   └── module.py             # Layer definitions
├── training/
│   ├── module.py             # LightningModule (training, schedulers, freezing)
│   ├── loss.py               # Loss functions (Box, Cls, DFL) with TAL
│   └── callbacks.py          # EMA, ProgressBar, Summary, EvalDashboard
├── tools/
│   ├── export.py             # Model export (ONNX, TFLite, SavedModel)
│   ├── qat.py                # Quantization-Aware Training
│   └── inference.py          # Inference utilities
└── utils/
    ├── bounding_box_utils.py # BBox utilities + BoxMatcher (TAL)
    ├── metrics.py            # COCO metrics, PR curves, confusion matrix
    ├── eval_dashboard.py     # Rich terminal metrics dashboard
    ├── progress.py           # Spinner and progress utilities
    └── logger.py             # Logging
```

## Citations

```bibtex
@inproceedings{wang2024yolov9,
    title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
    author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2024}
}

@inproceedings{wang2022yolov7,
    title={{YOLOv7}: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors},
    author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}

@inproceedings{tsui2024yolord,
    title={{YOLO-RD}: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary},
    author={Tsui, Hao-Tang and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
    booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2025}
}
```

## License

This project is released under the [MIT License](LICENSE).

---

[^1]: [YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)
[^2]: [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
[^3]: [YOLO-RD: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary](https://arxiv.org/abs/2410.15346)
