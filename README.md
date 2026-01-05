# YOLO: Training CLI for YOLOv9

![GitHub License](https://img.shields.io/github/license/WongKinYiu/YOLO)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-792ee5.svg)](https://lightning.ai/)
[![Tests](https://img.shields.io/badge/tests-346%20passed-brightgreen.svg)](tests/)

> **Fork Notice**: This is a fork of [WongKinYiu/YOLO](https://github.com/WongKinYiu/YOLO) with extensive additions for production training.

This repository extends the official YOLOv7[^1], YOLOv9[^2], and YOLO-RD[^3] implementation with a **robust CLI for training, validation, and export**.

## What's Different From the Original?

The original repository focuses on model architecture and research contributions. This fork adds:

- **Complete training CLI** with YAML configuration and command-line overrides
- **Data augmentation suite** - Mosaic 4/9, MixUp, CutMix, RandomPerspective, HSV
- **Multiple dataset formats** - COCO JSON and YOLO TXT with caching
- **Comprehensive metrics** - Full COCO evaluation, confusion matrix, PR curves
- **Rich monitoring** - Progress bars, eval dashboard, TensorBoard integration
- **Model export** - ONNX, TFLite (FP32/FP16/INT8), SavedModel

All additions are built on **PyTorch Lightning** for clean, scalable training.

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
| **Image Caching** | Load images to RAM (`ram`) for fastest training or cache decoded images to disk (`disk`) for moderate speedup |
| **Cache Resize** | Resize images to `image_size` during caching for reduced RAM usage (default: enabled) |
| **Data Fraction** | Stratified sampling to use a fraction of data for quick testing (e.g., `data_fraction: 0.1` for 10%) |
| **Custom Loaders** | Plug in custom image loaders for encrypted datasets, cloud storage, or proprietary formats |
| **Pin Memory** | Pre-load batches to pinned (page-locked) memory for faster CPU-to-GPU transfer |

### Inference & NMS
| Feature | Description |
|---------|-------------|
| **Configurable NMS** | Tune confidence threshold, IoU threshold, and max detections per image to balance precision vs recall |
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

# Export with custom output path
python -m yolo.cli export --checkpoint runs/best.ckpt --output model.onnx

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

When using expensive custom loaders (e.g., decryption), you may want to cache processed images to avoid re-processing every epoch.

**Important**: With `num_workers > 0`, each worker runs in a separate process with isolated memory. A simple in-memory cache won't work across workers.

**Solution**: Use `multiprocessing.Manager().dict()` for a shared cache:

```python
import io
import pickle
from multiprocessing import Manager
from PIL import Image
from yolo.data.loaders import ImageLoader

class CachedEncryptedImageLoader(ImageLoader):
    """Encrypted image loader with shared cache across workers."""

    def __init__(self, key: str):
        self.key = key
        self._manager = Manager()
        self._cache = self._manager.dict()  # Shared across all workers

    def __call__(self, path: str) -> Image.Image:
        # Check cache first
        if path in self._cache:
            return pickle.loads(self._cache[path])

        # Decrypt and load image
        with open(path, 'rb') as f:
            encrypted_data = f.read()
        decrypted_data = my_decrypt(encrypted_data, self.key)
        image = Image.open(io.BytesIO(decrypted_data)).convert("RGB")

        # Store in shared cache (serialized)
        self._cache[path] = pickle.dumps(image)
        return image
```

**How it works:**
- First epoch: Each image is decrypted once and cached
- Subsequent epochs: Images are loaded from shared cache (no decryption)
- All workers share the same cache via inter-process communication

**Trade-offs:**
- Memory usage: Cache grows with dataset size
- Serialization overhead: pickle adds some latency
- Best for: Expensive operations (decryption, cloud storage, custom formats)

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

### Dataset Caching

Accelerate data loading with label caching and optional image caching. Labels are parsed once and cached; subsequent runs load instantly. The cache auto-invalidates when source files change.

**Label Caching (Default: Enabled)**

Labels are parsed from `.txt` files once and saved to a `.cache` file. On subsequent runs, labels load from cache instead of re-parsing all files. The cache validates against file modification times and sizes.

**Image Caching (Optional)**

| Mode | Description |
|------|-------------|
| `none` | No image caching (default) |
| `ram` | Load all images to RAM (fastest, high memory usage) |
| `disk` | Save decoded images as `.npy` files (moderate speedup, persistent) |

**Configuration:**

```yaml
data:
  cache_labels: true           # Enable label caching (default: true)
  cache_images: none           # "none", "ram", or "disk"
  cache_resize_images: true    # Resize images to image_size when caching (saves RAM)
  cache_max_memory_gb: 8.0     # Max RAM for image caching
  cache_refresh: false         # Force cache regeneration
```

**Image Resize During Caching:**

When `cache_resize_images: true` (default), images are resized to `image_size` during RAM caching. This significantly reduces memory usage - a 4K image (4000x3000) takes ~36MB in RAM, but resized to 640x640 only ~1.2MB. The resize uses letterbox padding to preserve aspect ratio.

**CLI override:**

```shell
# Disable label caching
python -m yolo.cli fit --config config.yaml --data.cache_labels=false

# Enable RAM image caching
python -m yolo.cli fit --config config.yaml --data.cache_images=ram

# Disable image resize during caching (use original resolution)
python -m yolo.cli fit --config config.yaml --data.cache_resize_images=false

# Force cache regeneration (delete and rebuild)
python -m yolo.cli fit --config config.yaml --data.cache_refresh=true
```

**Force Cache Refresh:**

Use `--data.cache_refresh=true` to force deletion and regeneration of the cache. Useful when:
- Dataset files were modified but timestamps didn't change
- Cache file is corrupted
- Switching between different preprocessing configurations

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
| **YOLO Format Dataloader** | 30 tests | Dataset loading, transforms, collate, edge cases |
| **Cache** | 18 tests | Label caching, image caching, cache invalidation |
| **Integration** | 10 tests | Full pipeline tests (run with `--run-integration`) |

**Total: 346 tests** covering image loaders, data augmentation, training callbacks, metrics, eval dashboard, schedulers, layer freezing, model components, export, validate, caching, and utilities.

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

## Project Structure

```
yolo/
├── cli.py                    # CLI entry point (train, predict, export)
├── config/
│   ├── experiment/           # Training configs (default.yaml, debug.yaml)
│   └── model/                # Model architectures (v9-c.yaml, v9-s.yaml, ...)
├── data/
│   ├── datamodule.py         # LightningDataModule
│   └── transforms.py         # Data augmentations
├── model/
│   ├── yolo.py               # Model builder from DSL
│   └── module.py             # Layer definitions
├── training/
│   ├── module.py             # LightningModule (training, schedulers, freezing)
│   ├── loss.py               # Loss functions
│   └── callbacks.py          # Custom callbacks (EMA, metrics display)
├── tools/
│   ├── export.py             # Model export (ONNX, TFLite, SavedModel)
│   └── inference.py          # Inference utilities
└── utils/
    ├── bounding_box_utils.py # BBox utilities
    ├── metrics.py            # Detection metrics (mAP, P/R, confusion matrix)
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
