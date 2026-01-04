# YOLO: Official Implementation of YOLOv9, YOLOv7, YOLO-RD

[![Documentation Status](https://readthedocs.org/projects/yolo-docs/badge/?version=latest)](https://yolo-docs.readthedocs.io/en/latest/?badge=latest)
![GitHub License](https://img.shields.io/github/license/WongKinYiu/YOLO)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-792ee5.svg)](https://lightning.ai/)
[![Tests](https://img.shields.io/badge/tests-65%20passed-brightgreen.svg)](tests/)

Welcome to the official implementation of YOLOv7[^1], YOLOv9[^2], and YOLO-RD[^3].

## Why This Implementation?

**A world-class model deserves a world-class training pipeline.**

The original YOLO implementations focused on the model architecture and research contributions. This repository provides a **production-ready training pipeline** built on modern best practices:

- **PyTorch Lightning** - Clean, scalable training with automatic mixed precision, multi-GPU support, and comprehensive logging
- **Robust & Reproducible** - Deterministic training, proper validation metrics (COCO mAP), and checkpoint management
- **Simple Configuration** - YAML-based configs with CLI overrides, no complex framework dependencies
- **Standard Data Format** - Native COCO dataset support via `torchvision.datasets.CocoDetection`

The goal: **make training YOLO as reliable and straightforward as the model itself is powerful**.

## Papers

- [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- [**YOLOv7**: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)
- [**YOLO-RD**: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary](https://arxiv.org/abs/2410.15346)

## Installation

```shell
git clone https://github.com/WongKinYiu/YOLO.git
cd YOLO
pip install -r requirements.txt
```

## Quick Start

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

```shell
python -m yolo.cli validate --config yolo/config/experiment/default.yaml \
    --ckpt_path=runs/best.ckpt
```

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

Export trained models to ONNX format for deployment.

```shell
# Export to ONNX
python -m yolo.cli export --checkpoint runs/best.ckpt

# Export with custom output path
python -m yolo.cli export --checkpoint runs/best.ckpt --output model.onnx

# Export with FP16 (CUDA only)
python -m yolo.cli export --checkpoint runs/best.ckpt --half

# Export with dynamic batch size
python -m yolo.cli export --checkpoint runs/best.ckpt --dynamic-batch

# Export with ONNX simplification (requires onnx-simplifier)
python -m yolo.cli export --checkpoint runs/best.ckpt --simplify
```

#### Export Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint, -c` | required | Path to model checkpoint (.ckpt) |
| `--output, -o` | auto | Output path (.onnx) |
| `--size` | 640 | Input image size |
| `--opset` | 17 | ONNX opset version |
| `--simplify` | false | Simplify ONNX model (requires `pip install onnx-simplifier`) |
| `--dynamic-batch` | false | Enable dynamic batch size |
| `--half` | false | Export in FP16 (CUDA only) |
| `--device` | auto | Device (cuda/cpu) |

#### Convert to TFLite

To convert ONNX to TFLite, use [onnx2tf](https://github.com/PINTO0309/onnx2tf):

```shell
# Install onnx2tf
pip install onnx2tf tensorflow

# Convert ONNX to TFLite
onnx2tf -i model.onnx -o tflite_output

# The output directory will contain:
# - saved_model/          (TensorFlow SavedModel)
# - model_float32.tflite  (TFLite FP32)
# - model_float16.tflite  (TFLite FP16)
```

**Advanced onnx2tf options:**

```shell
# With INT8 quantization (requires calibration data)
onnx2tf -i model.onnx -o tflite_output -oiqt

# Specify output signature
onnx2tf -i model.onnx -o tflite_output -ois images:1,3,640,640

# For Edge TPU deployment
onnx2tf -i model.onnx -o tflite_output -oiqt -cind images calibration_data.npy
```

## Features

| Feature | Description |
|---------|-------------|
| **Multi-GPU Training** | Automatic DDP with `--trainer.devices=N` |
| **Mixed Precision** | FP16/BF16 training with `--trainer.precision=16-mixed` |
| **COCO Metrics** | mAP@0.5, mAP@0.5:0.95, per-size metrics |
| **Checkpointing** | Automatic best/last model saving (see [Checkpoints](#checkpoints)) |
| **Early Stopping** | Stop on validation plateau |
| **Logging** | TensorBoard, WandB support |
| **Custom Image Loader** | Support for encrypted or custom image formats |
| **Mosaic Augmentation** | 4-way and 9-way mosaic (see [Advanced Training](#advanced-training-techniques)) |
| **MixUp / CutMix** | Image blending augmentations |
| **EMA** | Exponential Moving Average of model weights |
| **Close Mosaic** | Disable augmentation for final N epochs |
| **Optimizer Selection** | SGD (default) or AdamW |

## Custom Image Loader

For datasets with special image formats (e.g., encrypted images), you can provide a custom image loader.

### Creating a Custom Loader

```python
# my_loaders.py
import io
from PIL import Image
from yolo.data.loaders import ImageLoader

class EncryptedImageLoader(ImageLoader):
    """Loader for AES-encrypted images."""

    def __init__(self, key: str):
        self.key = key

    def __call__(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            encrypted_data = f.read()

        # Your decryption logic here
        decrypted_data = my_decrypt(encrypted_data, self.key)
        return Image.open(io.BytesIO(decrypted_data)).convert("RGB")
```

### Configuration via YAML

```yaml
data:
  root: data/encrypted-dataset
  image_loader:
    class_path: my_loaders.EncryptedImageLoader
    init_args:
      key: "my-secret-key"
```

### Configuration via CLI

```shell
# Full configuration via CLI
python -m yolo.cli fit --config config.yaml \
    --data.image_loader.class_path=my_loaders.EncryptedImageLoader \
    --data.image_loader.init_args.key="my-secret-key"

# Override just the key (if loader is already in YAML)
python -m yolo.cli fit --config config.yaml \
    --data.image_loader.init_args.key="different-key"
```

### Notes

- Custom loaders must return a PIL Image in RGB mode
- The loader must be picklable for multi-worker data loading (`num_workers > 0`)
- When using a custom loader, a log message will confirm: `📷 Using custom image loader: EncryptedImageLoader`

## Data Loading Performance

Best practices for optimizing data loading performance during training.

### DataLoader Settings

```yaml
data:
  num_workers: 8      # Increase for faster loading (default: 8)
  pin_memory: true    # Faster GPU transfer (default: true)
  batch_size: 16      # Adjust based on GPU memory
```

**Guidelines:**
- `num_workers`: Set to number of CPU cores, or 2x for I/O-bound workloads
- `pin_memory`: Keep `true` for GPU training
- `batch_size`: Larger batches improve throughput but require more VRAM

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

MixUp blends two images together with a random weight, creating soft labels that improve model generalization.

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

CutMix cuts a rectangular region from one image and pastes it onto another, combining their labels.

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

## Testing

The project includes a comprehensive test suite to ensure correctness of all components.

### Running Tests

```shell
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_augmentations.py -v

# Run with coverage
python -m pytest tests/ --cov=yolo --cov-report=html
```

### Test Coverage

| Module | Tests | Description |
|--------|-------|-------------|
| **Mosaic4** | 4 tests | 4-way mosaic output shape, box bounds, center point range, empty boxes |
| **Mosaic9** | 3 tests | 9-way mosaic output shape, box bounds, empty boxes |
| **MixUp** | 3 tests | Output shape, box combination, Beta(32,32) distribution |
| **CutMix** | 4 tests | Output shape, bbox bounds, center-based sampling, IoA threshold |
| **RandomPerspective** | 7 tests | Transform disabled, output shape, box bounds, matrix composition, empty boxes |
| **EMA** | 8 tests | Initialization, decay formula, warmup, state serialization, gradients disabled |
| **EMACallback** | 3 tests | Initialization, disabled state |
| **Integration** | 4 tests | Full pipeline, transform chains, mosaic disable |
| **Edge Cases** | 4 tests | Single image, small images, non-square, reproducibility |
| **Model Modules** | 6 tests | Conv, Pool, ADown, CBLinear, SPPELAN |
| **Utils** | 12 tests | Auto-pad, activation functions, chunk division |

**Total: 65 tests** covering data augmentation, training callbacks, model components, and utilities.

## Metrics Configuration

The training pipeline supports comprehensive COCO-style evaluation metrics. All metrics are configurable via YAML or CLI.

### Available Metrics

| Metric | Config Key | Default | Description |
|--------|------------|---------|-------------|
| **mAP** | `log_map` | ✅ | mAP @ IoU=0.50:0.95 (COCO primary metric) |
| **mAP50** | `log_map_50` | ✅ | mAP @ IoU=0.50 |
| **mAP75** | `log_map_75` | ✅ | mAP @ IoU=0.75 |
| **mAP95** | `log_map_95` | ✅ | mAP @ IoU=0.95 (strict threshold) |
| **mAP per size** | `log_map_per_size` | ✅ | mAP for small/medium/large objects |
| **mAR100** | `log_mar_100` | ✅ | Mean Average Recall (max 100 detections) |
| **mAR per size** | `log_mar_per_size` | ✅ | mAR for small/medium/large objects |

**Size definitions (COCO standard):**
- Small: area < 32² pixels
- Medium: 32² ≤ area < 96² pixels
- Large: area ≥ 96² pixels

### Configuration Example

```yaml
model:
  # Enable/disable specific metrics
  log_map: true           # mAP @ 0.50:0.95
  log_map_50: true        # mAP @ 0.50
  log_map_75: true        # mAP @ 0.75
  log_map_95: true        # mAP @ 0.95
  log_map_per_size: true  # mAP_small, mAP_medium, mAP_large
  log_mar_100: true       # Mean Average Recall
  log_mar_per_size: true  # mAR_small, mAR_medium, mAR_large
```

### CLI Override Examples

```shell
# Disable per-size metrics for faster validation
python -m yolo.cli fit --config config.yaml \
    --model.log_map_per_size=false \
    --model.log_mar_per_size=false

# Only log essential metrics (mAP, mAP50)
python -m yolo.cli fit --config config.yaml \
    --model.log_map_75=false \
    --model.log_map_95=false \
    --model.log_map_per_size=false \
    --model.log_mar_100=false \
    --model.log_mar_per_size=false
```

### Validation Output

After each epoch, a formatted metrics table is displayed:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Epoch 10 - Validation Metrics                 │
├────────────┬────────────┬────────────┬────────────┬────────────┤
│    mAP     │   mAP50    │   mAP75    │   mAP95    │   mAR100   │
│   0.4523   │   0.6821   │   0.4912   │   0.2134   │   0.5234   │
├────────────┼────────────┼────────────┼────────────┼────────────┤
│   mAP_sm   │   mAP_md   │   mAP_lg   │    loss    │            │
│   0.2134   │   0.4521   │   0.5823   │   2.3456   │            │
├────────────┼────────────┼────────────┼────────────┼────────────┤
│   mAR_sm   │   mAR_md   │   mAR_lg   │            │            │
│   0.1823   │   0.4012   │   0.5412   │            │            │
└────────────┴────────────┴────────────┴────────────┴────────────┘
```

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
├── cli.py                 # LightningCLI entry point
├── config/
│   ├── experiment/        # Training configs (default.yaml, debug.yaml)
│   └── model/             # Model architectures (v9-c.yaml, v9-s.yaml, ...)
├── model/
│   ├── yolo.py            # Model builder from DSL
│   └── module.py          # Layer definitions
├── training/
│   ├── module.py          # LightningModule
│   └── loss.py            # Loss functions
└── data/
    ├── datamodule.py      # LightningDataModule
    └── transforms.py      # Data augmentations
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
