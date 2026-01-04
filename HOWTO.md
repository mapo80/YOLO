# How To Use YOLO

This guide explains how to use the YOLO training pipeline built on PyTorch Lightning.

## Quick Start

### Training

```bash
# Train with default config
python -m yolo.cli fit --config yolo/config/experiment/default.yaml

# Train with custom parameters
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.learning_rate=0.001 \
    --data.batch_size=32 \
    --trainer.max_epochs=100

# Debug/quick test
python -m yolo.cli fit --config yolo/config/experiment/debug.yaml
```

### Validation

```bash
# Validate a trained model
python -m yolo.cli validate --config yolo/config/experiment/default.yaml \
    --ckpt_path=runs/best.ckpt
```

### Inference

```bash
# Single image
python -m yolo.cli predict --checkpoint runs/best.ckpt --source image.jpg

# Directory of images
python -m yolo.cli predict --checkpoint runs/best.ckpt --source images/ --output results/

# Custom thresholds
python -m yolo.cli predict --checkpoint runs/best.ckpt --source image.jpg --conf 0.5 --iou 0.5
```

## Configuration

All training configuration is done through YAML files in `yolo/config/experiment/`.

### Example Configuration

```yaml
# yolo/config/experiment/my_experiment.yaml

trainer:
  max_epochs: 100
  accelerator: auto
  devices: auto
  precision: 16-mixed

model:
  model_config: v9-c      # Model architecture (v9-c, v9-s, v9-m, etc.)
  num_classes: 80
  learning_rate: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  box_loss_weight: 7.5
  cls_loss_weight: 0.5
  dfl_loss_weight: 1.5

data:
  root: data/coco
  train_images: train2017
  val_images: val2017
  train_ann: annotations/instances_train2017.json
  val_ann: annotations/instances_val2017.json
  batch_size: 16
  num_workers: 8
  image_size: [640, 640]
  # Augmentation
  mosaic_prob: 1.0
  mixup_prob: 0.15
  flip_lr: 0.5
```

### CLI Override

Any parameter can be overridden from command line:

```bash
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.model_config=v9-s \
    --model.learning_rate=0.005 \
    --data.batch_size=8 \
    --trainer.max_epochs=50
```

## Custom Model Architecture

Model architectures are defined in YAML files under `yolo/config/model/`.

### Architecture DSL

```yaml
# yolo/config/model/custom.yaml
name: custom

anchor:
  reg_max: 16
  strides: [8, 16, 32]

model:
  backbone:
    - Conv:
        args: {out_channels: 64, kernel_size: 3, stride: 2}
        source: 0
    - Conv:
        args: {out_channels: 128, kernel_size: 3, stride: 2}
    - RepNCSPELAN:
        args: {out_channels: 256, part_channels: 128}
        tags: B3  # Tag for later reference

  neck:
    - SPPELAN:
        args: {out_channels: 512}
    - UpSample:
        args: {scale_factor: 2, mode: nearest}
    - Concat:
        source: [-1, B3]  # Reference tagged layer

  head:
    - MultiheadDetection:
        source: [P3, P4, P5]
        tags: Main
        output: True
```

### DSL Keywords

- `tags`: Label any module for later reference
- `source`: Input source (default: `-1` = previous layer). Can be:
  - Relative index: `-1`, `-2`
  - Absolute index: `0`, `5`
  - Tag name: `B3`, `N4`
  - List: `[-1, B3]`
- `args`: Constructor arguments
- `output`: Mark as model output

## Custom Blocks

To add a custom block:

1. **Define in `yolo/model/module.py`:**

```python
class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

2. **Use in config:**

```yaml
- CustomBlock:
    args: {out_channels: 256}
```

## Dataset Formats

The pipeline supports two dataset formats: **COCO** (default) and **YOLO**.

Configure the format via the `data.format` parameter:

```yaml
data:
  format: coco   # or 'yolo'
```

Or override via CLI:
```bash
python -m yolo.cli fit --config config.yaml --data.format=yolo
```

### COCO Format (Default)

Standard COCO JSON annotation format.

**Directory structure:**
```
data/coco/
├── train2017/
│   └── *.jpg
├── val2017/
│   └── *.jpg
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

**Configuration:**
```yaml
data:
  format: coco
  root: data/coco
  train_images: train2017
  val_images: val2017
  train_ann: annotations/instances_train2017.json
  val_ann: annotations/instances_val2017.json
```

### YOLO Format

Standard YOLO `.txt` annotation format with normalized coordinates.

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

**Label file format** (one `.txt` per image, same name):
```
class_id x_center y_center width height
```
All coordinates are normalized (0-1 range).

**Configuration:**
```yaml
data:
  format: yolo
  root: dataset
  train_images: train/images
  train_labels: train/labels
  val_images: valid/images
  val_labels: valid/labels
```

### Custom Dataset

For custom datasets, configure the paths:

```bash
# COCO format
python -m yolo.cli fit \
    --data.format=coco \
    --data.root=data/my_dataset \
    --data.train_images=images/train \
    --data.val_images=images/val \
    --data.train_ann=annotations/train.json \
    --data.val_ann=annotations/val.json \
    --model.num_classes=10

# YOLO format
python -m yolo.cli fit \
    --data.format=yolo \
    --data.root=data/my_dataset \
    --data.train_images=train/images \
    --data.train_labels=train/labels \
    --data.val_images=valid/images \
    --data.val_labels=valid/labels \
    --model.num_classes=10
```

## Metrics

The following metrics are logged during validation:

| Metric | Description |
|--------|-------------|
| `val/mAP` | mAP @ IoU=0.50:0.95 (COCO primary) |
| `val/mAP50` | mAP @ IoU=0.50 |
| `val/mAP75` | mAP @ IoU=0.75 |
| `val/precision` | Mean precision across all classes |
| `val/recall` | Mean recall across all classes |
| `val/f1` | Mean F1 score across all classes |

### Metrics Plots

When `save_metrics_plots: true` (default), the following plots are automatically generated for each validation epoch:

- **PR_curve.png**: Precision-Recall curve per class
- **F1_curve.png**: F1 vs. Confidence threshold curve
- **P_curve.png**: Precision vs. Confidence curve
- **R_curve.png**: Recall vs. Confidence curve
- **confusion_matrix.png**: Confusion matrix with class predictions

Plots are saved to `runs/<experiment>/metrics/epoch_<N>/`.

### Configure Metrics

```yaml
model:
  # Enable/disable metrics logging
  log_map: true
  log_map_50: true
  log_map_75: true
  log_precision: true
  log_recall: true
  log_f1: true

  # Metrics plots
  save_metrics_plots: true
  metrics_plots_dir: null  # Auto: runs/<experiment>/metrics/
```

## Learning Rate Schedulers

The training pipeline supports multiple learning rate schedulers:

### Available Schedulers

| Scheduler | Description |
|-----------|-------------|
| `cosine` | Cosine annealing (default) - smooth decay to minimum LR |
| `linear` | Linear decay from initial to final LR |
| `step` | Step decay every N epochs |
| `one_cycle` | One cycle policy (super-convergence) |

### Configuration

```yaml
model:
  lr_scheduler: cosine  # Options: cosine, linear, step, one_cycle

  # Common parameters
  lr_min_factor: 0.01   # final_lr = initial_lr * lr_min_factor

  # Step scheduler
  step_size: 30         # Epochs between LR decay
  step_gamma: 0.1       # LR multiplication factor

  # OneCycle scheduler
  one_cycle_pct_start: 0.3        # Warmup fraction
  one_cycle_div_factor: 25.0      # initial_lr = max_lr / div_factor
  one_cycle_final_div_factor: 10000.0  # final_lr = initial_lr / final_div_factor
```

### Examples

```bash
# Cosine annealing (default)
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.lr_scheduler=cosine

# OneCycle for faster convergence
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.lr_scheduler=one_cycle \
    --model.one_cycle_pct_start=0.3

# Step decay (classic approach)
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.lr_scheduler=step \
    --model.step_size=30 \
    --model.step_gamma=0.1
```

## Layer Freezing (Transfer Learning)

Freeze layers for efficient transfer learning on custom datasets:

### Configuration

```yaml
model:
  # Freeze entire backbone
  freeze_backbone: true
  freeze_until_epoch: 10  # Unfreeze after epoch 10 (0 = always frozen)

  # Or freeze specific layers by name pattern
  freeze_layers:
    - backbone_conv1
    - stem
```

### Examples

```bash
# Freeze backbone for first 10 epochs
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.weight_path=true \
    --model.freeze_backbone=true \
    --model.freeze_until_epoch=10

# Freeze specific layers permanently
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.weight_path=true \
    --model.freeze_layers='[backbone_conv1, backbone_conv2]'
```

### Transfer Learning Workflow

1. **Load pretrained weights**: `--model.weight_path=true`
2. **Freeze backbone**: `--model.freeze_backbone=true`
3. **Train head for N epochs**: `--model.freeze_until_epoch=10`
4. **Unfreeze and fine-tune**: Automatic after `freeze_until_epoch`

## Model Export

Export trained models to various formats for deployment:

### Export Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| ONNX | Open Neural Network Exchange | Cross-platform, TensorRT, OpenVINO |
| TFLite | TensorFlow Lite | Mobile, Edge devices |
| SavedModel | TensorFlow SavedModel | TensorFlow Serving, TF.js |

### ONNX Export

```bash
# Basic export
python -m yolo.cli export --checkpoint best.ckpt --format onnx

# With optimizations
python -m yolo.cli export --checkpoint best.ckpt --format onnx \
    --simplify --dynamic-batch

# FP16 precision (CUDA only)
python -m yolo.cli export --checkpoint best.ckpt --format onnx --half
```

### TFLite Export

```bash
# FP32 (default)
python -m yolo.cli export --checkpoint best.ckpt --format tflite

# FP16 quantization
python -m yolo.cli export --checkpoint best.ckpt --format tflite \
    --quantization fp16

# INT8 quantization (requires calibration images)
python -m yolo.cli export --checkpoint best.ckpt --format tflite \
    --quantization int8 \
    --calibration-images /path/to/images/ \
    --num-calibration 100
```

### SavedModel Export

```bash
python -m yolo.cli export --checkpoint best.ckpt --format saved_model
```

### Export Options

| Option | Description | Default |
|--------|-------------|---------|
| `--size` | Input image size | 640 |
| `--opset` | ONNX opset version | 17 |
| `--simplify` | Simplify ONNX model | False |
| `--dynamic-batch` | Dynamic batch size (ONNX) | False |
| `--half` | FP16 precision (ONNX with CUDA) | False |
| `--quantization` | TFLite quantization (fp32, fp16, int8) | fp32 |

## Project Structure

```
yolo/
├── cli.py                    # CLI entry point (train, predict, export)
├── config/
│   ├── config.py             # Dataclass definitions
│   ├── experiment/           # Training configs
│   │   ├── default.yaml
│   │   └── debug.yaml
│   └── model/                # Model architecture DSL
│       ├── v9-c.yaml
│       ├── v9-s.yaml
│       └── ...
├── data/
│   ├── datamodule.py         # LightningDataModule
│   └── transforms.py         # Data augmentations
├── training/
│   ├── module.py             # LightningModule (training logic)
│   ├── loss.py               # Loss functions
│   └── callbacks.py          # Custom callbacks (EMA, metrics display)
├── model/
│   ├── yolo.py               # Model builder
│   └── module.py             # Layer definitions
├── tools/
│   ├── export.py             # Model export (ONNX, TFLite, SavedModel)
│   └── inference.py          # Inference utilities
└── utils/
    ├── bounding_box_utils.py # BBox utilities
    ├── metrics.py            # Detection metrics (mAP, P/R, confusion matrix)
    └── logger.py             # Logging
```
