# Training Guide - Simpsons Character Detection

This guide documents the steps to train a YOLOv9-Tiny model on the Simpsons character detection dataset using PyTorch Lightning.

## Dataset

**Source:** [Simpsons Character Detection](https://universe.roboflow.com/innopolice/simpsons-lnljz) on Roboflow

**Classes (7):**
1. abraham_grampa_simpson
2. bart_simpson
3. homer_simpson
4. lisa_simpson
5. maggie_simpson
6. marge_simpson
7. ned_flanders

**Dataset Split:**
- Train: 162 images
- Validation: 15 images
- Test: 7 images

---

## Step 1: Download Dataset

The dataset is available in two formats: **COCO** and **YOLO**.

### Install Roboflow

```bash
pip install roboflow
```

### Download in COCO Format

```python
from roboflow import Roboflow

# Get your API key from https://app.roboflow.com/settings/api
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("innopolice").project("simpsons-lnljz")
version = project.version(1)

# Download COCO format
dataset = version.download("coco", location="./simpsons-coco")
```

### Download in YOLO Format

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("innopolice").project("simpsons-lnljz")
version = project.version(1)

# Download YOLO format
dataset = version.download("yolov8", location="./simpsons-yolo")
```

---

## Step 2: Restructure Dataset (COCO Format)

Our training pipeline expects the standard COCO directory structure. Roboflow exports with images and annotations in the same folder, so we need to restructure:

```python
import shutil
from pathlib import Path

# Create standard structure
base = Path("training-experiment")
(base / "simpsons-coco-std/images/train").mkdir(parents=True, exist_ok=True)
(base / "simpsons-coco-std/images/val").mkdir(parents=True, exist_ok=True)
(base / "simpsons-coco-std/annotations").mkdir(parents=True, exist_ok=True)

# Copy images
for f in (base / "simpsons-coco/train").glob("*.jpg"):
    shutil.copy(f, base / "simpsons-coco-std/images/train/")

for f in (base / "simpsons-coco/valid").glob("*.jpg"):
    shutil.copy(f, base / "simpsons-coco-std/images/val/")

# Copy annotations
shutil.copy(
    base / "simpsons-coco/train/_annotations.coco.json",
    base / "simpsons-coco-std/annotations/instances_train.json"
)
shutil.copy(
    base / "simpsons-coco/valid/_annotations.coco.json",
    base / "simpsons-coco-std/annotations/instances_val.json"
)
```

**Final Structure:**
```
training-experiment/simpsons-coco-std/
├── images/
│   ├── train/
│   │   └── *.jpg (162 images)
│   └── val/
│       └── *.jpg (15 images)
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

---

## Step 3: Training Configuration

Create a YAML configuration file for training:

**File: `training-experiment/simpsons-train.yaml`**

```yaml
# =============================================================================
# Simpsons Character Detection - Training Configuration
# =============================================================================

# Trainer Configuration
trainer:
  max_epochs: 100
  accelerator: auto      # Supports: cuda, mps, cpu
  devices: auto          # Auto-detect available devices
  precision: 16-mixed    # Mixed precision for faster training
  gradient_clip_val: 10.0
  log_every_n_steps: 10
  check_val_every_n_epoch: 1

  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: training-experiment/checkpoints
        monitor: val/mAP
        mode: max
        save_top_k: 3
        save_last: true
        filename: "simpsons-{epoch:02d}-mAP={val/mAP:.4f}"

    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: val/mAP
        mode: max
        patience: 30

    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: step

    - class_path: lightning.pytorch.callbacks.RichProgressBar

  logger:
    - class_path: lightning.pytorch.loggers.TensorBoardLogger
      init_args:
        save_dir: training-experiment/
        name: logs

# Model Configuration
model:
  model_config: v9-t     # YOLOv9-Tiny
  num_classes: 7         # Simpsons characters
  image_size: [640, 640]
  learning_rate: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  box_loss_weight: 7.5
  cls_loss_weight: 0.5
  dfl_loss_weight: 1.5
  weight_path: null      # Train from scratch

# Data Configuration
data:
  root: training-experiment/simpsons-coco-std
  train_images: images/train
  val_images: images/val
  train_ann: annotations/instances_train.json
  val_ann: annotations/instances_val.json
  batch_size: 8
  num_workers: 4
  image_size: [640, 640]
  mosaic_prob: 1.0
  mixup_prob: 0.15
  flip_lr: 0.5
  degrees: 10.0
  close_mosaic_epochs: 10
```

---

## Step 4: Run Training

### Basic Training Command

```bash
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml
```

### Training on Different Devices

The CLI automatically detects the best available device:

```bash
# Auto-detect (CUDA > MPS > CPU)
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml

# Force CUDA (NVIDIA GPU)
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml \
    --trainer.accelerator=cuda

# Force MPS (Apple Silicon)
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml \
    --trainer.accelerator=mps

# Force CPU
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml \
    --trainer.accelerator=cpu
```

### Override Parameters from CLI

```bash
# Change batch size and learning rate
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml \
    --data.batch_size=16 \
    --model.learning_rate=0.001

# Reduce epochs for quick test
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml \
    --trainer.max_epochs=10

# Multi-GPU training
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml \
    --trainer.devices=2 \
    --trainer.strategy=ddp
```

---

## Step 5: Monitor Training

### TensorBoard

```bash
tensorboard --logdir training-experiment/logs
```

Open http://localhost:6006 in your browser.

### Metrics Logged

| Metric | Description |
|--------|-------------|
| `train/loss` | Total training loss |
| `train/box_loss` | Box regression loss |
| `train/cls_loss` | Classification loss |
| `train/dfl_loss` | Distribution focal loss |
| `val/mAP` | mAP @ IoU 0.50:0.95 |
| `val/mAP50` | mAP @ IoU 0.50 |
| `val/mAP75` | mAP @ IoU 0.75 |

---

## Step 6: Validation

```bash
python -m yolo.cli validate --config training-experiment/simpsons-train.yaml \
    --ckpt_path=training-experiment/checkpoints/best.ckpt
```

---

## Step 7: Inference

### Using the Sample Script

```bash
python examples/sample_inference.py \
    --image path/to/simpsons_image.jpg \
    --weights training-experiment/checkpoints/best.ckpt \
    --model v9-t \
    --num-classes 7
```

### Programmatic Inference

```python
import torch
from PIL import Image
from yolo import create_model, bbox_nms

# Load model
model = create_model("v9-t", num_classes=7)
checkpoint = torch.load("training-experiment/checkpoints/best.ckpt")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Inference
with torch.no_grad():
    predictions = model(image_tensor)
```

---

## Checkpoints

Checkpoints are saved to `training-experiment/checkpoints/`:

- `last.ckpt` - Latest checkpoint
- `simpsons-{epoch}-mAP={mAP}.ckpt` - Top 3 best checkpoints by mAP

---

## Device Support Summary

| Device | Accelerator | Notes |
|--------|-------------|-------|
| NVIDIA GPU | `cuda` | Fastest, recommended for training |
| Apple Silicon | `mps` | Good performance on M1/M2/M3 |
| CPU | `cpu` | Slowest, use for testing only |

The `accelerator: auto` setting automatically selects the best available device.

---

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml \
    --data.batch_size=4
```

### Mixed Precision on MPS

MPS (Apple Silicon) does not fully support mixed precision (16-mixed). Use precision=32:
```bash
# Default config uses precision=32 (safe for all devices)
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml

# For CUDA, you can enable mixed precision for faster training:
python -m yolo.cli fit --config training-experiment/simpsons-train.yaml \
    --trainer.precision=16-mixed
```

### Dataset Not Found

Ensure the dataset is in the correct location:
```
training-experiment/simpsons-coco-std/
├── images/train/
├── images/val/
└── annotations/
```
