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
# Run inference on an image
python examples/sample_inference.py --image path/to/image.jpg

# With custom weights
python examples/sample_inference.py --image path/to/image.jpg --weights weights/v9-c.pt
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

## Dataset Format

The pipeline uses standard **COCO format** via `torchvision.datasets.CocoDetection`.

### Expected Directory Structure

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

### Custom Dataset

For custom datasets, create COCO-format annotations:

```bash
python -m yolo.cli fit \
    --data.root=data/my_dataset \
    --data.train_images=images/train \
    --data.val_images=images/val \
    --data.train_ann=annotations/train.json \
    --data.val_ann=annotations/val.json \
    --model.num_classes=10
```

## Metrics

The following metrics are logged during validation:

| Metric | Description |
|--------|-------------|
| `val/mAP` | mAP @ IoU=0.50:0.95 (COCO primary) |
| `val/mAP50` | mAP @ IoU=0.50 |
| `val/mAP75` | mAP @ IoU=0.75 |
| `val/mAP_small` | mAP for small objects (< 32² px) |
| `val/mAP_medium` | mAP for medium objects (32²-96² px) |
| `val/mAP_large` | mAP for large objects (> 96² px) |
| `val/mAR100` | Mean Average Recall (max 100 detections) |

## Project Structure

```
yolo/
├── cli.py                    # LightningCLI entry point
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
│   ├── module.py             # LightningModule
│   └── loss.py               # Loss functions
├── model/
│   ├── yolo.py               # Model builder
│   └── module.py             # Layer definitions
└── utils/
    ├── bounding_box_utils.py # BBox utilities
    └── logger.py             # Logging
```
