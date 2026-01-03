# YOLO: Official Implementation of YOLOv9, YOLOv7, YOLO-RD

[![Documentation Status](https://readthedocs.org/projects/yolo-docs/badge/?version=latest)](https://yolo-docs.readthedocs.io/en/latest/?badge=latest)
![GitHub License](https://img.shields.io/github/license/WongKinYiu/YOLO)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-792ee5.svg)](https://lightning.ai/)

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
# Train with default config
python -m yolo.cli fit --config yolo/config/experiment/default.yaml

# Custom parameters
python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
    --model.model_config=v9-c \
    --data.batch_size=16 \
    --trainer.max_epochs=300

# Debug run (small dataset, few epochs)
python -m yolo.cli fit --config yolo/config/experiment/debug.yaml
```

### Validation

```shell
python -m yolo.cli validate --config yolo/config/experiment/default.yaml \
    --ckpt_path=runs/best.ckpt
```

### Inference

```shell
python examples/sample_inference.py --image path/to/image.jpg --weights weights/v9-c.pt
```

## Features

| Feature | Description |
|---------|-------------|
| **Multi-GPU Training** | Automatic DDP with `--trainer.devices=N` |
| **Mixed Precision** | FP16/BF16 training with `--trainer.precision=16-mixed` |
| **COCO Metrics** | mAP@0.5, mAP@0.5:0.95, per-size metrics |
| **Checkpointing** | Automatic best/last model saving |
| **Early Stopping** | Stop on validation plateau |
| **Logging** | TensorBoard, WandB support |

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
