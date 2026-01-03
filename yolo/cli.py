"""
YOLO Training CLI - LightningCLI entry point.

Usage:
    python -m yolo.cli fit --config config/experiment/default.yaml
    python -m yolo.cli fit --config config/experiment/default.yaml --model.learning_rate=0.001
    python -m yolo.cli validate --ckpt_path=runs/best.ckpt
    python -m yolo.cli test --ckpt_path=runs/best.ckpt
"""

from lightning.pytorch.cli import LightningCLI

from yolo.data.datamodule import YOLODataModule
from yolo.training.module import YOLOModule


def main():
    cli = LightningCLI(
        YOLOModule,
        YOLODataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
