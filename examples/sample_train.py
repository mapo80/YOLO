"""
Sample training script using the new Lightning-based training pipeline.

Usage:
    python examples/sample_train.py

Or use the CLI directly:
    python -m yolo.cli fit --config yolo/config/experiment/default.yaml
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar

from yolo.data.datamodule import YOLODataModule
from yolo.training.module import YOLOModule


def main():
    # Create model
    model = YOLOModule(
        model_config="v9-c",
        num_classes=80,
        image_size=[640, 640],
        learning_rate=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box_loss_weight=7.5,
        cls_loss_weight=0.5,
        dfl_loss_weight=1.5,
    )

    # Create datamodule
    datamodule = YOLODataModule(
        root="data/coco",
        train_images="train2017",
        val_images="val2017",
        train_ann="annotations/instances_train2017.json",
        val_ann="annotations/instances_val2017.json",
        batch_size=16,
        num_workers=8,
    )
    # Set image_size to match model (normally done by CLI automatically)
    datamodule._image_size = tuple(model.hparams.image_size)

    # Create trainer
    trainer = Trainer(
        max_epochs=500,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=10.0,
        callbacks=[
            ModelCheckpoint(
                monitor="val/mAP",
                mode="max",
                save_top_k=3,
                save_last=True,
            ),
            RichProgressBar(),
        ],
    )

    # Train
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
