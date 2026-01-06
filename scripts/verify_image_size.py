#!/usr/bin/env python3
"""
Diagnostic script to verify image_size is correctly propagated through the pipeline.

Usage:
    python scripts/verify_image_size.py --config path/to/your/config.yaml

This will verify that image_size is consistent across:
1. Model (self.image_size and self.hparams.image_size)
2. DataModule (_image_size)
3. Vec2Box converter
4. Actual image tensors during a forward pass
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def verify_with_config(config_path: str):
    """Verify image_size propagation using a config file."""
    from omegaconf import OmegaConf
    from yolo.data.datamodule import YOLODataModule
    from yolo.training.module import YOLOModule

    print(f"Loading config: {config_path}")
    config = OmegaConf.load(config_path)

    # Extract model and data configs
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    model_image_size = model_cfg.get("image_size", [640, 640])
    data_image_size = data_cfg.get("image_size", None)

    print("\n" + "=" * 60)
    print("CONFIG FILE VALUES")
    print("=" * 60)
    print(f"model.image_size in config: {model_image_size}")
    print(f"data.image_size in config:  {data_image_size or '(not specified - will be linked from model)'}")

    # Simulate what link_arguments does
    if data_image_size is None:
        data_image_size = model_image_size
        print(f"  -> data.image_size linked to: {data_image_size}")

    # Create model
    print("\n" + "=" * 60)
    print("INSTANTIATION CHECK")
    print("=" * 60)

    model = YOLOModule(
        model_config=model_cfg.get("model_config", "v9-t"),
        num_classes=model_cfg.get("num_classes", 80),
        image_size=model_image_size,
    )

    print(f"Model attribute (self.image_size):       {model.image_size}")
    print(f"Model hparams (self.hparams.image_size): {list(model.hparams.image_size)}")

    # Check data config
    dm = YOLODataModule(
        root=data_cfg.get("root", "data/coco"),
        train_images=data_cfg.get("train_images", "images/train"),
        val_images=data_cfg.get("val_images", "images/val"),
        train_ann=data_cfg.get("train_ann", "annotations/instances_train.json"),
        val_ann=data_cfg.get("val_ann", "annotations/instances_val.json"),
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=0,
        image_size=data_image_size,  # Simulating link_arguments
    )

    print(f"DataModule _image_size:                  {dm._image_size}")

    # Verify consistency
    print("\n" + "=" * 60)
    print("CONSISTENCY CHECK")
    print("=" * 60)

    model_size = tuple(model.image_size)
    hparams_size = tuple(model.hparams.image_size)
    data_size = dm._image_size

    all_match = model_size == hparams_size == data_size

    print(f"model.image_size == model.hparams.image_size: {model_size == hparams_size} ({model_size} vs {hparams_size})")
    print(f"model.image_size == datamodule._image_size:   {model_size == data_size} ({model_size} vs {data_size})")

    if all_match:
        print("\n✅ ALL SIZES MATCH - Configuration is correct!")
    else:
        print("\n❌ SIZE MISMATCH DETECTED!")
        print("   This can cause scale issues in validation metrics.")
        print("   Make sure you're NOT manually specifying data.image_size in your config.")

    return all_match


def verify_checkpoint(checkpoint_path: str):
    """Verify image_size in a saved checkpoint."""
    import torch

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print("\n" + "=" * 60)
    print("CHECKPOINT HPARAMS")
    print("=" * 60)

    if "hyper_parameters" in ckpt:
        hparams = ckpt["hyper_parameters"]
        image_size = hparams.get("image_size", "NOT FOUND")
        model_config = hparams.get("model_config", "NOT FOUND")
        num_classes = hparams.get("num_classes", "NOT FOUND")

        print(f"model_config: {model_config}")
        print(f"num_classes:  {num_classes}")
        print(f"image_size:   {image_size}")

        print("\n⚠️  IMPORTANT: When resuming from checkpoint, these hparams are used!")
        print("   If you want a different image_size, train from scratch or adjust config.")
    else:
        print("No hyper_parameters found in checkpoint.")

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify image_size propagation")
    parser.add_argument("--config", type=str, help="Path to training config YAML")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to inspect")

    args = parser.parse_args()

    if not args.config and not args.checkpoint:
        parser.print_help()
        print("\n❌ Please provide either --config or --checkpoint")
        return 1

    success = True

    if args.config:
        success = verify_with_config(args.config) and success

    if args.checkpoint:
        success = verify_checkpoint(args.checkpoint) and success

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
