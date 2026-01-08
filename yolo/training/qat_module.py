"""
QAT Module - PyTorch Lightning module for Quantization-Aware Training.

This module extends YOLOModule to support QAT fine-tuning of pre-trained models.
It handles:
- Loading a pre-trained checkpoint
- Preparing the model for QAT with fake quantization (Eager Mode)
- Fine-tuning with QAT-specific optimizations
- Converting to clean model after training for export
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from yolo.training.module import YOLOModule
from yolo.training.qat import (
    QATCallback,
    convert_qat_model,
    get_qat_training_config,
    prepare_model_for_qat,
    validate_model_for_qat,
)
from yolo.utils.logger import logger


class QATModule(L.LightningModule):
    """
    Lightning module for Quantization-Aware Training of YOLO models.

    This module wraps a pre-trained YOLOModule and prepares it for QAT fine-tuning.
    After training, the model can be exported to INT8 TFLite format.

    Uses Eager Mode QAT (not FX) because YOLO models have dynamic control flow.

    Args:
        checkpoint_path: Path to pre-trained YOLOModule checkpoint
        backend: Quantization backend ("qnnpack" for mobile, "x86" for server)
        learning_rate: Learning rate for QAT fine-tuning (default: 0.0001)
        weight_decay: Weight decay for regularization
        warmup_epochs: Number of warmup epochs
        freeze_bn_after_epoch: Freeze batch norm statistics after this epoch
        disable_observer_after_epoch: Disable observers after this epoch
        nms_conf_threshold: NMS confidence threshold
        nms_iou_threshold: NMS IoU threshold
        nms_max_detections: Maximum detections per image
    """

    def __init__(
        self,
        checkpoint_path: str,
        backend: str = "qnnpack",
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0005,
        warmup_epochs: int = 1,
        freeze_bn_after_epoch: int = 5,
        disable_observer_after_epoch: Optional[int] = None,
        nms_conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.65,
        nms_max_detections: int = 300,
        # Loss weights (inherit from base model if not specified)
        box_loss_weight: Optional[float] = None,
        cls_loss_weight: Optional[float] = None,
        dfl_loss_weight: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load base module from checkpoint
        logger.info(f"📂 Loading base model from: {checkpoint_path}")
        self._base_module = YOLOModule.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )

        # Extract configuration from base module
        self._num_classes = self._base_module.hparams.num_classes
        self._image_size = self._base_module.hparams.image_size
        self._model_cfg = self._base_module._model_cfg
        self._class_names = getattr(self._base_module, '_class_names', None)

        # Copy loss weights from base module if not specified
        self._box_loss_weight = box_loss_weight or self._base_module.hparams.box_loss_weight
        self._cls_loss_weight = cls_loss_weight or self._base_module.hparams.cls_loss_weight
        self._dfl_loss_weight = dfl_loss_weight or self._base_module.hparams.dfl_loss_weight

        # Validate model compatibility
        is_valid, warnings = validate_model_for_qat(self._base_module.model)
        if warnings:
            for warning in warnings:
                logger.warning(f"⚠️  {warning}")

        # Prepare model for QAT (Eager Mode)
        logger.info(f"🔧 Preparing model for QAT (backend: {backend})")
        self.model = prepare_model_for_qat(
            self._base_module.model,
            backend=backend,
            image_size=tuple(self._image_size),
            inplace=False,
        )

        # Set quantization backend
        torch.backends.quantized.engine = backend

        # Initialize components (will be set in setup())
        self._loss_fn = None
        self._vec2box = None
        self._det_metrics = None

        # QAT callback for batch norm freezing
        self._qat_callback = QATCallback(
            freeze_bn_after_epoch=freeze_bn_after_epoch,
            disable_observer_after_epoch=disable_observer_after_epoch,
        )

        # Track if model has been converted
        self._is_converted = False

        # Expose image_size for datamodule compatibility
        self.image_size = list(self._image_size)

        logger.info(f"✅ QAT module initialized")
        logger.info(f"   📊 Classes: {self._num_classes}")
        logger.info(f"   📐 Image size: {self._image_size}")
        logger.info(f"   🔧 Backend: {backend}")

    def setup(self, stage: str) -> None:
        """Setup loss function, converter, and metrics after model is on device."""
        if stage == "fit" or stage == "validate":
            from yolo.training.loss import YOLOLoss
            from yolo.utils.bounding_box_utils import Vec2Box
            from yolo.utils.metrics import DetMetrics

            device = self.device

            # Create Vec2Box converter using base model (same architecture)
            self._vec2box = Vec2Box(
                self._base_module.model,
                self._model_cfg.anchor,
                self._image_size,
                device,
            )

            # Create loss function
            self._loss_fn = YOLOLoss(
                vec2box=self._vec2box,
                class_num=self._num_classes,
                reg_max=self._model_cfg.anchor.reg_max,
                box_weight=self._box_loss_weight,
                cls_weight=self._cls_loss_weight,
                dfl_weight=self._dfl_loss_weight,
            )

            # Initialize detection metrics
            if self._class_names:
                self._det_metrics = DetMetrics(names=self._class_names)

    def forward(self, x: Tensor) -> Dict[str, Any]:
        """Forward pass through the QAT model."""
        # Apply fake quantization to weights during forward
        self._apply_fake_quant_to_weights()
        return self.model(x)

    def _apply_fake_quant_to_weights(self) -> None:
        """Apply fake quantization to weights during QAT training."""
        if not self.training:
            return

        for module in self.model.modules():
            if hasattr(module, 'weight_fake_quant') and hasattr(module, 'weight'):
                # The fake quantizer updates the weight in-place with quantized version
                # We need to temporarily replace weight with quantized weight
                pass  # PyTorch handles this automatically during forward

    def training_step(self, batch: Tuple[Tensor, List], batch_idx: int) -> Tensor:
        """Training step with QAT callback integration."""
        images, targets = batch

        # Apply QAT callback at epoch start
        if batch_idx == 0:
            self._qat_callback.on_train_epoch_start(self.current_epoch, self.model)

        # Forward pass (fake quant is applied automatically)
        outputs = self.model(images)

        # Compute loss
        loss, loss_dict = self._loss_fn(outputs, targets)

        # Log losses
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/loss", loss, prog_bar=False, sync_dist=True)

        if "box_loss" in loss_dict:
            self.log("box", loss_dict["box_loss"], prog_bar=True, sync_dist=True)
        if "cls_loss" in loss_dict:
            self.log("cls", loss_dict["cls_loss"], prog_bar=True, sync_dist=True)
        if "dfl_loss" in loss_dict:
            self.log("dfl", loss_dict["dfl_loss"], prog_bar=True, sync_dist=True)

        # Log learning rate
        opt = self.optimizers()
        if opt is not None:
            lr = opt.param_groups[0]["lr"]
            self.log("lr", lr, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Tuple[Tensor, List], batch_idx: int) -> None:
        """Validation step - compute predictions and update metrics."""
        from yolo.utils.bounding_box_utils import bbox_nms

        images, targets = batch

        # Forward pass
        outputs = self.model(images)

        # Convert predictions to boxes
        pred_cls, pred_anc, pred_box = self._vec2box(outputs["Main"])

        # Apply NMS
        predictions = bbox_nms(
            pred_cls,
            pred_box,
            nms_cfg=self._create_nms_config(),
        )

        # Format predictions and targets for metrics
        preds_formatted = self._format_predictions(predictions)
        targets_formatted = self._format_targets(targets)

        # Update detection metrics
        if self._det_metrics is not None:
            self._det_metrics.update(preds_formatted, targets_formatted)

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics."""
        if self._det_metrics is None:
            return

        # Process metrics
        conf_prod = self.hparams.nms_conf_threshold
        metrics = self._det_metrics.process(
            save_dir=None,
            plot=False,
            conf_prod=conf_prod,
        )

        # Log metrics
        log_dict = {
            "val/mAP": metrics["map"],
            "val/mAP50": metrics["map50"],
            "val/mAP75": metrics["map75"],
            "val/precision": metrics["precision"],
            "val/recall": metrics["recall"],
            "val/f1": metrics["f1"],
        }

        self.log_dict(log_dict, prog_bar=False, sync_dist=False)

        # Log mAP to progress bar
        self.log("mAP", metrics["map"], prog_bar=True, sync_dist=False)

        # Reset metrics
        self._det_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler for QAT."""
        # Collect trainable parameters
        param_groups = [
            {"params": [], "weight_decay": 0.0},  # bias
            {"params": [], "weight_decay": 0.0},  # batch norm
            {"params": []},  # other
        ]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name:
                param_groups[0]["params"].append(param)
            elif "bn" in name and "weight" in name:
                param_groups[1]["params"].append(param)
            else:
                param_groups[2]["params"].append(param)

        # Use AdamW for fine-tuning
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.learning_rate * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def convert_to_quantized(self) -> None:
        """
        Convert the QAT model to a clean model (remove fake quantizers).

        Call this after training is complete before exporting.
        """
        if self._is_converted:
            logger.warning("Model has already been converted")
            return

        logger.info("🔄 Converting QAT model (removing fake quantizers)...")
        self.model = convert_qat_model(self.model, inplace=True)
        self._is_converted = True
        logger.info("✅ Model converted - ready for export")

    def export_quantized(
        self,
        output_path: str,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> str:
        """
        Export the model to TorchScript.

        Args:
            output_path: Path for the exported model
            image_size: Input image size (default: from training)

        Returns:
            Path to the exported model
        """
        if not self._is_converted:
            self.convert_to_quantized()

        if image_size is None:
            image_size = tuple(self._image_size)

        # Create example input
        example_input = torch.randn(1, 3, image_size[0], image_size[1])

        # Trace the model
        self.model.eval()
        traced_model = torch.jit.trace(self.model, example_input)

        # Save
        traced_model.save(output_path)
        logger.info(f"💾 Model exported to: {output_path}")

        return output_path

    def get_model_for_export(self) -> nn.Module:
        """
        Get the model ready for export (with fake quantizers removed).

        Returns:
            Clean model ready for ONNX/TFLite export
        """
        if not self._is_converted:
            self.convert_to_quantized()
        return self.model

    def _create_nms_config(self):
        """Create NMS configuration object."""
        from dataclasses import dataclass

        @dataclass
        class NMSConfig:
            min_confidence: float
            min_iou: float
            max_bbox: int

        return NMSConfig(
            min_confidence=self.hparams.nms_conf_threshold,
            min_iou=self.hparams.nms_iou_threshold,
            max_bbox=self.hparams.nms_max_detections,
        )

    def _format_predictions(self, predictions: List[Tensor]) -> List[Dict[str, Tensor]]:
        """Format predictions for metrics."""
        formatted = []
        for pred in predictions:
            if len(pred) == 0:
                formatted.append({
                    "boxes": torch.zeros((0, 4), device=self.device),
                    "scores": torch.zeros((0,), device=self.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=self.device),
                })
            else:
                formatted.append({
                    "boxes": pred[:, 1:5],
                    "scores": pred[:, 5],
                    "labels": pred[:, 0].long(),
                })
        return formatted

    def _format_targets(self, targets: List) -> List[Dict[str, Tensor]]:
        """Format targets for metrics."""
        formatted = []
        for target in targets:
            if isinstance(target, dict):
                boxes = target.get("boxes", torch.zeros((0, 4), device=self.device))
                labels = target.get("labels", torch.zeros((0,), dtype=torch.long, device=self.device))
                formatted.append({
                    "boxes": boxes.to(self.device),
                    "labels": labels.to(self.device),
                })
            elif isinstance(target, Tensor):
                if len(target) == 0:
                    formatted.append({
                        "boxes": torch.zeros((0, 4), device=self.device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=self.device),
                    })
                else:
                    formatted.append({
                        "boxes": target[:, 1:5].to(self.device),
                        "labels": target[:, 0].long().to(self.device),
                    })
            else:
                # COCO format annotations
                boxes = []
                labels = []
                for ann in target:
                    if "bbox" in ann:
                        x, y, w, h = ann["bbox"]
                        boxes.append([x, y, x + w, y + h])
                        labels.append(ann.get("category_id", 0))

                if boxes:
                    formatted.append({
                        "boxes": torch.tensor(boxes, device=self.device, dtype=torch.float32),
                        "labels": torch.tensor(labels, device=self.device, dtype=torch.long),
                    })
                else:
                    formatted.append({
                        "boxes": torch.zeros((0, 4), device=self.device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=self.device),
                    })

        return formatted

    @classmethod
    def from_yolo_module(
        cls,
        yolo_module: YOLOModule,
        backend: str = "qnnpack",
        **kwargs,
    ) -> "QATModule":
        """
        Create a QATModule from an existing YOLOModule.

        This is useful when you have a trained model in memory.

        Args:
            yolo_module: Pre-trained YOLOModule instance
            backend: Quantization backend
            **kwargs: Additional arguments for QATModule

        Returns:
            QATModule instance ready for fine-tuning
        """
        # Save temporary checkpoint
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            temp_path = f.name

        # Save module state
        trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
        trainer.strategy.connect(yolo_module)
        trainer.save_checkpoint(temp_path)

        # Create QAT module from checkpoint
        qat_module = cls(
            checkpoint_path=temp_path,
            backend=backend,
            **kwargs,
        )

        # Clean up
        Path(temp_path).unlink()

        return qat_module
