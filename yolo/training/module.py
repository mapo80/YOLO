"""
YOLOModule - PyTorch Lightning module for YOLO training.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision

from yolo.model.yolo import YOLO
from yolo.training.loss import YOLOLoss
from yolo.utils.bounding_box_utils import Vec2Box, bbox_nms


class YOLOModule(L.LightningModule):
    """
    Lightning module for YOLO object detection training.

    Args:
        model_config: Name of model architecture (e.g., "v9-c", "v9-s")
        num_classes: Number of detection classes
        image_size: Input image size [width, height]
        learning_rate: Initial learning rate
        momentum: SGD momentum
        weight_decay: Weight decay for regularization
        warmup_epochs: Number of warmup epochs
        warmup_momentum: Starting momentum for warmup
        warmup_bias_lr: Starting bias learning rate for warmup
        box_loss_weight: Weight for box regression loss
        cls_loss_weight: Weight for classification loss
        dfl_loss_weight: Weight for distribution focal loss
        weight_path: Path to pretrained weights (None to train from scratch)
        nms_conf_threshold: NMS confidence threshold
        nms_iou_threshold: NMS IoU threshold
        nms_max_detections: Maximum detections per image
    """

    def __init__(
        self,
        # Model
        model_config: str = "v9-c",
        num_classes: int = 80,
        image_size: List[int] = [640, 640],
        # Optimizer
        learning_rate: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        # Warmup
        warmup_epochs: int = 3,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        # Loss weights
        box_loss_weight: float = 7.5,
        cls_loss_weight: float = 0.5,
        dfl_loss_weight: float = 1.5,
        # Weights
        weight_path: Optional[str] = None,
        # NMS
        nms_conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.65,
        nms_max_detections: int = 300,
        # Metrics - IoU thresholds
        log_map: bool = True,
        log_map_50: bool = True,
        log_map_75: bool = True,
        log_map_95: bool = True,
        log_map_per_size: bool = True,
        log_mar_100: bool = True,
        log_mar_per_size: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load model architecture from DSL YAML
        model_yaml_path = Path(__file__).parent.parent / "config" / "model" / f"{model_config}.yaml"
        model_cfg = OmegaConf.load(model_yaml_path)

        # Ensure anchor config has defaults
        if "anchor" not in model_cfg:
            model_cfg.anchor = {}
        if "reg_max" not in model_cfg.anchor:
            model_cfg.anchor.reg_max = 16
        if "strides" not in model_cfg.anchor:
            model_cfg.anchor.strides = [8, 16, 32]

        # Build model
        self.model = YOLO(model_cfg, class_num=num_classes)

        # Load pretrained weights if provided
        if weight_path:
            self.model.save_load_weights(Path(weight_path))

        # Store config for loss/converter setup
        self._model_cfg = model_cfg
        self._loss_fn: Optional[YOLOLoss] = None
        self._vec2box: Optional[Vec2Box] = None

        # Metrics - standard mAP with COCO thresholds
        self.val_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
        )

        # Separate metric for mAP@95 (strict IoU threshold)
        # Must include 0.5 and 0.75 for map_50/map_75 calculation
        self.val_map_95 = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[0.5, 0.75, 0.95],  # Include 0.5, 0.75 to avoid errors
        ) if log_map_95 else None

    def setup(self, stage: str) -> None:
        """Setup loss function and converter after model is on device."""
        if stage == "fit" or stage == "validate":
            device = self.device
            image_size = self.hparams.image_size

            # Create Vec2Box converter
            self._vec2box = Vec2Box(
                self.model,
                self._model_cfg.anchor,
                image_size,
                device
            )

            # Create loss function
            self._loss_fn = YOLOLoss(
                vec2box=self._vec2box,
                class_num=self.hparams.num_classes,
                reg_max=self._model_cfg.anchor.reg_max,
                box_weight=self.hparams.box_loss_weight,
                cls_weight=self.hparams.cls_loss_weight,
                dfl_weight=self.hparams.dfl_loss_weight,
            )

    def forward(self, x: Tensor) -> Dict[str, Any]:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch: Tuple[Tensor, List], batch_idx: int) -> Tensor:
        """Training step."""
        images, targets = batch

        # Forward pass
        outputs = self.model(images)

        # Compute loss
        loss, loss_dict = self._loss_fn(outputs, targets)

        # Log total loss to progress bar
        self.log("loss", loss, prog_bar=True, sync_dist=True)

        # Log individual loss components to progress bar with short names
        if "box_loss" in loss_dict:
            self.log("box", loss_dict["box_loss"], prog_bar=True, sync_dist=True)
        if "cls_loss" in loss_dict:
            self.log("cls", loss_dict["cls_loss"], prog_bar=True, sync_dist=True)
        if "dfl_loss" in loss_dict:
            self.log("dfl", loss_dict["dfl_loss"], prog_bar=True, sync_dist=True)

        # Log full names to TensorBoard/loggers only
        self.log("train/loss", loss, prog_bar=False, sync_dist=True)
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, prog_bar=False, sync_dist=True)

        # Log learning rate
        opt = self.optimizers()
        if opt is not None:
            lr = opt.param_groups[0]["lr"]
            self.log("lr", lr, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Tuple[Tensor, List], batch_idx: int) -> None:
        """Validation step - compute predictions and update metrics."""
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

        # Update metrics
        self.val_map.update(preds_formatted, targets_formatted)
        if self.val_map_95 is not None:
            self.val_map_95.update(preds_formatted, targets_formatted)

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics."""
        metrics = self.val_map.compute()

        log_dict = {}

        if self.hparams.log_map:
            log_dict["val/mAP"] = metrics["map"]

        if self.hparams.log_map_50:
            log_dict["val/mAP50"] = metrics["map_50"]

        if self.hparams.log_map_75:
            log_dict["val/mAP75"] = metrics["map_75"]

        if self.hparams.log_map_95 and self.val_map_95 is not None:
            # mAP at IoU=0.95 from dedicated metric
            metrics_95 = self.val_map_95.compute()
            log_dict["val/mAP95"] = metrics_95["map"]
            self.val_map_95.reset()

        if self.hparams.log_map_per_size:
            log_dict["val/mAP_small"] = metrics["map_small"]
            log_dict["val/mAP_medium"] = metrics["map_medium"]
            log_dict["val/mAP_large"] = metrics["map_large"]

        if self.hparams.log_mar_100:
            log_dict["val/mAR100"] = metrics["mar_100"]

        if self.hparams.log_mar_per_size:
            log_dict["val/mAR_small"] = metrics["mar_small"]
            log_dict["val/mAR_medium"] = metrics["mar_medium"]
            log_dict["val/mAR_large"] = metrics["mar_large"]

        # Log to loggers (TensorBoard, etc.) but not to progress bar
        # The MetricsTableCallback handles the clean display
        self.log_dict(log_dict, prog_bar=False, sync_dist=True)
        self.val_map.reset()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters into groups
        bias_params = []
        norm_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if "bias" in name:
                bias_params.append(param)
            elif "bn" in name and "weight" in name:
                norm_params.append(param)
            else:
                other_params.append(param)

        # Parameter groups with different weight decay
        param_groups = [
            {"params": bias_params, "weight_decay": 0.0},
            {"params": norm_params, "weight_decay": 0.0},
            {"params": other_params},
        ]

        optimizer = torch.optim.SGD(
            param_groups,
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=True,
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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """Custom optimizer step with warmup."""
        # Warmup logic
        if self.trainer.global_step < self._warmup_steps:
            warmup_progress = self.trainer.global_step / self._warmup_steps

            # Linear warmup for learning rate
            lr_scale = warmup_progress
            for i, pg in enumerate(optimizer.param_groups):
                if i == 0:  # bias params
                    pg["lr"] = self.hparams.warmup_bias_lr * (1 - warmup_progress) + \
                               self.hparams.learning_rate * warmup_progress
                else:
                    pg["lr"] = self.hparams.learning_rate * lr_scale

                # Momentum warmup
                if "momentum" in pg:
                    pg["momentum"] = self.hparams.warmup_momentum + \
                                     (self.hparams.momentum - self.hparams.warmup_momentum) * warmup_progress

        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    @property
    def _warmup_steps(self) -> int:
        """Calculate total warmup steps."""
        if not hasattr(self, "_cached_warmup_steps"):
            steps_per_epoch = len(self.trainer.train_dataloader)
            self._cached_warmup_steps = self.hparams.warmup_epochs * steps_per_epoch
        return self._cached_warmup_steps

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
        """Format predictions for torchmetrics."""
        formatted = []
        for pred in predictions:
            if len(pred) == 0:
                formatted.append({
                    "boxes": torch.zeros((0, 4), device=self.device),
                    "scores": torch.zeros((0,), device=self.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=self.device),
                })
            else:
                # pred format: [class, x1, y1, x2, y2, confidence]
                formatted.append({
                    "boxes": pred[:, 1:5],
                    "scores": pred[:, 5],
                    "labels": pred[:, 0].long(),
                })
        return formatted

    def _format_targets(self, targets: List) -> List[Dict[str, Tensor]]:
        """Format targets for torchmetrics."""
        formatted = []
        for target in targets:
            if isinstance(target, dict):
                # Already in dict format from CocoDetection
                boxes = target.get("boxes", torch.zeros((0, 4), device=self.device))
                labels = target.get("labels", torch.zeros((0,), dtype=torch.long, device=self.device))
                formatted.append({
                    "boxes": boxes.to(self.device),
                    "labels": labels.to(self.device),
                })
            elif isinstance(target, Tensor):
                # Tensor format: [class, x1, y1, x2, y2]
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
                # List of annotations from COCO format
                boxes = []
                labels = []
                for ann in target:
                    if "bbox" in ann:
                        # COCO format: [x, y, w, h] -> [x1, y1, x2, y2]
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
