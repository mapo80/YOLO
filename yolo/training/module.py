"""
YOLOModule - PyTorch Lightning module for YOLO training.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import torch
from omegaconf import OmegaConf
from torch import Tensor

from yolo.model.yolo import YOLO
from yolo.tools.dataset_preparation import prepare_weight
from yolo.training.loss import YOLOLoss
from yolo.utils.bounding_box_utils import Vec2Box, bbox_nms
from yolo.utils.logger import logger
from yolo.utils.metrics import DetMetrics


class YOLOModule(L.LightningModule):
    """
    Lightning module for YOLO object detection training.

    Args:
        model_config: Name of model architecture (e.g., "v9-c", "v9-s")
        num_classes: Number of detection classes
        image_size: Input image size [width, height]
        optimizer: Optimizer type ("sgd" or "adamw")
        learning_rate: Initial learning rate
        momentum: SGD momentum (only used with optimizer="sgd")
        weight_decay: Weight decay for regularization
        adamw_betas: Beta coefficients for AdamW (only used with optimizer="adamw")
        warmup_epochs: Number of warmup epochs
        warmup_momentum: Starting momentum for warmup (SGD only)
        warmup_bias_lr: Starting bias learning rate for warmup
        box_loss_weight: Weight for box regression loss
        cls_loss_weight: Weight for classification loss
        dfl_loss_weight: Weight for distribution focal loss
        weight_path: Path to pretrained weights, or True to auto-download based on model_config
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
        optimizer: str = "sgd",  # "sgd" or "adamw"
        learning_rate: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        adamw_betas: List[float] = [0.9, 0.999],
        # Warmup
        warmup_epochs: int = 3,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        # Loss weights
        box_loss_weight: float = 7.5,
        cls_loss_weight: float = 0.5,
        dfl_loss_weight: float = 1.5,
        # Weights: None = from scratch, True = auto-download, str = path to weights
        weight_path: Optional[Union[str, bool]] = None,
        # NMS
        nms_conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.65,
        nms_max_detections: int = 300,
        # Metrics
        log_map: bool = True,
        log_map_50: bool = True,
        log_map_75: bool = True,
        log_precision: bool = True,
        log_recall: bool = True,
        log_f1: bool = True,
        # Class names for metrics (None = use indices)
        class_names: Optional[Dict[int, str]] = None,
        # Metrics plots
        save_metrics_plots: bool = True,
        metrics_plots_dir: Optional[str] = None,
        # Learning rate scheduler
        # Options: "cosine" (default), "linear", "one_cycle", "step"
        lr_scheduler: str = "cosine",
        # Scheduler-specific parameters
        lr_min_factor: float = 0.01,  # For cosine/linear: final_lr = initial_lr * lr_min_factor
        step_size: int = 30,  # For step scheduler: epochs between LR decay
        step_gamma: float = 0.1,  # For step scheduler: LR multiplication factor
        one_cycle_pct_start: float = 0.3,  # For one_cycle: % of training spent in warmup
        one_cycle_div_factor: float = 25.0,  # For one_cycle: initial_lr = max_lr / div_factor
        one_cycle_final_div_factor: float = 1e4,  # For one_cycle: final_lr = initial_lr / final_div_factor
        # Layer freezing for transfer learning
        freeze_backbone: bool = False,
        freeze_until_epoch: int = 0,  # Unfreeze backbone after this epoch (0 = always frozen)
        freeze_layers: Optional[List[str]] = None,  # Specific layer patterns to freeze
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
        logger.info(f"🏗️  Building model: {model_config} ({num_classes} classes)")
        self.model = YOLO(model_cfg, class_num=num_classes)

        # Load pretrained weights if provided
        if weight_path:
            self._load_weights(weight_path, model_config)
        else:
            logger.info("🎲 Training from scratch (no pretrained weights)")

        # Store config for loss/converter setup
        self._model_cfg = model_cfg
        self._loss_fn: Optional[YOLOLoss] = None
        self._vec2box: Optional[Vec2Box] = None

        # Build class names dict
        if class_names is not None:
            self._class_names = class_names
        else:
            # Default: use class indices as names
            self._class_names = {i: str(i) for i in range(num_classes)}

        # Detection metrics with custom implementation
        self._det_metrics: Optional[DetMetrics] = None

        # Track if we're resuming from a checkpoint
        self._resumed_from_checkpoint = False

        # Layer freezing state
        self._frozen_layers: List[str] = []

        # Apply initial layer freezing
        if freeze_backbone or freeze_layers:
            self._apply_layer_freezing()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading from checkpoint - log resume info."""
        self._resumed_from_checkpoint = True
        epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        logger.info(f"🔄 Resuming from checkpoint (epoch {epoch + 1}, step {global_step})")

    def setup(self, stage: str) -> None:
        """Setup loss function, converter, and metrics after model is on device."""
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

            # Initialize detection metrics
            self._det_metrics = DetMetrics(names=self._class_names)

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

        # Update detection metrics
        if self._det_metrics is not None:
            self._det_metrics.update(preds_formatted, targets_formatted)

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics."""
        if self._det_metrics is None:
            return

        # Determine save directory for plots
        save_dir = None
        if self.hparams.save_metrics_plots:
            if self.hparams.metrics_plots_dir:
                save_dir = Path(self.hparams.metrics_plots_dir)
            elif self.trainer.log_dir:
                save_dir = Path(self.trainer.log_dir) / "metrics"
            else:
                save_dir = Path("runs") / "metrics"

            # Add epoch subdirectory
            save_dir = save_dir / f"epoch_{self.current_epoch + 1}"

        # Process metrics and generate plots
        metrics = self._det_metrics.process(
            save_dir=save_dir,
            plot=self.hparams.save_metrics_plots,
        )

        log_dict = {}

        # Log mAP metrics
        if self.hparams.log_map:
            log_dict["val/mAP"] = metrics["map"]

        if self.hparams.log_map_50:
            log_dict["val/mAP50"] = metrics["map50"]

        if self.hparams.log_map_75:
            log_dict["val/mAP75"] = metrics["map75"]

        # Log precision, recall, F1
        if self.hparams.log_precision:
            log_dict["val/precision"] = metrics["precision"]

        if self.hparams.log_recall:
            log_dict["val/recall"] = metrics["recall"]

        if self.hparams.log_f1:
            log_dict["val/f1"] = metrics["f1"]

        # Log to loggers (TensorBoard, etc.) but not to progress bar
        # The MetricsTableCallback handles the clean display
        self.log_dict(log_dict, prog_bar=False, sync_dist=True)

        # Reset metrics for next epoch
        self._det_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters into groups (only trainable parameters)
        bias_params = []
        norm_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters
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

        # Select optimizer based on config
        opt_name = self.hparams.optimizer.lower()

        if opt_name == "adamw":
            # AdamW optimizer
            betas = tuple(self.hparams.adamw_betas)
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.hparams.learning_rate,
                betas=betas,
                weight_decay=self.hparams.weight_decay,
            )
            logger.info(f"Using AdamW optimizer (lr={self.hparams.learning_rate}, betas={betas})")
        else:
            # Default: SGD with Nesterov momentum
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
                nesterov=True,
            )
            logger.info(f"Using SGD optimizer (lr={self.hparams.learning_rate}, momentum={self.hparams.momentum})")

        # Create learning rate scheduler based on config
        scheduler = self._create_lr_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step" if self.hparams.lr_scheduler == "one_cycle" else "epoch",
            },
        }

    def _create_lr_scheduler(self, optimizer):
        """
        Create learning rate scheduler based on configuration.

        Supported schedulers:
            - cosine: Cosine annealing with warm restarts
            - linear: Linear decay from initial to final LR
            - step: Step decay every N epochs
            - one_cycle: One cycle policy (super-convergence)
        """
        scheduler_name = self.hparams.lr_scheduler.lower()
        max_epochs = self.trainer.max_epochs
        lr = self.hparams.learning_rate
        eta_min = lr * self.hparams.lr_min_factor

        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=eta_min,
            )
            logger.info(f"Using Cosine Annealing scheduler (T_max={max_epochs}, eta_min={eta_min:.2e})")

        elif scheduler_name == "linear":
            # Linear decay from lr to eta_min
            def linear_lambda(epoch):
                if epoch >= max_epochs:
                    return self.hparams.lr_min_factor
                return 1.0 - (1.0 - self.hparams.lr_min_factor) * (epoch / max_epochs)

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=linear_lambda,
            )
            logger.info(f"Using Linear scheduler (final_lr={eta_min:.2e})")

        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.step_size,
                gamma=self.hparams.step_gamma,
            )
            logger.info(f"Using Step scheduler (step_size={self.hparams.step_size}, gamma={self.hparams.step_gamma})")

        elif scheduler_name == "one_cycle":
            # Calculate total steps
            steps_per_epoch = len(self.trainer.train_dataloader)
            total_steps = max_epochs * steps_per_epoch

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=self.hparams.one_cycle_pct_start,
                div_factor=self.hparams.one_cycle_div_factor,
                final_div_factor=self.hparams.one_cycle_final_div_factor,
            )
            logger.info(
                f"Using OneCycle scheduler (max_lr={lr}, total_steps={total_steps}, "
                f"pct_start={self.hparams.one_cycle_pct_start})"
            )

        else:
            raise ValueError(
                f"Unknown lr_scheduler: {scheduler_name}. "
                f"Supported: cosine, linear, step, one_cycle"
            )

        return scheduler

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

                # Momentum warmup (only for SGD, AdamW doesn't have momentum param)
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

    def _load_weights(self, weight_path: Union[str, bool], model_config: str) -> None:
        """
        Load pretrained weights.

        Args:
            weight_path: True for auto-download, or path to weights file
            model_config: Model config name (e.g., "v9-c") for auto-download
        """
        if weight_path is True:
            # Auto-download: derive weight path from model config
            weights_dir = Path("weights")
            weight_file = weights_dir / f"{model_config}.pt"

            if not weight_file.exists():
                logger.info(f"🌐 Weight {weight_file} not found, downloading...")
                prepare_weight(weight_path=weight_file)

            if weight_file.exists():
                self.model.save_load_weights(weight_file)
                logger.info(f"✅ Loaded pretrained weights: {weight_file}")
            else:
                logger.warning(f"⚠️ Could not download weights for {model_config}")
        else:
            # Explicit path provided
            weight_file = Path(weight_path)
            if weight_file.exists():
                self.model.save_load_weights(weight_file)
                logger.info(f"✅ Loaded pretrained weights: {weight_file}")
            else:
                # Try to download if it looks like a model name
                logger.info(f"🌐 Weight {weight_file} not found, attempting download...")
                prepare_weight(weight_path=weight_file)
                if weight_file.exists():
                    self.model.save_load_weights(weight_file)
                    logger.info(f"✅ Loaded pretrained weights: {weight_file}")
                else:
                    logger.warning(f"⚠️ Weight file not found: {weight_file}")

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

    # =========================================================================
    # Layer Freezing Methods
    # =========================================================================

    def _apply_layer_freezing(self) -> None:
        """
        Apply layer freezing based on configuration.

        Freezes backbone layers or specific layer patterns for transfer learning.
        """
        frozen_count = 0
        self._frozen_layers = []

        if self.hparams.freeze_backbone:
            # Freeze backbone layers (typically the feature extractor)
            # In YOLO architectures, backbone is usually the first major section
            for name, param in self.model.named_parameters():
                # Freeze backbone layers (not neck/head)
                if self._is_backbone_layer(name):
                    param.requires_grad = False
                    self._frozen_layers.append(name)
                    frozen_count += 1

        if self.hparams.freeze_layers:
            # Freeze specific layer patterns
            for name, param in self.model.named_parameters():
                for pattern in self.hparams.freeze_layers:
                    if pattern in name:
                        param.requires_grad = False
                        if name not in self._frozen_layers:
                            self._frozen_layers.append(name)
                            frozen_count += 1
                        break

        if frozen_count > 0:
            logger.info(f"🥶 Frozen {frozen_count} layers for transfer learning")

    def _is_backbone_layer(self, layer_name: str) -> bool:
        """
        Determine if a layer belongs to the backbone.

        Args:
            layer_name: Full name of the layer parameter

        Returns:
            True if layer is part of backbone
        """
        # YOLO v9 backbone patterns (typically early layers before neck/head)
        backbone_patterns = [
            "backbone",
            "stem",
            "dark",  # DarkNet layers
            "stage1", "stage2", "stage3",  # Early stages
            "conv1", "conv2", "conv3",  # Early convolutions
            "layer1", "layer2", "layer3",  # ResNet-style layer naming
        ]

        layer_lower = layer_name.lower()

        # Check if layer matches any backbone pattern
        for pattern in backbone_patterns:
            if pattern in layer_lower:
                return True

        # Check layer index - freeze first N% of model
        # This is a fallback for architectures with numeric naming
        parts = layer_name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            layer_idx = int(parts[1])
            # Freeze first ~60% of layers (backbone typically)
            total_layers = len(list(self.model.parameters()))
            if layer_idx < total_layers * 0.6:
                return True

        return False

    def _unfreeze_all_layers(self) -> None:
        """Unfreeze all previously frozen layers."""
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if name in self._frozen_layers:
                param.requires_grad = True
                unfrozen_count += 1

        self._frozen_layers = []

        if unfrozen_count > 0:
            logger.info(f"🔥 Unfrozen {unfrozen_count} layers")

    def on_train_epoch_start(self) -> None:
        """Check if we should unfreeze layers at this epoch."""
        if (
            self.hparams.freeze_until_epoch > 0
            and self.current_epoch >= self.hparams.freeze_until_epoch
            and len(self._frozen_layers) > 0
        ):
            logger.info(
                f"📅 Epoch {self.current_epoch + 1}: Unfreezing layers "
                f"(freeze_until_epoch={self.hparams.freeze_until_epoch})"
            )
            self._unfreeze_all_layers()

            # Reconfigure optimizer with all parameters
            # Note: This is handled automatically by Lightning if needed

    def get_trainable_parameters(self) -> int:
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_frozen_parameters(self) -> int:
        """Get count of frozen parameters."""
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
