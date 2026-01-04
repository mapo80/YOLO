"""
Custom Lightning callbacks for YOLO training.
"""

import copy
import math
from typing import Any, Dict, Optional, Union

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from yolo.utils.logger import logger
from yolo.utils.eval_dashboard import EvalDashboard, EvalConfig


class YOLOProgressBar(RichProgressBar):
    """
    Custom progress bar that displays training metrics in a clean, compact format.

    Metrics shown: loss | box | cls | dfl | lr
    Epoch display: 1-indexed (Epoch 1/100 instead of Epoch 0/99)
    Validation progress: Visible during validation phase
    """

    def __init__(self):
        super().__init__(
            leave=True,
            theme=RichProgressBarTheme(
                description="white",
                progress_bar="#6206E0",
                progress_bar_finished="#6206E0",
                progress_bar_pulse="#6206E0",
                batch_progress="white",
                time="grey54",
                processing_speed="grey54",
                metrics="grey70",
            ),
        )

    def _get_train_description(self, current_epoch: int) -> str:
        """Override to display 1-indexed epochs (Epoch 1/100 instead of Epoch 0/99)."""
        train_description = f"Epoch {current_epoch + 1}"
        if self.trainer.max_epochs is not None:
            train_description += f"/{self.trainer.max_epochs}"
        if len(self.validation_description) > len(train_description):
            # Padding to avoid flickering due to uneven lengths
            train_description = f"{train_description:{len(self.validation_description)}}"
        return train_description

    def on_validation_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Override to make validation progress bar visible during training."""
        if self.is_disabled or not self.has_dataloader_changed(dataloader_idx):
            return

        assert self.progress is not None

        if trainer.sanity_checking:
            if self.val_sanity_progress_bar_id is not None:
                self.progress.update(self.val_sanity_progress_bar_id, advance=0, visible=False)

            self.val_sanity_progress_bar_id = self._add_task(
                self.total_val_batches_current_dataloader,
                self.sanity_check_description,
                visible=False,
            )
        else:
            if self.val_progress_bar_id is not None:
                self.progress.update(self.val_progress_bar_id, advance=0, visible=False)

            # Make validation progress bar VISIBLE (unlike default which hides it)
            self.val_progress_bar_id = self._add_task(
                self.total_val_batches_current_dataloader,
                self.validation_description,
                visible=True,  # Show validation progress
            )

        self.refresh()

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Override to keep validation progress bar visible until completion."""
        # Update progress to 100% before hiding
        if self.is_enabled and self.val_progress_bar_id is not None and trainer.state.fn == "fit":
            assert self.progress is not None
            # Show completed state briefly before hiding
            total = self.progress.tasks[self.val_progress_bar_id].total
            if total is not None:
                self.progress.update(self.val_progress_bar_id, completed=total, visible=True)
            self.refresh()
            # Then hide it
            self.progress.update(self.val_progress_bar_id, advance=0, visible=False)
            self.refresh()

    def get_metrics(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> Dict[str, Union[int, str, float, Dict[str, float]]]:
        """Override to customize which metrics are shown and their format."""
        items = super().get_metrics(trainer, pl_module)

        # Remove v_num as it's not useful
        items.pop("v_num", None)

        # Keep only essential metrics with short names
        result = {}
        if "loss" in items:
            result["loss"] = round(float(items["loss"]), 2)
        if "box" in items:
            result["box"] = round(float(items["box"]), 2)
        if "cls" in items:
            result["cls"] = round(float(items["cls"]), 2)
        if "dfl" in items:
            result["dfl"] = round(float(items["dfl"]), 2)
        if "lr" in items:
            result["lr"] = round(float(items["lr"]), 5)

        return result


class MetricsTableCallback(Callback):
    """
    Callback that prints a clean, formatted metrics table after each validation epoch.
    Dynamically adapts to show only the metrics that are being logged.
    Shows if current epoch is best (synchronized with ModelCheckpoint).

    Example output (best epoch):
    ┌─────────────────────────────────────────────────────────────────┐
    │              Epoch 5 - Validation Metrics ★ NEW BEST           │
    │                        mAP: 0.4523 (best)                       │
    ├────────────┬────────────┬────────────┬────────────┬────────────┤
    │    mAP     │   mAP50    │   mAP75    │   mAP95    │   mAR100   │
    │   0.4523   │   0.6821   │   0.4912   │   0.2134   │   0.5234   │
    └────────────┴────────────┴────────────┴────────────┴────────────┘

    Example output (not best):
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Epoch 8 - Validation Metrics                   │
    │                mAP: 0.4312 (-0.0211 vs best @ ep5)              │
    ├────────────┬────────────┬────────────┬────────────┬────────────┤
    │    mAP     │   mAP50    │   mAP75    │   mAP95    │   mAR100   │
    │   0.4312   │   0.6512   │   0.4701   │   0.1998   │   0.5012   │
    └────────────┴────────────┴────────────┴────────────┴────────────┘
    """

    def __init__(self):
        super().__init__()
        self._last_train_loss: Optional[float] = None

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Capture the latest training loss."""
        if "train/loss" in trainer.callback_metrics:
            self._last_train_loss = trainer.callback_metrics["train/loss"].item()

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Print metrics table at the end of training epoch (after validation)."""
        # Skip if no metrics yet
        if not trainer.callback_metrics:
            return

        # Skip during sanity check
        if trainer.sanity_checking:
            return

        # Only print if we have validation metrics
        if "val/mAP" not in trainer.callback_metrics:
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # Get best info from ModelCheckpoint (synchronized with actual checkpointing)
        current_score = None
        best_score = None
        best_epoch = None
        is_best = False

        # Find ModelCheckpoint callback that monitors val/mAP
        ckpt_callback = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.monitor == "val/mAP":
                ckpt_callback = cb
                break

        if ckpt_callback is not None and "val/mAP" in metrics:
            val = metrics["val/mAP"]
            current_score = val.item() if hasattr(val, "item") else val

            if current_score != -1:  # -1 means metric not available
                # Get best from checkpoint callback
                if ckpt_callback.best_model_score is not None:
                    best_score = ckpt_callback.best_model_score.item() if hasattr(ckpt_callback.best_model_score, "item") else ckpt_callback.best_model_score

                    # Check if current is best: must be >= best AND actually improved
                    # (current >= best AND (current > best OR this is the first non-zero improvement))
                    is_best = current_score >= best_score and (
                        current_score > best_score + 1e-6 or  # Actually improved
                        (current_score == best_score and best_epoch is None)  # First epoch with this score
                    )

                    # Extract best epoch from best_model_path if available
                    if ckpt_callback.best_model_path:
                        import re
                        match = re.search(r'epoch=(\d+)', ckpt_callback.best_model_path)
                        if match:
                            best_epoch = int(match.group(1))
                            # If we have best_epoch and score is same, not a new best
                            if current_score <= best_score:
                                is_best = False
                else:
                    # First validation, no best yet - this will be best only if score > 0
                    is_best = current_score > 0
                    best_score = current_score

        # Build and print the table
        print(self._build_table(epoch, metrics, is_best, current_score, best_score, best_epoch))

    def _build_table(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
        current_score: Optional[float] = None,
        best_score: Optional[float] = None,
        best_epoch: Optional[int] = None,
    ) -> str:
        """Build a formatted metrics table dynamically based on available metrics."""
        col_width = 12

        def format_val(key: str) -> str:
            if key not in metrics:
                return "—"
            val = metrics[key]
            if hasattr(val, "item"):
                val = val.item()
            if val == -1:
                return "—"
            return f"{val:.4f}"

        # Define metric groups with (display_name, metric_key)
        # Row 1: Primary mAP metrics (based on what's logged)
        primary_metrics = []
        if "val/mAP" in metrics:
            primary_metrics.append(("mAP", "val/mAP"))
        if "val/mAP50" in metrics:
            primary_metrics.append(("mAP50", "val/mAP50"))
        if "val/mAP75" in metrics:
            primary_metrics.append(("mAP75", "val/mAP75"))

        # Row 2: Precision, Recall, F1 + loss (if logged)
        secondary_metrics = []
        if "val/precision" in metrics:
            secondary_metrics.append(("Prec", "val/precision"))
        if "val/recall" in metrics:
            secondary_metrics.append(("Recall", "val/recall"))
        if "val/f1" in metrics:
            secondary_metrics.append(("F1", "val/f1"))
        # Add train loss
        if self._last_train_loss is not None:
            secondary_metrics.append(("loss", "_train_loss"))

        # Row 3: Legacy mAP per size (if logged)
        tertiary_metrics = []
        if "val/mAP_small" in metrics:
            tertiary_metrics.append(("mAP_sm", "val/mAP_small"))
        if "val/mAP_medium" in metrics:
            tertiary_metrics.append(("mAP_md", "val/mAP_medium"))
        if "val/mAP_large" in metrics:
            tertiary_metrics.append(("mAP_lg", "val/mAP_large"))

        # Determine number of columns (max of all rows)
        num_cols = max(len(primary_metrics), len(secondary_metrics), len(tertiary_metrics), 1)
        table_width = (col_width + 1) * num_cols + 1

        # Helper functions
        def h_line(left: str, mid: str, right: str, n: int) -> str:
            segment = "─" * col_width
            return left + mid.join([segment] * n) + right

        def center_text(text: str) -> str:
            return f"│{text.center(table_width - 2)}│"

        def make_row(items: list, pad_to: int) -> str:
            cells = []
            for name, key in items:
                if key == "_train_loss":
                    val = f"{self._last_train_loss:.4f}" if self._last_train_loss else "—"
                else:
                    val = format_val(key)
                cells.append(f"{val:^{col_width}}")
            # Pad with empty cells if needed
            while len(cells) < pad_to:
                cells.append(" " * col_width)
            return "│" + "│".join(cells) + "│"

        def make_header_row(items: list, pad_to: int) -> str:
            cells = []
            for name, _ in items:
                cells.append(f"{name:^{col_width}}")
            while len(cells) < pad_to:
                cells.append(" " * col_width)
            return "│" + "│".join(cells) + "│"

        # Build table
        lines = []

        # Top border
        lines.append(f"┌{'─' * (table_width - 2)}┐")

        # Title with best indicator (display 1-indexed epochs)
        display_epoch = epoch + 1
        if is_best:
            title = f"Epoch {display_epoch} - Validation Metrics ★ NEW BEST"
        else:
            title = f"Epoch {display_epoch} - Validation Metrics"
        lines.append(center_text(title))

        # Show improvement/comparison line
        if current_score is not None:
            if is_best:
                delta_str = f"mAP: {current_score:.4f} (best)"
            elif best_score is not None and best_epoch is not None:
                delta_from_best = current_score - best_score
                # Display best_epoch as 1-indexed too
                delta_str = f"mAP: {current_score:.4f} ({delta_from_best:+.4f} vs best @ ep{best_epoch + 1})"
            elif best_score is not None:
                delta_from_best = current_score - best_score
                delta_str = f"mAP: {current_score:.4f} ({delta_from_best:+.4f} vs best)"
            else:
                delta_str = f"mAP: {current_score:.4f}"
            lines.append(center_text(delta_str))

        # Primary metrics row
        if primary_metrics:
            lines.append(h_line("├", "┬", "┤", num_cols))
            lines.append(make_header_row(primary_metrics, num_cols))
            lines.append(make_row(primary_metrics, num_cols))

        # Secondary metrics row (mAP per size)
        if secondary_metrics:
            lines.append(h_line("├", "┼", "┤", num_cols))
            lines.append(make_header_row(secondary_metrics, num_cols))
            lines.append(make_row(secondary_metrics, num_cols))

        # Tertiary metrics row (mAR per size)
        if tertiary_metrics:
            lines.append(h_line("├", "┼", "┤", num_cols))
            lines.append(make_header_row(tertiary_metrics, num_cols))
            lines.append(make_row(tertiary_metrics, num_cols))

        # Bottom border
        lines.append(h_line("└", "┴", "┘", num_cols))

        return "\n" + "\n".join(lines)


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of model weights that is updated with EMA at each
    training step. The EMA model typically achieves better accuracy than the
    final training weights.

    The decay rate ramps up during warmup to avoid overwriting initial weights
    too slowly when the model is learning rapidly.

    Formula:
        effective_decay = decay * (1 - exp(-updates / tau))
        ema_weights = effective_decay * ema_weights + (1 - effective_decay) * model_weights

    Args:
        model: Model to track
        decay: Base EMA decay rate (higher = slower update, more smoothing)
        tau: Warmup steps for decay ramping (higher = longer warmup)
        updates: Initial update count (for checkpoint resume)

    Example decay progression (decay=0.9999, tau=2000):
        updates=0    -> effective_decay=0.000
        updates=1000 -> effective_decay=0.393
        updates=2000 -> effective_decay=0.632
        updates=5000 -> effective_decay=0.918
        updates=10000 -> effective_decay=0.993
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        tau: float = 2000.0,
        updates: int = 0,
    ):
        # Create deep copy of model for EMA
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.tau = tau
        self.updates = updates

        # Disable gradients for EMA model
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model: torch.nn.Module) -> None:
        """
        Update EMA weights with current model weights.

        Should be called after each optimizer step.
        """
        self.updates += 1

        # Calculate effective decay with warmup
        d = self.decay * (1 - math.exp(-self.updates / self.tau))

        # Get model state dict (handle DDP wrapper)
        model_state = model.state_dict()

        # Update EMA weights
        with torch.no_grad():
            for name, ema_param in self.ema.state_dict().items():
                if name in model_state:
                    model_param = model_state[name]
                    # Only update float parameters (skip buffers like running_mean)
                    if ema_param.dtype.is_floating_point:
                        ema_param.mul_(d).add_(model_param, alpha=1 - d)

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            "ema_state_dict": self.ema.state_dict(),
            "decay": self.decay,
            "tau": self.tau,
            "updates": self.updates,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.ema.load_state_dict(state_dict["ema_state_dict"])
        self.decay = state_dict["decay"]
        self.tau = state_dict["tau"]
        self.updates = state_dict["updates"]


class EMACallback(Callback):
    """
    Lightning callback for Exponential Moving Average of model weights.

    Maintains a shadow copy of model weights that is updated with EMA at each
    training step. Uses EMA weights for validation and saves them to checkpoints.

    Features:
    - Updates EMA after each training batch
    - Swaps to EMA weights for validation (typically better metrics)
    - Automatically saves/loads EMA state with checkpoints
    - Fully configurable decay and warmup
    - Can be disabled without code changes

    Args:
        decay: EMA decay rate. Higher values = more smoothing.
            Typical values: 0.9999 (default), 0.999 (faster updates)
        tau: Warmup steps for decay ramping. The effective decay starts at 0
            and ramps up to `decay` over approximately `tau` steps.
        enabled: Set to False to completely disable EMA. Useful for quick
            experimentation or when EMA is not beneficial.

    Example configuration (YAML):
        trainer:
          callbacks:
            - class_path: yolo.training.callbacks.EMACallback
              init_args:
                decay: 0.9999
                tau: 2000
                enabled: true

    Example CLI override:
        # Disable EMA
        --trainer.callbacks.X.init_args.enabled=false

        # Adjust decay
        --trainer.callbacks.X.init_args.decay=0.999
    """

    def __init__(
        self,
        decay: float = 0.9999,
        tau: float = 2000.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.decay = decay
        self.tau = tau
        self.enabled = enabled

        # Runtime state
        self._ema: Optional[ModelEMA] = None
        self._original_state_dict: Optional[Dict[str, Any]] = None

    def on_fit_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Initialize EMA model at the start of training."""
        if not self.enabled:
            return

        # Initialize EMA with current model weights
        self._ema = ModelEMA(
            model=pl_module,
            decay=self.decay,
            tau=self.tau,
            updates=0,
        )
        logger.info(
            f"EMA initialized (decay={self.decay}, tau={self.tau})"
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA weights after each training step."""
        if not self.enabled or self._ema is None:
            return

        self._ema.update(pl_module)

    def on_validation_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Swap to EMA weights for validation."""
        if not self.enabled or self._ema is None:
            return

        # Save original weights
        self._original_state_dict = copy.deepcopy(pl_module.state_dict())

        # Load EMA weights into model
        pl_module.load_state_dict(self._ema.ema.state_dict())

    def on_validation_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Restore original weights after validation."""
        if not self.enabled or self._original_state_dict is None:
            return

        # Restore original training weights
        pl_module.load_state_dict(self._original_state_dict)
        self._original_state_dict = None

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Save EMA state to checkpoint."""
        if not self.enabled or self._ema is None:
            return

        checkpoint["ema_callback"] = {
            "ema_state": self._ema.state_dict(),
            "decay": self.decay,
            "tau": self.tau,
        }

    def on_load_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Load EMA state from checkpoint."""
        if not self.enabled:
            return

        if "ema_callback" not in checkpoint:
            return

        ema_data = checkpoint["ema_callback"]

        # Initialize EMA if not yet created
        if self._ema is None:
            self._ema = ModelEMA(
                model=pl_module,
                decay=ema_data.get("decay", self.decay),
                tau=ema_data.get("tau", self.tau),
                updates=0,
            )

        # Load saved EMA state
        if "ema_state" in ema_data:
            self._ema.load_state_dict(ema_data["ema_state"])
            logger.info(
                f"EMA restored from checkpoint (updates={self._ema.updates})"
            )


class TrainingSummaryCallback(Callback):
    """
    Callback that displays a training summary table before training starts.

    Shows organized sections:
    - Output Directories (Checkpoints, Logs, Metrics)
    - Model (Architecture, Classes, Image Size, Weights)
    - Dataset (Format, Root, Batch Size, Workers)
    - Training (Epochs, Device, Precision, Optimizer, LR, Scheduler)
    """

    def on_fit_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Display training summary before training starts."""
        from pathlib import Path
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()

        # Create table with unified format
        table = Table(
            title="Training Configuration",
            title_style="bold cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Setting", style="bold white", width=24, no_wrap=True)
        table.add_column("Value", style="green", overflow="fold")

        # === Output Directories Section ===
        table.add_row("[bold yellow]Output Directories[/]", "")

        # Checkpoint directory
        ckpt_dir = self._get_checkpoint_dir(trainer)
        table.add_row("  Checkpoints", str(ckpt_dir) if ckpt_dir else "N/A")

        # Log directory
        log_dir = self._get_log_dir(trainer)
        table.add_row("  Logs", str(log_dir) if log_dir else "N/A")

        # Metrics plots directory
        metrics_dir = self._get_metrics_dir(trainer, pl_module)
        table.add_row("  Metrics", str(metrics_dir) if metrics_dir else "Disabled")

        # === Model Section ===
        table.add_row("", "")  # Empty row for spacing
        table.add_row("[bold yellow]Model[/]", "")

        if hasattr(pl_module, "hparams"):
            hparams = pl_module.hparams
            if hasattr(hparams, "model_config"):
                table.add_row("  Architecture", str(hparams.model_config))
            if hasattr(hparams, "num_classes"):
                table.add_row("  Classes", str(hparams.num_classes))
            if hasattr(hparams, "image_size"):
                img_sz = hparams.image_size
                if isinstance(img_sz, (list, tuple)):
                    table.add_row("  Image Size", f"{img_sz[0]}x{img_sz[1]}")
                else:
                    table.add_row("  Image Size", f"{img_sz}x{img_sz}")
            # Weights info
            weight_path = getattr(hparams, "weight_path", None)
            if weight_path is True:
                table.add_row("  Weights", "Pretrained (auto-download)")
            elif weight_path:
                table.add_row("  Weights", str(weight_path))
            else:
                table.add_row("  Weights", "From scratch")

        # === Dataset Section ===
        table.add_row("", "")
        table.add_row("[bold yellow]Dataset[/]", "")

        if trainer.datamodule and hasattr(trainer.datamodule, "hparams"):
            dm_hparams = trainer.datamodule.hparams
            # Format
            data_format = getattr(dm_hparams, "format", "coco")
            table.add_row("  Format", data_format.upper())
            # Root
            if hasattr(dm_hparams, "root"):
                table.add_row("  Root", str(dm_hparams.root))
            # Batch size
            if hasattr(dm_hparams, "batch_size"):
                table.add_row("  Batch Size", str(dm_hparams.batch_size))
            # Workers
            if hasattr(dm_hparams, "num_workers"):
                table.add_row("  Workers", str(dm_hparams.num_workers))

        # === Training Section ===
        table.add_row("", "")
        table.add_row("[bold yellow]Training[/]", "")

        table.add_row("  Max Epochs", str(trainer.max_epochs))

        # Device
        device = self._get_device_name(trainer)
        table.add_row("  Device", device)

        # Precision
        precision = self._get_precision_name(trainer)
        table.add_row("  Precision", precision)

        if hasattr(pl_module, "hparams"):
            hparams = pl_module.hparams
            # Optimizer
            if hasattr(hparams, "optimizer"):
                table.add_row("  Optimizer", hparams.optimizer.upper())
            # Learning rate
            if hasattr(hparams, "learning_rate"):
                table.add_row("  Learning Rate", str(hparams.learning_rate))
            # LR Scheduler
            if hasattr(hparams, "lr_scheduler"):
                table.add_row("  LR Scheduler", hparams.lr_scheduler)

        console.print()
        console.print(table)
        console.print()

    def _get_checkpoint_dir(self, trainer: L.Trainer) -> Optional[str]:
        """Get checkpoint directory from ModelCheckpoint callback."""
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                if cb.dirpath:
                    return cb.dirpath
        if trainer.log_dir:
            from pathlib import Path
            return str(Path(trainer.log_dir) / "checkpoints")
        return None

    def _get_log_dir(self, trainer: L.Trainer) -> Optional[str]:
        """Get log directory."""
        if trainer.log_dir:
            return trainer.log_dir
        if trainer.logger and hasattr(trainer.logger, "log_dir"):
            return trainer.logger.log_dir
        return None

    def _get_metrics_dir(self, trainer: L.Trainer, pl_module: L.LightningModule) -> Optional[str]:
        """Get metrics plots directory."""
        if hasattr(pl_module, "hparams"):
            if getattr(pl_module.hparams, "save_metrics_plots", False):
                if getattr(pl_module.hparams, "metrics_plots_dir", None):
                    return pl_module.hparams.metrics_plots_dir
                elif trainer.log_dir:
                    from pathlib import Path
                    return str(Path(trainer.log_dir) / "metrics")
        return None

    def _get_device_name(self, trainer: L.Trainer) -> str:
        """Get clean device name."""
        accelerator = trainer.accelerator
        if accelerator:
            acc_name = accelerator.__class__.__name__
            if acc_name.endswith("Accelerator"):
                acc_name = acc_name[:-11]
            return acc_name.upper()
        return "CPU"

    def _get_precision_name(self, trainer: L.Trainer) -> str:
        """Get clean precision name."""
        precision = str(trainer.precision)
        if precision == "32-true" or precision == "32":
            return "FP32"
        elif precision == "16-mixed":
            return "FP16 Mixed"
        elif precision == "bf16-mixed":
            return "BF16 Mixed"
        return precision


class ClassNamesCallback(Callback):
    """
    Callback that automatically loads class names from dataset and passes them to model.

    This ensures class names are available for metrics display during training,
    showing actual class names instead of indices in the eval dashboard.
    """

    def __init__(self):
        super().__init__()
        self._class_names_set = False

    def on_fit_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Load class names from datamodule at the start of training."""
        self._update_metrics_names(trainer, pl_module)

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Ensure class names are set after metrics are created."""
        if not self._class_names_set:
            self._update_metrics_names(trainer, pl_module)
            self._class_names_set = True

    def on_validation_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Ensure class names are set for validation."""
        self._update_metrics_names(trainer, pl_module)

    def _update_metrics_names(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Update metrics with class names from datamodule."""
        # Get class names from datamodule
        class_names = None
        if trainer.datamodule and hasattr(trainer.datamodule, "class_names"):
            class_names = trainer.datamodule.class_names

        if not class_names:
            return

        # Update detection metrics if available
        if hasattr(pl_module, "_det_metrics") and pl_module._det_metrics is not None:
            pl_module._det_metrics.names = class_names


class EvalDashboardCallback(Callback):
    """
    Callback that displays comprehensive eval dashboard after each validation epoch.

    Shows:
    - Quality metrics (mAP, AP50, AP75, AR@100, size-based metrics)
    - Operative metrics at production confidence threshold
    - Trend sparklines (last N epochs)
    - Threshold sweep table
    - Per-class TOP/WORST classes
    - Error health check (FP, FN, confusions)

    Example usage in config:
        trainer:
          callbacks:
            - class_path: yolo.training.callbacks.EvalDashboardCallback
              init_args:
                conf_prod: 0.25
                show_trends: true
                top_n_classes: 3
    """

    def __init__(
        self,
        conf_prod: float = 0.25,
        show_trends: bool = True,
        top_n_classes: int = 3,
    ):
        """
        Initialize EvalDashboardCallback.

        Args:
            conf_prod: Production confidence threshold for operative metrics
            show_trends: Whether to show sparkline trends
            top_n_classes: Number of top/worst classes to display
        """
        super().__init__()
        config = EvalConfig(
            conf_prod=conf_prod,
            show_trends=show_trends,
            top_n_classes=top_n_classes,
        )
        self.dashboard = EvalDashboard(config)
        self.conf_prod = conf_prod

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Display dashboard at the end of training epoch (after validation)."""
        # Skip if no metrics yet
        if not trainer.callback_metrics:
            return

        # Skip during sanity check
        if trainer.sanity_checking:
            return

        # Only print if we have validation metrics
        if "val/mAP" not in trainer.callback_metrics:
            return

        # Get extended validation metrics from module
        metrics = getattr(pl_module, "_last_validation_metrics", None)
        if metrics is None:
            return

        # Get number of validation images
        num_images = 0
        if trainer.val_dataloaders is not None:
            try:
                num_images = len(trainer.val_dataloaders.dataset)
            except (AttributeError, TypeError):
                pass

        # Get image size from module if available
        image_size = (640, 640)
        if hasattr(pl_module, "hparams"):
            if hasattr(pl_module.hparams, "image_size"):
                img_sz = pl_module.hparams.image_size
                if isinstance(img_sz, int):
                    image_size = (img_sz, img_sz)
                elif isinstance(img_sz, (list, tuple)) and len(img_sz) >= 2:
                    image_size = (img_sz[0], img_sz[1])

        # Get run ID from logger
        run_id = ""
        if trainer.logger is not None:
            run_id = getattr(trainer.logger, "name", "") or ""

        self.dashboard.print(
            metrics=metrics,
            epoch=trainer.current_epoch + 1,
            total_epochs=trainer.max_epochs,
            run_id=run_id,
            num_images=num_images,
            image_size=image_size,
        )
