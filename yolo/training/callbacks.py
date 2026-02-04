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
        logger.debug("[MetricsTable] on_train_epoch_end started")
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
        logger.debug("[MetricsTable] Building table...")
        print(self._build_table(epoch, metrics, is_best, current_score, best_score, best_epoch))
        logger.debug("[MetricsTable] on_train_epoch_end completed")

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


def _ema_lerp(start: torch.Tensor, end: torch.Tensor, weight: float) -> torch.Tensor:
    """
    Linear interpolation: start + (end - start) * weight
    EXACTLY like yolo-original lerp function.
    """
    return start + (end - start) * weight


class ModelEMA:
    """
    Exponential Moving Average - EXACT COPY of yolo-original/yolo/utils/model_utils.py EMA class.

    Key differences from previous implementations:
    1. ema_state_dict initialized in on_validation_start (not on first update)
    2. Uses lerp formula: model + (ema - model) * decay
    3. batch_step_counter for gradient accumulation tracking
    4. NO .clone() calls - yolo-original doesn't use them
    """

    def __init__(
        self,
        decay: float = 0.9999,
        tau: float = 2000.0,
    ):
        self.decay = decay
        self.tau = tau
        self.step = 0
        self.batch_step_counter = 0
        self.ema_state_dict: Optional[Dict[str, Any]] = None

    def setup(self, model: torch.nn.Module, ema_model: torch.nn.Module, world_size: int = 1) -> None:
        """Setup EMA - call this in on_fit_start."""
        # Store reference to EMA model (created by callback)
        self.ema_model = ema_model
        # Adjust tau for world size like yolo-original
        self.tau = self.tau / world_size

    def init_ema_state_dict(self, model: torch.nn.Module) -> None:
        """Initialize ema_state_dict - called in on_validation_start like yolo-original."""
        if self.ema_state_dict is None:
            self.ema_state_dict = copy.deepcopy(model.state_dict())

    def load_to_ema_model(self) -> None:
        """Load ema_state_dict to ema_model - called in on_validation_start."""
        if self.ema_state_dict is not None and self.ema_model is not None:
            self.ema_model.load_state_dict(self.ema_state_dict)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, accumulate_grad_batches: int) -> None:
        """
        Update EMA weights - EXACTLY like yolo-original on_train_batch_end.

        yolo-original code (line 69-76):
            self.batch_step_counter += 1
            if self.batch_step_counter % trainer.accumulate_grad_batches:
                return
            self.step += 1
            decay_factor = self.decay * (1 - exp(-self.step / self.tau))
            for key, param in pl_module.model.state_dict().items():
                self.ema_state_dict[key] = lerp(param.detach(), self.ema_state_dict[key], decay_factor)
        """
        self.batch_step_counter += 1

        # Skip if not an optimizer step (gradient accumulation)
        if self.batch_step_counter % accumulate_grad_batches:
            return

        self.step += 1
        decay_factor = self.decay * (1 - math.exp(-self.step / self.tau))

        # Update EMA state dict - EXACTLY like yolo-original
        # lerp(model, ema, decay) = model + (ema - model) * decay
        if self.ema_state_dict is not None:
            for key, param in model.state_dict().items():
                # Ensure EMA tensor is on same device as model param (fix resume from checkpoint)
                ema_val = self.ema_state_dict[key]
                if ema_val.device != param.device:
                    ema_val = ema_val.to(param.device)
                self.ema_state_dict[key] = _ema_lerp(param.detach(), ema_val, decay_factor)

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            "ema_state_dict": self.ema_state_dict,
            "decay": self.decay,
            "tau": self.tau,
            "step": self.step,
            "batch_step_counter": self.batch_step_counter,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.ema_state_dict = state_dict["ema_state_dict"]
        self.decay = state_dict["decay"]
        self.tau = state_dict["tau"]
        self.step = state_dict["step"]
        self.batch_step_counter = state_dict["batch_step_counter"]
        if self.ema_model is not None and self.ema_state_dict is not None:
            self.ema_model.load_state_dict(self.ema_state_dict)


class EMACallback(Callback):
    """
    Lightning callback for Exponential Moving Average - EXACT replica of yolo-original.

    This is a direct port of yolo-original/yolo/utils/model_utils.py EMA class
    to work with Lightning callbacks.

    Key behaviors (matching yolo-original exactly):
    1. setup(): Creates pl_module.ema as deepcopy of pl_module.model
    2. on_validation_start(): Initializes ema_state_dict if None, loads to ema model
    3. on_train_batch_end(): Updates ema_state_dict using lerp formula
    4. NO weight swapping for validation - yolo-original doesn't do it either

    Args:
        decay: EMA decay rate (default 0.9999)
        tau: Warmup steps for decay ramping (default 2000)
        enabled: Set to False to disable EMA
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

        # Runtime state - exactly like yolo-original
        self._ema: Optional[ModelEMA] = None
        self._ema_model: Optional[torch.nn.Module] = None

    def setup(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        stage: str,
    ) -> None:
        """
        Setup EMA - EXACTLY like yolo-original setup().

        yolo-original code:
            pl_module.ema = deepcopy(pl_module.model)
            self.tau /= trainer.world_size
            for param in pl_module.ema.parameters():
                param.requires_grad = False
        """
        if not self.enabled or stage != "fit":
            return

        # Skip if already setup (e.g., from checkpoint restore)
        if self._ema is not None:
            return

        # Create EMA model exactly like yolo-original
        self._ema_model = copy.deepcopy(pl_module.model)
        for param in self._ema_model.parameters():
            param.requires_grad = False

        # Store reference on pl_module like yolo-original does
        pl_module.ema = self._ema_model

        # Initialize EMA tracker
        self._ema = ModelEMA(decay=self.decay, tau=self.tau)
        self._ema.setup(pl_module.model, self._ema_model, trainer.world_size)

        logger.info(f"EMA initialized (decay={self.decay}, tau={self._ema.tau})")

    def on_validation_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """
        Initialize ema_state_dict and load to ema model - EXACTLY like yolo-original.

        yolo-original code:
            self.batch_step_counter = 0
            if self.ema_state_dict is None:
                self.ema_state_dict = deepcopy(pl_module.model.state_dict())
            pl_module.ema.load_state_dict(self.ema_state_dict)
        """
        if not self.enabled or self._ema is None:
            return

        # Reset batch counter like yolo-original
        self._ema.batch_step_counter = 0

        # Initialize ema_state_dict if needed (like yolo-original)
        self._ema.init_ema_state_dict(pl_module.model)

        # Load ema_state_dict to ema model
        self._ema.load_to_ema_model()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Update EMA weights - EXACTLY like yolo-original on_train_batch_end.

        The gradient accumulation check is handled inside self._ema.update().
        """
        if not self.enabled or self._ema is None:
            return

        self._ema.update(pl_module.model, trainer.accumulate_grad_batches)

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Save EMA state to checkpoint."""
        if not self.enabled or self._ema is None:
            return

        checkpoint["ema_callback"] = self._ema.state_dict()

    def on_load_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Load EMA state from checkpoint."""
        if not self.enabled:
            return

        if "ema_callback" in checkpoint:
            # Create EMA model if not exists
            if self._ema_model is None:
                self._ema_model = copy.deepcopy(pl_module.model)
                for param in self._ema_model.parameters():
                    param.requires_grad = False
                pl_module.ema = self._ema_model

            # Create and restore EMA tracker
            if self._ema is None:
                self._ema = ModelEMA(decay=self.decay, tau=self.tau)
                self._ema.ema_model = self._ema_model

            self._ema.load_state_dict(checkpoint["ema_callback"])
            logger.info(f"EMA restored from checkpoint (step={self._ema.step})")


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
            # Data fraction (if not 100%)
            data_fraction = getattr(dm_hparams, "data_fraction", 1.0)
            if data_fraction < 1.0:
                table.add_row("  Data Fraction", f"{data_fraction * 100:.0f}%")
            # Caching
            cache_images = getattr(dm_hparams, "cache_images", "none")
            if cache_images != "none":
                cache_resize = getattr(dm_hparams, "cache_resize_images", True)
                cache_info = cache_images.upper()
                if cache_resize:
                    cache_info += " (resized)"
                table.add_row("  Image Cache", cache_info)

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

        # Progress bar legend with target values
        legend = Table(
            title="Progress Bar Metrics",
            title_style="bold cyan",
            box=box.SIMPLE,
            padding=(0, 1),
            expand=False,
        )
        legend.add_column("Metric", style="bold white", width=8)
        legend.add_column("Description", style="dim", width=28)
        legend.add_column("Target", style="green", width=18)
        legend.add_row("loss", "Total loss (box + cls + dfl)", "< 3.0 (converged)")
        legend.add_row("box", "Bounding box regression", "< 0.5 (good fit)")
        legend.add_row("cls", "Classification loss", "< 1.0 (learned)")
        legend.add_row("dfl", "Distribution focal loss", "< 1.0 (stable)")
        legend.add_row("lr", "Learning rate", "Decays over epochs")
        console.print(legend)
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
        logger.debug("[EvalDashboard] on_train_epoch_end started")
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

            # Update dashboard config with nms_iou from module hparams
            if hasattr(pl_module.hparams, "nms_iou_threshold"):
                self.dashboard.config.nms_iou = pl_module.hparams.nms_iou_threshold

        # Get run ID from logger
        run_id = ""
        if trainer.logger is not None:
            run_id = getattr(trainer.logger, "name", "") or ""

        logger.debug("[EvalDashboard] Printing dashboard...")
        self.dashboard.print(
            metrics=metrics,
            epoch=trainer.current_epoch + 1,
            total_epochs=trainer.max_epochs,
            run_id=run_id,
            num_images=num_images,
            image_size=image_size,
        )
        logger.debug("[EvalDashboard] on_train_epoch_end completed")
