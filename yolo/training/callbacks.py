"""
Custom Lightning callbacks for YOLO training.
"""

from typing import Any, Dict, Optional, Union

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.console import Console
from rich.table import Table
from rich.text import Text


class YOLOProgressBar(RichProgressBar):
    """
    Custom progress bar that displays training metrics in a clean, compact format.

    Metrics shown: loss | box | cls | dfl | lr
    Epoch display: 1-indexed (Epoch 1/100 instead of Epoch 0/99)
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

                    # Check if current is best (within floating point tolerance)
                    is_best = abs(current_score - best_score) < 1e-6

                    # Extract best epoch from best_model_path if available
                    if ckpt_callback.best_model_path:
                        import re
                        match = re.search(r'epoch=(\d+)', ckpt_callback.best_model_path)
                        if match:
                            best_epoch = int(match.group(1))
                else:
                    # First validation, no best yet - this will be best
                    is_best = True
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
        if "val/mAP95" in metrics:
            primary_metrics.append(("mAP95", "val/mAP95"))
        if "val/mAR100" in metrics:
            primary_metrics.append(("mAR100", "val/mAR100"))

        # Row 2: mAP per size + loss (if logged)
        secondary_metrics = []
        if "val/mAP_small" in metrics:
            secondary_metrics.append(("mAP_sm", "val/mAP_small"))
        if "val/mAP_medium" in metrics:
            secondary_metrics.append(("mAP_md", "val/mAP_medium"))
        if "val/mAP_large" in metrics:
            secondary_metrics.append(("mAP_lg", "val/mAP_large"))
        # Add train loss
        if self._last_train_loss is not None:
            secondary_metrics.append(("loss", "_train_loss"))

        # Row 3: mAR per size (if logged)
        tertiary_metrics = []
        if "val/mAR_small" in metrics:
            tertiary_metrics.append(("mAR_sm", "val/mAR_small"))
        if "val/mAR_medium" in metrics:
            tertiary_metrics.append(("mAR_md", "val/mAR_medium"))
        if "val/mAR_large" in metrics:
            tertiary_metrics.append(("mAR_lg", "val/mAR_large"))

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
