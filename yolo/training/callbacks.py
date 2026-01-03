"""
Custom Lightning callbacks for YOLO training.
"""

from typing import Any, Dict, Optional, Union

import lightning as L
from lightning.pytorch.callbacks import Callback, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.console import Console
from rich.table import Table
from rich.text import Text


class YOLOProgressBar(RichProgressBar):
    """
    Custom progress bar that displays training metrics in a clean, compact format.

    Metrics shown: loss | box | cls | dfl | lr
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

    Example output:
    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                            Epoch 5 - Validation Metrics                              │
    ├──────────────┬──────────────┬──────────────┬──────────────┬──────────────────────────┤
    │     mAP      │    mAP50     │    mAP75     │    mAP95     │          mAR100          │
    │    0.4523    │    0.6821    │    0.4912    │    0.2134    │          0.5234          │
    ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────────────────┤
    │   mAP_small  │  mAP_medium  │   mAP_large  │  train/loss  │                          │
    │    0.2134    │    0.4521    │    0.5823    │    2.3456    │                          │
    └──────────────┴──────────────┴──────────────┴──────────────┴──────────────────────────┘
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

        # Build and print the table
        print(self._build_table(epoch, metrics))

    def _build_table(self, epoch: int, metrics: Dict[str, Any]) -> str:
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
        lines.append(center_text(f"Epoch {epoch} - Validation Metrics"))

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
