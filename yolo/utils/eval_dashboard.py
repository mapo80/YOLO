"""
Eval Dashboard - Unified table visualization for detection metrics.

Provides comprehensive metrics display in console/terminal format
for quick decision making during training and validation.

Uses Rich library for beautiful terminal output with unified table format.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich import box


@dataclass
class EvalConfig:
    """Configuration for eval dashboard."""

    conf_prod: float = 0.25
    nms_iou: float = 0.65
    max_det: int = 300
    top_n_classes: int = 3
    show_trends: bool = True
    trend_window: int = 10


class MetricsHistory:
    """Tracks metrics across epochs for trend visualization."""

    def __init__(self, window: int = 10):
        self.window = window
        self.history: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(float(value))
            if len(self.history[key]) > self.window:
                self.history[key] = self.history[key][-self.window:]

    def sparkline(self, key: str) -> str:
        if key not in self.history or len(self.history[key]) < 2:
            return "----------"

        values = self.history[key]
        blocks = " _.,~'^"

        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return "=" * len(values)

        spark = ""
        for v in values:
            idx = int((v - min_val) / (max_val - min_val) * (len(blocks) - 1))
            spark += blocks[idx]

        return spark

    def get_range(self, key: str) -> Tuple[float, float]:
        if key not in self.history or not self.history[key]:
            return (0.0, 0.0)
        return (min(self.history[key]), max(self.history[key]))


class EvalDashboard:
    """Dashboard for eval metrics visualization using unified Rich table."""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.history = MetricsHistory(self.config.trend_window)
        self.best_map = 0.0
        self.best_epoch = 0
        self.console = Console()

    def render(
        self,
        metrics: Dict[str, Any],
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        run_id: str = "",
        num_images: int = 0,
        image_size: Tuple[int, int] = (640, 640),
        latency_ms: Optional[float] = None,
        memory_mb: Optional[float] = None,
        model_size_mb: Optional[float] = None,
    ) -> str:
        """Render dashboard (for compatibility). Use print() instead."""
        trend_metrics = {
            "map": metrics.get("map", 0),
            "map50": metrics.get("map50", 0),
            "ar_100": metrics.get("ar_100", 0),
        }
        self.history.update(trend_metrics)

        current_map = metrics.get("map", 0)
        is_best = current_map > self.best_map
        if is_best and epoch is not None:
            self.best_map = current_map
            self.best_epoch = epoch

        return f"mAP: {metrics.get('map', 0):.4f} | mAP50: {metrics.get('map50', 0):.4f}"

    def print(
        self,
        metrics: Dict[str, Any],
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        run_id: str = "",
        num_images: int = 0,
        image_size: Tuple[int, int] = (640, 640),
        latency_ms: Optional[float] = None,
        memory_mb: Optional[float] = None,
        model_size_mb: Optional[float] = None,
    ) -> None:
        """Print unified dashboard to console."""
        # Update history for trends
        trend_metrics = {
            "map": metrics.get("map", 0),
            "map50": metrics.get("map50", 0),
            "ar_100": metrics.get("ar_100", 0),
        }
        self.history.update(trend_metrics)

        # Track best model
        current_map = metrics.get("map", 0)
        previous_best = self.best_map
        is_best = current_map > self.best_map

        if previous_best > 0:
            delta_from_best = current_map - previous_best
        else:
            delta_from_best = None

        if is_best and epoch is not None:
            self.best_map = current_map
            self.best_epoch = epoch

        # Build title
        title_parts = []
        if epoch and total_epochs:
            title_parts.append(f"Epoch {epoch}/{total_epochs}")
        title_parts.append("Validation Metrics")
        if is_best:
            title_parts.append("[bold green]NEW BEST[/bold green]")

        title = " - ".join(title_parts)

        # Create unified table
        table = Table(
            title=title,
            title_style="bold cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Metric", style="bold white", width=24, no_wrap=True)
        table.add_column("Value", style="green", overflow="fold")

        # === Quality Metrics Section ===
        table.add_row("[bold yellow]Quality Metrics[/]", "")

        # mAP with delta
        map_str = f"{current_map:.4f}"
        if is_best:
            map_str += " (best)"
        elif delta_from_best is not None:
            if delta_from_best >= 0:
                map_str += f" [green](+{delta_from_best:.4f})[/green]"
            else:
                map_str += f" [red]({delta_from_best:.4f})[/red]"
        table.add_row("  mAP (AP50-95)", map_str)

        table.add_row("  mAP50", f"{metrics.get('map50', 0):.4f}")
        table.add_row("  mAP75", f"{metrics.get('map75', 0):.4f}")
        table.add_row("  AR@100", f"{metrics.get('ar_100', 0):.4f}")

        # Size-based metrics on one line
        map_s = metrics.get("map_small", -1)
        map_m = metrics.get("map_medium", -1)
        map_l = metrics.get("map_large", -1)
        if map_s >= 0 or map_m >= 0 or map_l >= 0:
            size_str = f"S: {map_s:.4f}  M: {map_m:.4f}  L: {map_l:.4f}"
            table.add_row("  mAP by size", size_str)

        # === Operative Metrics Section ===
        conf = self.config.conf_prod
        table.add_row("", "")
        table.add_row(f"[bold yellow]Operative @ {conf}[/]", "")

        p_conf = metrics.get("precision_at_conf", 0)
        r_conf = metrics.get("recall_at_conf", 0)
        f1_conf = metrics.get("f1_at_conf", 0)
        best_f1 = metrics.get("best_f1", 0)
        best_conf = metrics.get("best_f1_conf", 0)

        table.add_row("  Precision", f"{p_conf:.4f}")
        table.add_row("  Recall", f"{r_conf:.4f}")
        table.add_row("  F1", f"{f1_conf:.4f}")
        table.add_row("  Best F1", f"{best_f1:.4f} @ conf={best_conf:.2f}")

        # === Error Analysis Section ===
        fp = metrics.get("total_fp", 0)
        fn = metrics.get("total_fn", 0)
        det_mean = metrics.get("mean_det_per_img", 0)
        confusions = metrics.get("top_confusions", [])

        table.add_row("", "")
        table.add_row("[bold yellow]Error Analysis[/]", "")
        table.add_row("  False Positives", str(fp))
        table.add_row("  False Negatives", str(fn))
        table.add_row("  Det/Image (mean)", f"{det_mean:.1f}")

        if confusions:
            conf_strs = [f"{c['pred']} → {c['true']} ({c['count']})" for c in confusions[:2]]
            table.add_row("  Top Confusions", ", ".join(conf_strs))

        # === Per-Class Section ===
        per_class = metrics.get("per_class", [])
        if per_class:
            sorted_by_ap = sorted(per_class, key=lambda x: x.get("ap", 0), reverse=True)

            table.add_row("", "")
            table.add_row("[bold yellow]Per-Class (by AP)[/]", "")
            for cls in sorted_by_ap:
                cls_info = f"AP: {cls['ap']:.4f}  AP50: {cls['ap50']:.4f}  R: {cls['recall']:.4f}  GT: {cls['support']}"
                table.add_row(f"  {cls['name']}", cls_info)

        # === Threshold Sweep Section ===
        sweep = metrics.get("threshold_sweep", {})
        if sweep:
            table.add_row("", "")
            table.add_row("[bold yellow]Threshold Sweep[/]", "")
            for thresh in sorted(sweep.keys()):
                data = sweep[thresh]
                sweep_str = f"P: {data['p']:.2f}  R: {data['r']:.2f}  F1: {data['f1']:.2f}"
                table.add_row(f"  conf={thresh:.2f}", sweep_str)

        # === Trends Section ===
        if self.config.show_trends and len(self.history.history.get("map", [])) > 1:
            table.add_row("", "")
            table.add_row("[bold yellow]Trends[/]", "")
            for key, label in [("map", "mAP"), ("map50", "mAP50"), ("ar_100", "AR@100")]:
                spark = self.history.sparkline(key)
                min_v, max_v = self.history.get_range(key)
                trend_str = f"{spark}  (min: {min_v:.3f}, max: {max_v:.3f})"
                table.add_row(f"  {label}", trend_str)

        # === Info Section ===
        table.add_row("", "")
        table.add_row("[bold yellow]Info[/]", "")

        info_parts = []
        if run_id:
            info_parts.append(f"run: {run_id}")
        if num_images:
            info_parts.append(f"images: {num_images}")
        info_parts.append(f"size: {image_size[0]}x{image_size[1]}")
        table.add_row("  ", "  ".join(info_parts))

        conf_info = f"conf: {self.config.conf_prod}  iou: {self.config.nms_iou}"
        table.add_row("  ", conf_info)

        # Print the table
        self.console.print()
        self.console.print(table)
        self.console.print()

    def reset(self) -> None:
        """Reset dashboard state."""
        self.history = MetricsHistory(self.config.trend_window)
        self.best_map = 0.0
        self.best_epoch = 0
