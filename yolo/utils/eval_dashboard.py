"""
Eval Dashboard - ASCII visualization for detection metrics.

Provides comprehensive metrics display in console/terminal format
for quick decision making during training and validation.

Uses Rich library for beautiful terminal output with proper spacing.
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
    threshold_sweep: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    show_trends: bool = True
    trend_window: int = 10


class MetricsHistory:
    """Tracks metrics across epochs for trend visualization."""

    def __init__(self, window: int = 10):
        """
        Initialize metrics history.

        Args:
            window: Number of epochs to track for trends
        """
        self.window = window
        self.history: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        """
        Update history with new metrics.

        Args:
            metrics: Dict of metric name -> value
        """
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(float(value))
            if len(self.history[key]) > self.window:
                self.history[key] = self.history[key][-self.window :]

    def sparkline(self, key: str, width: int = 10) -> str:
        """
        Generate ASCII sparkline for metric history.

        Args:
            key: Metric name
            width: Target width of sparkline

        Returns:
            ASCII sparkline string
        """
        if key not in self.history or len(self.history[key]) < 2:
            return "-" * width

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
        """Get min/max range for a metric."""
        if key not in self.history or not self.history[key]:
            return (0.0, 0.0)
        return (min(self.history[key]), max(self.history[key]))


class EvalDashboard:
    """Dashboard for eval metrics visualization using Rich tables."""

    def __init__(self, config: Optional[EvalConfig] = None):
        """
        Initialize dashboard.

        Args:
            config: Dashboard configuration
        """
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
        """
        Render complete dashboard to string.

        Args:
            metrics: Dict with all computed metrics from DetMetrics.process()
            epoch: Current epoch number (1-indexed)
            total_epochs: Total number of epochs
            run_id: Training run identifier
            num_images: Number of validation images
            image_size: Image dimensions (width, height)
            latency_ms: Optional inference latency in milliseconds
            memory_mb: Optional GPU memory usage in MB
            model_size_mb: Optional model file size in MB

        Returns:
            Formatted dashboard string (for compatibility, but print() is preferred)
        """
        # Update history for trends
        trend_metrics = {
            "map": metrics.get("map", 0),
            "map50": metrics.get("map50", 0),
            "ar_100": metrics.get("ar_100", 0),
        }
        self.history.update(trend_metrics)

        # Track best model
        current_map = metrics.get("map", 0)
        is_best = current_map > self.best_map
        if is_best and epoch is not None:
            self.best_map = current_map
            self.best_epoch = epoch

        # For compatibility, return a simple string summary
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
        """Print dashboard to console using Rich tables."""
        # Update history for trends
        trend_metrics = {
            "map": metrics.get("map", 0),
            "map50": metrics.get("map50", 0),
            "ar_100": metrics.get("ar_100", 0),
        }
        self.history.update(trend_metrics)

        # Track best model and calculate delta
        current_map = metrics.get("map", 0)
        previous_best = self.best_map
        is_best = current_map > self.best_map

        # Calculate delta from best (negative if worse, positive if better)
        if previous_best > 0:
            delta_from_best = current_map - previous_best
        else:
            delta_from_best = None  # First epoch, no comparison

        if is_best and epoch is not None:
            self.best_map = current_map
            self.best_epoch = epoch

        # Print sections
        self.console.print()  # Empty line before dashboard
        self._print_header(epoch, total_epochs, run_id, num_images, image_size, is_best, current_map, delta_from_best)
        self.console.print()
        self._print_kpi_quality(metrics)
        self.console.print()
        self._print_kpi_operative(metrics)

        if self.config.show_trends and len(self.history.history.get("map", [])) > 1:
            self.console.print()
            self._print_trends()

        sweep = metrics.get("threshold_sweep", {})
        if sweep:
            self.console.print()
            self._print_threshold_sweep(metrics)

        per_class = metrics.get("per_class", [])
        if per_class:
            self.console.print()
            self._print_per_class(metrics)

        self.console.print()
        self._print_error_health(metrics)

        if latency_ms is not None or memory_mb is not None or model_size_mb is not None:
            self.console.print()
            self._print_deploy_metrics(latency_ms, memory_mb, model_size_mb)

        self.console.print()  # Empty line after dashboard

    def _print_header(
        self,
        epoch: Optional[int],
        total_epochs: Optional[int],
        run_id: str,
        num_images: int,
        image_size: Tuple[int, int],
        is_best: bool = False,
        current_map: float = 0.0,
        delta_from_best: Optional[float] = None,
    ) -> None:
        """Print header section."""
        table = Table(title="EVAL SUMMARY", title_style="bold cyan", box=box.ROUNDED, padding=(0, 1))

        # Build header info
        parts = []
        if run_id:
            parts.append(f"run: {run_id}")
        if epoch and total_epochs:
            parts.append(f"epoch: {epoch}/{total_epochs}")
        if num_images:
            parts.append(f"imgs: {num_images}")
        parts.append(f"size: {image_size[0]}x{image_size[1]}")

        table.add_column("Info", style="white")
        table.add_column("mAP", style="cyan")
        table.add_column("Settings", style="dim")

        header_line = "  ".join(parts)

        # Build mAP column with delta and best indicator
        map_str = f"{current_map:.4f}"
        if is_best:
            map_str += " [bold green][NEW BEST][/bold green]"
        elif delta_from_best is not None:
            if delta_from_best >= 0:
                map_str += f" [green](+{delta_from_best:.4f})[/green]"
            else:
                map_str += f" [red]({delta_from_best:.4f})[/red]"

        conf_line = f"conf: {self.config.conf_prod}  iou: {self.config.nms_iou}  max_det: {self.config.max_det}"

        table.add_row(header_line, map_str, conf_line)
        self.console.print(table)

    def _print_kpi_quality(self, metrics: Dict[str, Any]) -> None:
        """Print quality KPI section."""
        table = Table(title="KPI (QUALITY)", title_style="bold green", box=box.ROUNDED, padding=(0, 1))

        table.add_column("AP50-95", justify="right", style="cyan")
        table.add_column("AP50", justify="right", style="cyan")
        table.add_column("AP75", justify="right", style="cyan")
        table.add_column("AR@100", justify="right", style="cyan")
        table.add_column("APs", justify="right", style="dim")
        table.add_column("APm", justify="right", style="dim")
        table.add_column("APl", justify="right", style="dim")

        def fmt(v):
            if v is None or v < 0:
                return "-"
            return f"{v:.4f}"

        table.add_row(
            fmt(metrics.get("map", 0)),
            fmt(metrics.get("map50", 0)),
            fmt(metrics.get("map75", 0)),
            fmt(metrics.get("ar_100", 0)),
            fmt(metrics.get("map_small", 0)),
            fmt(metrics.get("map_medium", 0)),
            fmt(metrics.get("map_large", 0)),
        )

        self.console.print(table)

    def _print_kpi_operative(self, metrics: Dict[str, Any]) -> None:
        """Print operative KPI section."""
        conf = metrics.get("conf_prod", self.config.conf_prod)

        table = Table(title=f"KPI (OPERATIVE @ {conf})", title_style="bold yellow", box=box.ROUNDED, padding=(0, 1))

        table.add_column(f"P@{conf}", justify="right", style="cyan")
        table.add_column(f"R@{conf}", justify="right", style="cyan")
        table.add_column(f"F1@{conf}", justify="right", style="cyan")
        table.add_column("best-F1", justify="right", style="green")
        table.add_column("conf_best", justify="right", style="green")

        p_conf = metrics.get("precision_at_conf", 0)
        r_conf = metrics.get("recall_at_conf", 0)
        f1_conf = metrics.get("f1_at_conf", 0)
        best_f1 = metrics.get("best_f1", 0)
        best_conf = metrics.get("best_f1_conf", 0)

        table.add_row(
            f"{p_conf:.4f}",
            f"{r_conf:.4f}",
            f"{f1_conf:.4f}",
            f"{best_f1:.4f}",
            f"{best_conf:.2f}",
        )
        self.console.print(table)

    def _print_trends(self) -> None:
        """Print trends section with sparklines."""
        table = Table(
            title=f"TRENDS (last {self.config.trend_window} epochs)",
            title_style="bold magenta",
            box=box.ROUNDED,
            padding=(0, 1)
        )

        table.add_column("Metric", style="dim")
        table.add_column("Trend", style="cyan")
        table.add_column("Range", style="dim")

        for key, label in [("map", "AP50-95"), ("map50", "AP50"), ("ar_100", "AR@100")]:
            spark = self.history.sparkline(key)
            min_v, max_v = self.history.get_range(key)
            table.add_row(label, spark, f"min: {min_v:.2f}  max: {max_v:.2f}")

        self.console.print(table)

    def _print_threshold_sweep(self, metrics: Dict[str, Any]) -> None:
        """Print threshold sweep table."""
        sweep = metrics.get("threshold_sweep", {})
        if not sweep:
            return

        table = Table(title="THRESHOLD SWEEP", title_style="bold blue", box=box.ROUNDED, padding=(0, 1))

        table.add_column("conf", justify="right", style="dim")
        table.add_column("P", justify="right", style="cyan")
        table.add_column("R", justify="right", style="cyan")
        table.add_column("F1", justify="right", style="green")

        for thresh in sorted(sweep.keys()):
            data = sweep[thresh]
            table.add_row(
                f"{thresh:.2f}",
                f"{data['p']:.2f}",
                f"{data['r']:.2f}",
                f"{data['f1']:.2f}",
            )

        self.console.print(table)

    def _print_per_class(self, metrics: Dict[str, Any]) -> None:
        """Print per-class TOP/WORST section."""
        per_class = metrics.get("per_class", [])
        if not per_class:
            return

        n = self.config.top_n_classes
        sorted_by_ap = sorted(per_class, key=lambda x: x.get("ap", 0), reverse=True)

        top_classes = sorted_by_ap[:n]
        worst_classes = sorted_by_ap[-n:] if len(sorted_by_ap) > n else []

        # TOP classes table
        table = Table(title="PER-CLASS: TOP by AP50-95", title_style="bold green", box=box.ROUNDED, padding=(0, 1))

        table.add_column("Class", style="white")
        table.add_column("AP50-95", justify="right", style="cyan")
        table.add_column("AP50", justify="right", style="cyan")
        table.add_column("R@conf", justify="right", style="cyan")
        table.add_column("GT", justify="right", style="dim")

        for cls in top_classes:
            table.add_row(
                cls["name"],
                f"{cls['ap']:.4f}",
                f"{cls['ap50']:.4f}",
                f"{cls['recall']:.4f}",
                str(cls["support"]),
            )

        self.console.print(table)

        # WORST classes table
        if worst_classes:
            self.console.print()
            table2 = Table(title="PER-CLASS: WORST by AP50-95", title_style="bold red", box=box.ROUNDED, padding=(0, 1))

            table2.add_column("Class", style="white")
            table2.add_column("AP50-95", justify="right", style="cyan")
            table2.add_column("AP50", justify="right", style="cyan")
            table2.add_column("R@conf", justify="right", style="cyan")
            table2.add_column("GT", justify="right", style="dim")

            for cls in worst_classes:
                table2.add_row(
                    cls["name"],
                    f"{cls['ap']:.4f}",
                    f"{cls['ap50']:.4f}",
                    f"{cls['recall']:.4f}",
                    str(cls["support"]),
                )

            self.console.print(table2)

    def _print_error_health(self, metrics: Dict[str, Any]) -> None:
        """Print error health check section."""
        fp = metrics.get("total_fp", 0)
        fn = metrics.get("total_fn", 0)
        det_mean = metrics.get("mean_det_per_img", 0)
        det_p95 = metrics.get("p95_det_per_img", 0)
        confusions = metrics.get("top_confusions", [])

        table = Table(title="ERROR HEALTH CHECK", title_style="bold red", box=box.ROUNDED, padding=(0, 1))

        table.add_column("FP", justify="right", style="red")
        table.add_column("FN", justify="right", style="red")
        table.add_column("det/img mean", justify="right", style="cyan")
        table.add_column("det/img p95", justify="right", style="cyan")

        table.add_row(str(fp), str(fn), f"{det_mean:.1f}", f"{det_p95:.0f}")

        self.console.print(table)

        if confusions:
            self.console.print()
            table2 = Table(title="Top Confusions (pred → true)", title_style="dim", box=box.ROUNDED, padding=(0, 1))
            table2.add_column("#", style="dim")
            table2.add_column("Predicted", style="yellow")
            table2.add_column("→", style="dim")
            table2.add_column("True", style="cyan")
            table2.add_column("Count", justify="right", style="red")

            for i, conf in enumerate(confusions[:3], 1):
                table2.add_row(str(i), conf["pred"], "→", conf["true"], str(conf["count"]))

            self.console.print(table2)

    def _print_deploy_metrics(
        self,
        latency_ms: Optional[float],
        memory_mb: Optional[float],
        model_size_mb: Optional[float],
    ) -> None:
        """Print deploy metrics section."""
        table = Table(title="DEPLOY METRICS", title_style="bold cyan", box=box.ROUNDED, padding=(0, 1))

        if latency_ms is not None:
            table.add_column("Latency", justify="right", style="cyan")
            table.add_column("FPS", justify="right", style="green")
        if memory_mb is not None:
            table.add_column("Memory", justify="right", style="cyan")
        if model_size_mb is not None:
            table.add_column("Model Size", justify="right", style="cyan")

        row = []
        if latency_ms is not None:
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0
            row.extend([f"{latency_ms:.2f}ms", f"{fps:.1f}"])
        if memory_mb is not None:
            row.append(f"{memory_mb:.1f}MB")
        if model_size_mb is not None:
            row.append(f"{model_size_mb:.1f}MB")

        table.add_row(*row)
        self.console.print(table)

    def reset(self) -> None:
        """Reset dashboard state."""
        self.history = MetricsHistory(self.config.trend_window)
        self.best_map = 0.0
        self.best_epoch = 0
