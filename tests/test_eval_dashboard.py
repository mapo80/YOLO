"""
Tests for EvalDashboard module.
"""

import pytest
from yolo.utils.eval_dashboard import EvalDashboard, EvalConfig, MetricsHistory


class TestEvalConfig:
    """Tests for EvalConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EvalConfig()
        assert config.conf_prod == 0.25
        assert config.nms_iou == 0.65
        assert config.max_det == 300
        assert config.top_n_classes == 3
        assert config.show_trends is True
        assert config.trend_window == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EvalConfig(
            conf_prod=0.5,
            nms_iou=0.45,
            max_det=100,
            top_n_classes=5,
            show_trends=False,
        )
        assert config.conf_prod == 0.5
        assert config.nms_iou == 0.45
        assert config.max_det == 100
        assert config.top_n_classes == 5
        assert config.show_trends is False


class TestMetricsHistory:
    """Tests for MetricsHistory class."""

    def test_initialization(self):
        """Test history initialization."""
        history = MetricsHistory(window=5)
        assert history.window == 5
        assert len(history.history) == 0

    def test_update_single_metric(self):
        """Test updating a single metric."""
        history = MetricsHistory(window=5)
        history.update({"map": 0.5})
        assert "map" in history.history
        assert len(history.history["map"]) == 1
        assert history.history["map"][0] == 0.5

    def test_update_multiple_metrics(self):
        """Test updating multiple metrics."""
        history = MetricsHistory(window=5)
        history.update({"map": 0.5, "map50": 0.7, "ar_100": 0.6})
        assert len(history.history) == 3
        assert history.history["map"] == [0.5]
        assert history.history["map50"] == [0.7]
        assert history.history["ar_100"] == [0.6]

    def test_window_limit(self):
        """Test that history respects window limit."""
        history = MetricsHistory(window=3)
        for i in range(5):
            history.update({"map": i * 0.1})
        assert len(history.history["map"]) == 3
        # Use pytest.approx for float comparison
        assert history.history["map"] == pytest.approx([0.2, 0.3, 0.4])

    def test_sparkline_insufficient_data(self):
        """Test sparkline with insufficient data."""
        history = MetricsHistory()
        spark = history.sparkline("map")
        assert len(spark) == 10  # Default width
        assert spark == "-" * 10

    def test_sparkline_single_value(self):
        """Test sparkline with single value."""
        history = MetricsHistory()
        history.update({"map": 0.5})
        spark = history.sparkline("map")
        assert spark == "-" * 10  # Need at least 2 values

    def test_sparkline_constant_values(self):
        """Test sparkline with constant values."""
        history = MetricsHistory()
        for _ in range(5):
            history.update({"map": 0.5})
        spark = history.sparkline("map")
        assert spark == "=" * 5

    def test_sparkline_increasing_values(self):
        """Test sparkline with increasing values."""
        history = MetricsHistory()
        for i in range(5):
            history.update({"map": i * 0.1})
        spark = history.sparkline("map")
        assert len(spark) == 5
        # First char should be lowest block, last should be highest
        assert spark[0] != spark[-1]

    def test_get_range(self):
        """Test getting min/max range."""
        history = MetricsHistory()
        for i in range(5):
            history.update({"map": i * 0.1})
        min_v, max_v = history.get_range("map")
        assert min_v == 0.0
        assert max_v == 0.4

    def test_get_range_empty(self):
        """Test getting range for non-existent metric."""
        history = MetricsHistory()
        min_v, max_v = history.get_range("unknown")
        assert min_v == 0.0
        assert max_v == 0.0

    def test_skip_non_numeric(self):
        """Test that non-numeric values are skipped."""
        history = MetricsHistory()
        history.update({"map": 0.5, "name": "test", "per_class": []})
        assert "map" in history.history
        assert "name" not in history.history
        assert "per_class" not in history.history


class TestEvalDashboard:
    """Tests for EvalDashboard class."""

    @pytest.fixture
    def dashboard(self):
        """Create dashboard instance."""
        return EvalDashboard()

    @pytest.fixture
    def sample_metrics(self):
        """Sample metrics dictionary."""
        return {
            "map": 0.4523,
            "map50": 0.6821,
            "map75": 0.4912,
            "map_small": 0.2134,
            "map_medium": 0.4521,
            "map_large": 0.5823,
            "ar_1": 0.3234,
            "ar_10": 0.4234,
            "ar_100": 0.5234,
            "ar_small": 0.3123,
            "ar_medium": 0.5234,
            "ar_large": 0.6234,
            "precision": 0.7234,
            "recall": 0.6521,
            "f1": 0.6860,
            "precision_at_conf": 0.7823,
            "recall_at_conf": 0.6234,
            "f1_at_conf": 0.6941,
            "conf_prod": 0.25,
            "best_f1": 0.7123,
            "best_f1_conf": 0.32,
            "total_fp": 234,
            "total_fn": 156,
            "mean_det_per_img": 8.3,
            "p95_det_per_img": 15.0,
            "mean_iou_tp": 0.72,
            "per_class": [
                {"name": "class_a", "ap": 0.6, "ap50": 0.8, "recall": 0.7, "precision": 0.75, "support": 100},
                {"name": "class_b", "ap": 0.5, "ap50": 0.7, "recall": 0.6, "precision": 0.65, "support": 80},
                {"name": "class_c", "ap": 0.4, "ap50": 0.6, "recall": 0.5, "precision": 0.55, "support": 60},
            ],
            "top_confusions": [
                {"pred": "class_a", "true": "class_b", "count": 15},
                {"pred": "class_b", "true": "class_c", "count": 10},
            ],
            "threshold_sweep": {
                0.1: {"p": 0.52, "r": 0.89, "f1": 0.66},
                0.2: {"p": 0.68, "r": 0.72, "f1": 0.70},
                0.3: {"p": 0.78, "r": 0.61, "f1": 0.69},
            },
        }

    def test_initialization(self, dashboard):
        """Test dashboard initialization."""
        assert dashboard.config is not None
        assert dashboard.history is not None
        assert dashboard.best_map == 0.0
        assert dashboard.best_epoch == 0

    def test_initialization_with_config(self):
        """Test dashboard with custom config."""
        config = EvalConfig(conf_prod=0.5)
        dashboard = EvalDashboard(config)
        assert dashboard.config.conf_prod == 0.5

    def test_render_returns_string(self, dashboard, sample_metrics):
        """Test that render returns a string."""
        output = dashboard.render(sample_metrics)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_render_contains_basic_info(self, dashboard, sample_metrics):
        """Test that render returns basic metrics info (for compatibility)."""
        output = dashboard.render(
            sample_metrics,
            epoch=5,
            total_epochs=100,
            run_id="test_run",
            num_images=1000,
        )
        # render() now returns a simple summary string
        assert "mAP" in output
        assert "0.4523" in output or "0.45" in output

    def test_print_works_without_error(self, dashboard, sample_metrics, capsys):
        """Test that print outputs to console without errors."""
        # This tests the full Rich-based output
        dashboard.print(
            sample_metrics,
            epoch=5,
            total_epochs=100,
            run_id="test_run",
            num_images=1000,
        )
        captured = capsys.readouterr()
        # Rich outputs to stdout, should contain some content
        assert len(captured.out) > 0

    def test_print_with_deploy_metrics(self, dashboard, sample_metrics, capsys):
        """Test print with deploy metrics."""
        dashboard.print(
            sample_metrics,
            latency_ms=8.5,
            memory_mb=256.0,
            model_size_mb=10.5,
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_render_without_deploy_metrics(self, dashboard, sample_metrics):
        """Test render returns string without crashing."""
        output = dashboard.render(sample_metrics)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_best_tracking(self, dashboard, sample_metrics):
        """Test best model tracking."""
        # First epoch
        dashboard.render(sample_metrics, epoch=1, total_epochs=10)
        assert dashboard.best_map == sample_metrics["map"]
        assert dashboard.best_epoch == 1

        # Better epoch
        better_metrics = sample_metrics.copy()
        better_metrics["map"] = 0.5
        dashboard.render(better_metrics, epoch=2, total_epochs=10)
        assert dashboard.best_map == 0.5
        assert dashboard.best_epoch == 2

        # Worse epoch
        worse_metrics = sample_metrics.copy()
        worse_metrics["map"] = 0.4
        dashboard.render(worse_metrics, epoch=3, total_epochs=10)
        assert dashboard.best_map == 0.5  # Still 0.5
        assert dashboard.best_epoch == 2  # Still epoch 2

    def test_trends_update(self, dashboard, sample_metrics):
        """Test that trends are updated."""
        for i in range(3):
            metrics = sample_metrics.copy()
            metrics["map"] = 0.4 + i * 0.05
            dashboard.render(metrics, epoch=i + 1, total_epochs=10)

        assert len(dashboard.history.history["map"]) == 3

    def test_reset(self, dashboard, sample_metrics):
        """Test dashboard reset."""
        dashboard.render(sample_metrics, epoch=1, total_epochs=10)
        assert dashboard.best_map > 0

        dashboard.reset()
        assert dashboard.best_map == 0.0
        assert dashboard.best_epoch == 0
        assert len(dashboard.history.history) == 0

    def test_empty_metrics(self, dashboard):
        """Test with minimal/empty metrics."""
        output = dashboard.render({})
        assert isinstance(output, str)

    def test_negative_metrics_handled(self, dashboard):
        """Test that negative metrics (COCO undefined) are handled."""
        metrics = {
            "map": 0.5,
            "map_small": -1,  # COCO returns -1 for undefined
            "map_medium": -1,
            "ar_100": -1,
        }
        output = dashboard.render(metrics)
        assert isinstance(output, str)
        # Should not crash

    def test_no_per_class(self, dashboard, sample_metrics):
        """Test render without per_class data."""
        metrics = sample_metrics.copy()
        del metrics["per_class"]
        output = dashboard.render(metrics)
        assert isinstance(output, str)

    def test_no_confusions(self, dashboard, sample_metrics):
        """Test render without confusion data."""
        metrics = sample_metrics.copy()
        del metrics["top_confusions"]
        output = dashboard.render(metrics)
        assert isinstance(output, str)
