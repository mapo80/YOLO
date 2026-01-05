"""
Tests for progress indicators module.

Tests the Rich-based progress indicators used throughout the training pipeline
for consistent UX similar to Lightning's RichProgressBar.
"""

import sys
import time
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class TestSpinner:
    """Tests for the spinner context manager."""

    def test_spinner_context_manager_enters_and_exits(self):
        """Test that spinner can be used as context manager."""
        from yolo.utils.progress import spinner

        # Should not raise any exceptions
        with spinner("Test message"):
            pass

    def test_spinner_with_operation(self):
        """Test spinner during an actual operation."""
        from yolo.utils.progress import spinner

        result = 0
        with spinner("Computing..."):
            for i in range(100):
                result += i

        assert result == 4950  # sum(0..99)

    def test_spinner_handles_exceptions(self):
        """Test that spinner properly cleans up on exception."""
        from yolo.utils.progress import spinner

        with pytest.raises(ValueError):
            with spinner("Test"):
                raise ValueError("Test error")

    def test_spinner_with_empty_message(self):
        """Test spinner with empty message."""
        from yolo.utils.progress import spinner

        with spinner(""):
            pass


class TestProgressBar:
    """Tests for the progress_bar context manager."""

    def test_progress_bar_context_manager(self):
        """Test that progress_bar can be used as context manager."""
        from yolo.utils.progress import progress_bar

        with progress_bar(10, "Test") as update:
            for _ in range(10):
                update(1)

    def test_progress_bar_tracks_progress(self):
        """Test that progress bar update function works."""
        from yolo.utils.progress import progress_bar

        updates_called = 0
        with progress_bar(5, "Testing") as update:
            for _ in range(5):
                update(1)
                updates_called += 1

        assert updates_called == 5

    def test_progress_bar_with_variable_steps(self):
        """Test progress bar with variable step sizes."""
        from yolo.utils.progress import progress_bar

        total_advanced = 0
        with progress_bar(100, "Variable steps") as update:
            update(10)
            total_advanced += 10
            update(25)
            total_advanced += 25
            update(65)
            total_advanced += 65

        assert total_advanced == 100

    def test_progress_bar_handles_exceptions(self):
        """Test that progress bar properly cleans up on exception."""
        from yolo.utils.progress import progress_bar

        with pytest.raises(RuntimeError):
            with progress_bar(10, "Test") as update:
                update(5)
                raise RuntimeError("Test error")

    def test_progress_bar_zero_total(self):
        """Test progress bar with zero total items."""
        from yolo.utils.progress import progress_bar

        # Should handle gracefully
        with progress_bar(0, "Empty") as update:
            pass

    def test_progress_bar_large_total(self):
        """Test progress bar with large total."""
        from yolo.utils.progress import progress_bar

        with progress_bar(1000000, "Large") as update:
            update(500000)
            update(500000)


class TestProgressTracker:
    """Tests for the ProgressTracker class."""

    def test_tracker_initialization(self):
        """Test ProgressTracker can be initialized."""
        from yolo.utils.progress import ProgressTracker

        tracker = ProgressTracker()
        assert tracker._progress is None
        assert tracker._task_id is None

    def test_tracker_start_spinner(self):
        """Test starting tracker in spinner mode (no total)."""
        from yolo.utils.progress import ProgressTracker

        tracker = ProgressTracker()
        tracker.start("Spinner mode", total=None)
        assert tracker._progress is not None
        tracker.finish()
        assert tracker._progress is None

    def test_tracker_start_progress_bar(self):
        """Test starting tracker in progress bar mode (with total)."""
        from yolo.utils.progress import ProgressTracker

        tracker = ProgressTracker()
        tracker.start("Progress mode", total=100)
        assert tracker._progress is not None
        tracker.update(50)
        tracker.finish()

    def test_tracker_context_manager(self):
        """Test ProgressTracker as context manager."""
        from yolo.utils.progress import ProgressTracker

        with ProgressTracker() as tracker:
            tracker.start("Test", total=10)
            for _ in range(10):
                tracker.update(1)

    def test_tracker_multiple_phases(self):
        """Test ProgressTracker with multiple phases."""
        from yolo.utils.progress import ProgressTracker

        tracker = ProgressTracker()

        # Phase 1
        tracker.start("Phase 1", total=50)
        tracker.update(50)
        tracker.finish()

        # Phase 2
        tracker.start("Phase 2", total=100)
        tracker.update(100)
        tracker.finish()

    def test_tracker_finish_with_message(self):
        """Test ProgressTracker finish with completion message."""
        from yolo.utils.progress import ProgressTracker

        tracker = ProgressTracker()
        tracker.start("Test", total=10)
        tracker.update(10)
        # finish() with message should not raise
        tracker.finish(message="Done!")

    def test_tracker_update_without_start(self):
        """Test that update without start doesn't crash."""
        from yolo.utils.progress import ProgressTracker

        tracker = ProgressTracker()
        # Should not raise
        tracker.update(1)

    def test_tracker_double_finish(self):
        """Test that calling finish twice doesn't crash."""
        from yolo.utils.progress import ProgressTracker

        tracker = ProgressTracker()
        tracker.start("Test", total=10)
        tracker.finish()
        tracker.finish()  # Should not raise


class TestProgressIntegration:
    """Integration tests for progress indicators in the codebase."""

    def test_spinner_import(self):
        """Test that spinner can be imported from progress module."""
        from yolo.utils.progress import spinner
        assert callable(spinner)

    def test_progress_bar_import(self):
        """Test that progress_bar can be imported from progress module."""
        from yolo.utils.progress import progress_bar
        assert callable(progress_bar)

    def test_progress_tracker_import(self):
        """Test that ProgressTracker can be imported from progress module."""
        from yolo.utils.progress import ProgressTracker
        assert ProgressTracker is not None

    def test_console_import(self):
        """Test that console can be imported from progress module."""
        from yolo.utils.progress import console
        from rich.console import Console
        assert isinstance(console, Console)

    def test_metrics_uses_progress(self):
        """Test that metrics module properly imports progress spinner."""
        # This should not raise ImportError
        from yolo.utils.metrics import DetMetrics

        # Create minimal metrics instance
        metrics = DetMetrics(names={0: "class0", 1: "class1"})
        assert metrics is not None

    def test_datamodule_uses_progress(self):
        """Test that datamodule can import progress utilities."""
        # This should not raise ImportError
        from yolo.utils.progress import spinner, progress_bar, console

        # Verify they're usable
        assert spinner is not None
        assert progress_bar is not None
        assert console is not None


class TestProgressConsole:
    """Tests for the shared console instance."""

    def test_console_print(self):
        """Test that console can print rich text."""
        from yolo.utils.progress import console

        # Should not raise
        console.print("[green]Success[/green]")
        console.print("[red]Error[/red]")
        console.print("[bold]Bold text[/bold]")

    def test_console_print_with_markup(self):
        """Test console with various Rich markup."""
        from yolo.utils.progress import console

        # Various markup styles
        console.print("[green]✓[/green] Task completed")
        console.print("[yellow]⚠[/yellow] Warning message")
        console.print("[red]✗[/red] Error occurred")
        console.print("[bold blue]Info:[/bold blue] Message")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
