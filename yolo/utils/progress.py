"""
Progress indicators for long-running operations.

Uses Rich library (same as Lightning's RichProgressBar) for consistent UX.
Provides spinners and progress bars for operations like:
- Image caching
- DataLoader worker initialization
- COCO metrics computation
"""

import contextlib
from typing import Optional

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console

# Shared console instance for consistent output
console = Console()


@contextlib.contextmanager
def spinner(message: str = "Processing..."):
    """
    Show animated spinner during long operations.

    Uses Rich library for consistent styling with Lightning's progress bars.

    Args:
        message: Text to display next to spinner

    Example:
        with spinner("Loading data..."):
            load_data()
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,  # Remove when done
    )

    with progress:
        progress.add_task(description=message, total=None)
        yield progress


@contextlib.contextmanager
def progress_bar(
    total: int,
    description: str = "Processing...",
    show_speed: bool = True,
):
    """
    Show progress bar for operations with known total.

    Uses Rich library for consistent styling with Lightning's progress bars.

    Args:
        total: Total number of items to process
        description: Text to display
        show_speed: Whether to show items/sec

    Example:
        with progress_bar(len(items), "Caching images") as update:
            for item in items:
                process(item)
                update(1)
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ]

    if show_speed:
        columns.append(TimeRemainingColumn())

    progress = Progress(
        *columns,
        console=console,
        transient=True,
    )

    with progress:
        task_id = progress.add_task(description=description, total=total)

        def update(advance: int = 1):
            progress.update(task_id, advance=advance)

        yield update


class ProgressTracker:
    """
    Reusable progress tracker for multi-step operations.

    Useful when you need to track progress across multiple phases.

    Example:
        tracker = ProgressTracker()
        tracker.start("Phase 1", total=100)
        for i in range(100):
            tracker.update(1)
        tracker.finish()

        tracker.start("Phase 2", total=50)
        ...
    """

    def __init__(self):
        self._progress: Optional[Progress] = None
        self._task_id: Optional[int] = None

    def start(self, description: str, total: Optional[int] = None):
        """Start a new progress task (spinner if total is None, bar otherwise)."""
        if self._progress is not None:
            self.finish()

        if total is None:
            # Spinner mode
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            )
        else:
            # Progress bar mode
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            )

        self._progress.start()
        self._task_id = self._progress.add_task(description=description, total=total)

    def update(self, advance: int = 1):
        """Update progress by given amount."""
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=advance)

    def finish(self, message: Optional[str] = None):
        """Finish current task and optionally print completion message."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

        if message:
            console.print(message)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
