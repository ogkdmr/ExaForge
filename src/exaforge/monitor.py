"""Job monitoring with Rich progress bars, throughput counters, and logging.

Provides a :class:`Monitor` that plugs into the orchestrator's
``on_progress`` callback to display a live progress bar, track
throughput and error rates, and optionally write a structured log file.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from exaforge.config import MonitorConfig

logger = logging.getLogger(__name__)


class Monitor:
    """Live progress tracking and structured logging.

    Parameters
    ----------
    config : MonitorConfig
        Monitor settings (log file, progress interval, Rich toggle).
    total : int
        Total number of items to process.
    """

    def __init__(self, config: MonitorConfig, total: int = 0) -> None:
        self.config = config
        self.total = total
        self._completed = 0
        self._failed = 0
        self._latencies: list[float] = []
        self._start_time = time.monotonic()
        self._last_report = 0.0
        self._log_handle: Optional[Any] = None

        self._progress: Optional[Any] = None
        self._task_id: Optional[Any] = None

        if config.log_file:
            config.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(config.log_file, "a", encoding="utf-8")

        if config.enable_rich and total > 0:
            self._init_rich()

    def _init_rich(self) -> None:
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("[green]{task.fields[throughput]}"),
            )
            self._task_id = self._progress.add_task(
                "Inference", total=self.total, throughput="0.0 items/s"
            )
            self._progress.start()
        except ImportError:
            logger.warning("Rich not available — falling back to log output")

    # ------------------------------------------------------------------
    # Callback for the orchestrator
    # ------------------------------------------------------------------

    def on_progress(
        self, completed: int, total: int, item_id: str, latency: float
    ) -> None:
        """Called by the orchestrator after each item completes."""
        self._completed = completed
        self.total = total
        self._latencies.append(latency)

        if self._progress is not None and self._task_id is not None:
            throughput = self._throughput_str()
            self._progress.update(
                self._task_id,
                completed=completed,
                total=total,
                throughput=throughput,
            )

        now = time.monotonic()
        if now - self._last_report >= self.config.progress_interval:
            self._periodic_report()
            self._last_report = now

    def record_failure(self, item_id: str, error: str) -> None:
        """Record a failed item."""
        self._failed += 1
        self._log_event("failure", {"item_id": item_id, "error": error})

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _periodic_report(self) -> None:
        elapsed = time.monotonic() - self._start_time
        avg_lat = (
            sum(self._latencies) / len(self._latencies)
            if self._latencies
            else 0
        )
        msg = (
            f"Progress: {self._completed}/{self.total} "
            f"({self._throughput_str()}) | "
            f"failed: {self._failed} | "
            f"avg latency: {avg_lat:.2f}s | "
            f"elapsed: {elapsed:.0f}s"
        )
        logger.info(msg)
        self._log_event(
            "progress",
            {
                "completed": self._completed,
                "total": self.total,
                "failed": self._failed,
                "avg_latency": round(avg_lat, 3),
                "elapsed": round(elapsed, 1),
            },
        )

    def _throughput_str(self) -> str:
        elapsed = time.monotonic() - self._start_time
        if elapsed <= 0:
            return "0.0 items/s"
        rate = self._completed / elapsed
        return f"{rate:.1f} items/s"

    def _log_event(self, event: str, data: dict) -> None:
        if self._log_handle is not None:
            record = {
                "event": event,
                "timestamp": time.time(),
                **data,
            }
            self._log_handle.write(json.dumps(record) + "\n")
            self._log_handle.flush()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of the run."""
        elapsed = time.monotonic() - self._start_time
        avg_lat = (
            sum(self._latencies) / len(self._latencies)
            if self._latencies
            else 0.0
        )
        return {
            "completed": self._completed,
            "failed": self._failed,
            "total": self.total,
            "elapsed_seconds": round(elapsed, 2),
            "avg_latency": round(avg_lat, 3),
            "throughput": self._throughput_str(),
        }

    def close(self) -> None:
        """Stop the progress bar and close the log file."""
        if self._progress is not None:
            self._progress.stop()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
