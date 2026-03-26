"""Job monitoring with Rich progress bars, throughput counters, and logging.

Provides a :class:`Monitor` that plugs into the orchestrator's
``on_progress`` callback to display a live progress bar, track
throughput and error rates, and optionally write a structured log file.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
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
    concurrency : int
        Unused — kept for API compatibility.
    endpoint_urls : list[str]
        URLs of all vLLM endpoints in the pool. Used to compute
        per-endpoint and per-node throughput rates. If empty these
        metrics are omitted.
    """

    def __init__(
        self,
        config: MonitorConfig,
        total: int = 0,
        concurrency: int = 0,
        endpoint_urls: Optional[list[str]] = None,
    ) -> None:
        self.config = config
        self.total = total
        self._completed = 0
        self._failed = 0
        self._latencies: list[float] = []
        self._start_time = time.monotonic()
        self._last_report = 0.0
        self._log_handle: Optional[Any] = None

        # Per-endpoint completion counters (URL → count)
        self._endpoint_completions: dict[str, int] = defaultdict(int)

        # Derive node set from hostnames (http://host:port → host)
        urls = endpoint_urls or []
        self._num_endpoints: int = len(urls)
        self._num_nodes: int = len({urlparse(u).hostname for u in urls if u})

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
        self,
        completed: int,
        total: int,
        item_id: str,
        latency: float,
        endpoint_url: str = "",
    ) -> None:
        """Called by the orchestrator after each item completes."""
        self._completed = completed
        self.total = total
        self._latencies.append(latency)
        if endpoint_url:
            self._endpoint_completions[endpoint_url] += 1

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
            else 0.0
        )
        rates = self._throughput_rates()
        throughput_parts = [
            f"{rates['per_min']:.1f} items/min ({rates['per_s']:.2f}/s) global",
        ]
        if self._num_endpoints:
            throughput_parts.append(
                f"{rates['per_endpoint_per_min']:.2f}/min per endpoint"
                f" ({self._num_endpoints} endpoints)"
            )
        if self._num_nodes:
            throughput_parts.append(
                f"{rates['per_node_per_min']:.2f}/min per node"
                f" ({self._num_nodes} nodes)"
            )
        msg = (
            f"Progress: {self._completed}/{self.total} "
            f"| {' | '.join(throughput_parts)} "
            f"| failed: {self._failed} "
            f"| avg latency: {avg_lat:.2f}s "
            f"| elapsed: {elapsed:.0f}s"
        )
        logger.info(msg)
        log_data: dict = {
            "completed": self._completed,
            "total": self.total,
            "failed": self._failed,
            "avg_latency_s": round(avg_lat, 3),
            "elapsed_s": round(elapsed, 1),
            "throughput_per_s": round(rates["per_s"], 3),
            "throughput_per_min": round(rates["per_min"], 2),
        }
        if self._num_endpoints:
            log_data["num_endpoints"] = self._num_endpoints
            log_data["throughput_per_endpoint_per_min"] = round(rates["per_endpoint_per_min"], 3)
        if self._num_nodes:
            log_data["num_nodes"] = self._num_nodes
            log_data["throughput_per_node_per_min"] = round(rates["per_node_per_min"], 3)
        self._log_event("progress", log_data)

    def _throughput_rates(self) -> dict[str, float]:
        """Return a dict of throughput rates (all in items/min and items/s)."""
        elapsed = time.monotonic() - self._start_time
        if elapsed <= 0 or self._completed == 0:
            return {
                "per_s": 0.0,
                "per_min": 0.0,
                "per_endpoint_per_min": 0.0,
                "per_node_per_min": 0.0,
            }
        per_s = self._completed / elapsed
        per_min = per_s * 60
        return {
            "per_s": per_s,
            "per_min": per_min,
            "per_endpoint_per_min": per_min / self._num_endpoints if self._num_endpoints else 0.0,
            "per_node_per_min": per_min / self._num_nodes if self._num_nodes else 0.0,
        }

    def _throughput_str(self) -> str:
        """Compact throughput string for the Rich progress bar."""
        rates = self._throughput_rates()
        parts = [f"{rates['per_min']:.1f}/min ({rates['per_s']:.2f}/s) global"]
        if self._num_endpoints:
            parts.append(f"{rates['per_endpoint_per_min']:.2f}/min·ep")
        if self._num_nodes:
            parts.append(f"{rates['per_node_per_min']:.2f}/min·node")
        return " | ".join(parts)

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
        rates = self._throughput_rates()
        result: dict[str, Any] = {
            "completed": self._completed,
            "failed": self._failed,
            "total": self.total,
            "elapsed_seconds": round(elapsed, 2),
            "avg_latency_s": round(avg_lat, 3),
            "throughput_per_s": round(rates["per_s"], 3),
            "throughput_per_min": round(rates["per_min"], 2),
        }
        if self._num_endpoints:
            result["num_endpoints"] = self._num_endpoints
            result["throughput_per_endpoint_per_min"] = round(rates["per_endpoint_per_min"], 3)
        if self._num_nodes:
            result["num_nodes"] = self._num_nodes
            result["throughput_per_node_per_min"] = round(rates["per_node_per_min"], 3)
        return result

    def close(self) -> None:
        """Stop the progress bar and close the log file."""
        if self._progress is not None:
            self._progress.stop()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
