"""Tests for the ExaForge monitoring module."""

from __future__ import annotations

import json
from pathlib import Path

from exaforge.config import MonitorConfig
from exaforge.monitor import Monitor


class TestMonitor:
    def _cfg(
        self, tmp_dir: Path, log: bool = False, rich: bool = False
    ) -> MonitorConfig:
        return MonitorConfig(
            log_file=tmp_dir / "test.log" if log else None,
            progress_interval=0.0,
            enable_rich=rich,
        )

    def test_on_progress_tracks_count(self, tmp_dir: Path) -> None:
        m = Monitor(self._cfg(tmp_dir), total=10)
        m.on_progress(1, 10, "item-1", 0.5)
        m.on_progress(2, 10, "item-2", 0.3)
        assert m._completed == 2

    def test_latency_tracking(self, tmp_dir: Path) -> None:
        m = Monitor(self._cfg(tmp_dir), total=5)
        m.on_progress(1, 5, "a", 1.0)
        m.on_progress(2, 5, "b", 2.0)
        assert len(m._latencies) == 2
        assert sum(m._latencies) == 3.0

    def test_summary(self, tmp_dir: Path) -> None:
        m = Monitor(self._cfg(tmp_dir), total=3)
        m.on_progress(1, 3, "a", 1.0)
        m.on_progress(2, 3, "b", 2.0)
        m.record_failure("c", "timeout")

        s = m.summary()
        assert s["completed"] == 2
        assert s["failed"] == 1
        assert s["total"] == 3
        assert s["avg_latency"] > 0

    def test_log_file_written(self, tmp_dir: Path) -> None:
        m = Monitor(self._cfg(tmp_dir, log=True), total=3)
        m.on_progress(1, 3, "a", 1.0)
        m.close()

        log_path = tmp_dir / "test.log"
        assert log_path.is_file()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["event"] == "progress"

    def test_record_failure(self, tmp_dir: Path) -> None:
        m = Monitor(self._cfg(tmp_dir, log=True), total=1)
        m.record_failure("item-x", "connection refused")
        m.close()

        log_path = tmp_dir / "test.log"
        lines = log_path.read_text().strip().splitlines()
        events = [json.loads(l) for l in lines]
        failures = [e for e in events if e["event"] == "failure"]
        assert len(failures) == 1
        assert failures[0]["item_id"] == "item-x"

    def test_close_idempotent(self, tmp_dir: Path) -> None:
        m = Monitor(self._cfg(tmp_dir), total=0)
        m.close()
        m.close()

    def test_no_rich_when_disabled(self, tmp_dir: Path) -> None:
        m = Monitor(self._cfg(tmp_dir, rich=False), total=5)
        assert m._progress is None

    def test_throughput_str_initial(self, tmp_dir: Path) -> None:
        m = Monitor(self._cfg(tmp_dir), total=1)
        s = m._throughput_str()
        assert "items/s" in s
