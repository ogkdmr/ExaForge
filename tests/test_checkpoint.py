"""Tests for the ExaForge checkpoint system."""

from __future__ import annotations

import json
from pathlib import Path

from exaforge.checkpoint import CheckpointManager
from exaforge.config import CheckpointConfig


class TestCheckpointManager:
    def _cfg(self, tmp_dir: Path, enabled: bool = True) -> CheckpointConfig:
        return CheckpointConfig(
            enabled=enabled,
            checkpoint_file=tmp_dir / "checkpoint.json",
        )

    def test_mark_done_and_is_done(self, tmp_dir: Path) -> None:
        mgr = CheckpointManager(self._cfg(tmp_dir))
        assert not mgr.is_done("item-1")
        mgr.mark_done("item-1")
        assert mgr.is_done("item-1")

    def test_save_and_reload(self, tmp_dir: Path) -> None:
        cfg = self._cfg(tmp_dir)
        mgr = CheckpointManager(cfg)
        mgr.mark_done("a")
        mgr.mark_done("b")
        mgr.total_items = 10
        mgr.save()

        mgr2 = CheckpointManager(cfg)
        assert mgr2.is_done("a")
        assert mgr2.is_done("b")
        assert not mgr2.is_done("c")
        assert mgr2.completed_count == 2
        assert mgr2.total_items == 10

    def test_mark_done_batch_saves(self, tmp_dir: Path) -> None:
        cfg = self._cfg(tmp_dir)
        mgr = CheckpointManager(cfg)
        mgr.mark_done_batch(["x", "y", "z"])
        assert cfg.checkpoint_file.is_file()

        mgr2 = CheckpointManager(cfg)
        assert mgr2.completed_count == 3

    def test_filter_pending(self, tmp_dir: Path) -> None:
        mgr = CheckpointManager(self._cfg(tmp_dir))
        mgr.mark_done("done-1")
        mgr.mark_done("done-2")
        pending = mgr.filter_pending(
            ["done-1", "pending-1", "done-2", "pending-2"]
        )
        assert pending == ["pending-1", "pending-2"]

    def test_disabled_never_filters(self, tmp_dir: Path) -> None:
        mgr = CheckpointManager(self._cfg(tmp_dir, enabled=False))
        mgr.mark_done("item-1")
        assert not mgr.is_done("item-1")
        assert mgr.filter_pending(["item-1", "item-2"]) == [
            "item-1",
            "item-2",
        ]

    def test_disabled_does_not_save(self, tmp_dir: Path) -> None:
        cfg = self._cfg(tmp_dir, enabled=False)
        mgr = CheckpointManager(cfg)
        mgr.mark_done("item-1")
        mgr.save()
        assert not cfg.checkpoint_file.exists()

    def test_corrupted_checkpoint_handled(self, tmp_dir: Path) -> None:
        cfg = self._cfg(tmp_dir)
        cfg.checkpoint_file.write_text("not json")
        mgr = CheckpointManager(cfg)
        assert mgr.completed_count == 0

    def test_checkpoint_file_is_valid_json(self, tmp_dir: Path) -> None:
        cfg = self._cfg(tmp_dir)
        mgr = CheckpointManager(cfg)
        mgr.mark_done_batch(["a", "b"])
        data = json.loads(cfg.checkpoint_file.read_text())
        assert "completed" in data
        assert "elapsed_seconds" in data

    def test_completed_count(self, tmp_dir: Path) -> None:
        mgr = CheckpointManager(self._cfg(tmp_dir))
        assert mgr.completed_count == 0
        mgr.mark_done("a")
        mgr.mark_done("b")
        assert mgr.completed_count == 2

    def test_idempotent_mark(self, tmp_dir: Path) -> None:
        mgr = CheckpointManager(self._cfg(tmp_dir))
        mgr.mark_done("a")
        mgr.mark_done("a")
        assert mgr.completed_count == 1
