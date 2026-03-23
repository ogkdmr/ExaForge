"""Checkpoint and resume support.

Tracks which input items have been successfully processed so that
interrupted jobs can resume without re-doing completed work.  The
checkpoint file is written atomically via :func:`lustre.atomic_write`
to survive crashes and Lustre hiccups.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from exaforge.config import CheckpointConfig
from exaforge.lustre import atomic_write

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Track completed item IDs and persist them to disk.

    Parameters
    ----------
    config : CheckpointConfig
        Checkpoint settings (enabled flag, file path).
    """

    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config
        self._completed: set[str] = set()
        self._start_time: float = time.monotonic()
        self._total_items: int = 0

        if config.enabled:
            self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load previously completed IDs from disk."""
        path = self.config.checkpoint_file
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._completed = set(data.get("completed", []))
            self._total_items = data.get("total_items", 0)
            logger.info(
                "Loaded checkpoint: %d items already completed",
                len(self._completed),
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load checkpoint: %s", exc)

    def save(self) -> None:
        """Persist the current state to disk atomically."""
        if not self.config.enabled:
            return
        data = {
            "completed": sorted(self._completed),
            "total_items": self._total_items,
            "elapsed_seconds": time.monotonic() - self._start_time,
        }
        atomic_write(
            self.config.checkpoint_file,
            json.dumps(data, indent=2) + "\n",
        )

    # ------------------------------------------------------------------
    # Query and update
    # ------------------------------------------------------------------

    def is_done(self, item_id: str) -> bool:
        """Return True if *item_id* has already been completed."""
        if not self.config.enabled:
            return False
        return item_id in self._completed

    def mark_done(self, item_id: str) -> None:
        """Record *item_id* as completed."""
        self._completed.add(item_id)

    def mark_done_batch(self, item_ids: list[str]) -> None:
        """Record multiple item IDs as completed and save."""
        self._completed.update(item_ids)
        self.save()

    @property
    def completed_count(self) -> int:
        """Number of completed items."""
        return len(self._completed)

    @property
    def total_items(self) -> int:
        return self._total_items

    @total_items.setter
    def total_items(self, value: int) -> None:
        self._total_items = value

    def filter_pending(self, item_ids: list[str]) -> list[str]:
        """Return only item IDs that are not yet completed."""
        if not self.config.enabled:
            return item_ids
        return [i for i in item_ids if i not in self._completed]
