"""Buffered JSONL writer optimised for Lustre.

Accumulates records in memory and flushes them in large sequential
writes to avoid overwhelming Lustre metadata servers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from exaforge.config import JsonlWriterConfig
from exaforge.lustre import ensure_output_dir

from .base import BaseWriter, OutputRecord


class JsonlWriter(BaseWriter):
    """Write OutputRecords as newline-delimited JSON.

    Records are buffered in memory and flushed when the buffer reaches
    ``config.buffer_size`` or when :meth:`flush` / :meth:`close` is
    called explicitly.
    """

    def __init__(self, config: JsonlWriterConfig) -> None:
        self.config = config
        self._buffer: list[str] = []
        self._chunk_idx = 0
        self._output_dir: Optional[Path] = None
        self._file_handle: Optional[object] = None

    @property
    def output_dir(self) -> Path:
        if self._output_dir is None:
            self._output_dir = ensure_output_dir(self.config.output_dir)
        return self._output_dir

    def _current_path(self) -> Path:
        return (
            self.output_dir
            / f"{self.config.base_name}_{self._chunk_idx:04d}.jsonl"
        )

    def write(self, records: list[OutputRecord]) -> None:
        for rec in records:
            entry: dict = {
                "id": rec.id,
                "response": rec.response,
            }
            entry.update(rec.metadata)
            self._buffer.append(json.dumps(entry, ensure_ascii=False))

        if len(self._buffer) >= self.config.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        path = self._current_path()
        with open(path, "a", encoding="utf-8") as fp:
            fp.write("\n".join(self._buffer))
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())

        self._buffer.clear()

    def close(self) -> None:
        self.flush()

    def rotate(self) -> None:
        """Flush the current chunk and start writing to a new file."""
        self.flush()
        self._chunk_idx += 1
