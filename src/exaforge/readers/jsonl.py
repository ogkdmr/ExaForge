"""Reader that bulk-loads records from JSONL files.

Each line of a JSONL file is expected to be a JSON object.  The reader
extracts the ``text`` and ``id`` fields (configurable) and carries the
full original record as metadata so it can be merged back into the
output.

Supports the two-phase scan / read_by_ids workflow so the orchestrator
can discover IDs cheaply, filter via the checkpoint, and only parse the
lines that will actually be processed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from exaforge.config import JsonlReaderConfig

from .base import BaseReader, InputItem

logger = logging.getLogger(__name__)


class JsonlReader(BaseReader):
    """Load JSONL files from a directory, one InputItem per JSON line.

    The reader caches a lightweight index on first access (mapping
    item IDs to ``(file_path, line_number)`` positions) so that
    :meth:`scan` is cheap and :meth:`read_by_ids` only parses the
    lines it needs.
    """

    def __init__(self, config: JsonlReaderConfig) -> None:
        self.config = config
        self._index: Optional[dict[str, tuple[Path, int]]] = None

    # ------------------------------------------------------------------
    # Discovery (cached)
    # ------------------------------------------------------------------

    def _discover_files(self) -> list[Path]:
        """Return deduplicated, sorted list of matching JSONL files."""
        input_dir = self.config.input_dir
        if not input_dir.is_dir():
            raise FileNotFoundError(
                f"Input directory does not exist: {input_dir}"
            )

        paths: list[Path] = []
        for pattern in self.config.glob_patterns:
            paths.extend(sorted(input_dir.glob(pattern)))

        seen: set[Path] = set()
        unique: list[Path] = []
        for p in paths:
            if p not in seen and p.is_file():
                seen.add(p)
                unique.append(p)
        return unique

    def _build_index(self) -> dict[str, tuple[Path, int]]:
        """Build a mapping of ``{item_id: (file_path, line_number)}``.

        This reads every file line-by-line but only parses the JSON
        enough to extract the ``id_field``.  Content is NOT retained,
        so memory stays low for large datasets.
        """
        if self._index is not None:
            return self._index

        files = self._discover_files()
        index: dict[str, tuple[Path, int]] = {}

        for p in files:
            raw = p.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(raw.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                item_id = str(
                    record.get(
                        self.config.id_field, f"{p.stem}:{line_no}"
                    )
                )
                index[item_id] = (p, line_no)

        logger.info("Indexed %d items from %d JSONL file(s)", len(index), len(files))
        self._index = index
        return index

    # ------------------------------------------------------------------
    # Two-phase API (preferred by the orchestrator)
    # ------------------------------------------------------------------

    def scan(self) -> list[str]:
        """Return item IDs without loading text content."""
        return list(self._build_index().keys())

    def read_by_ids(self, ids: set[str]) -> list[InputItem]:
        """Load only items whose ID is in *ids*.

        Re-reads the JSONL files but skips lines whose ID is not in the
        requested set, avoiding full deserialization of every record.
        """
        index = self._build_index()
        # Group requested IDs by file for sequential I/O
        by_file: dict[Path, set[int]] = {}
        for item_id in ids:
            entry = index.get(item_id)
            if entry is None:
                logger.warning("ID %r not found in index — skipping", item_id)
                continue
            fpath, line_no = entry
            by_file.setdefault(fpath, set()).add(line_no)

        items: list[InputItem] = []
        loaded = 0
        target = len(ids)

        for fpath, line_numbers in by_file.items():
            raw = fpath.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(raw.splitlines(), start=1):
                if line_no not in line_numbers:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = str(record.get(self.config.text_field, ""))
                item_id = str(
                    record.get(
                        self.config.id_field, f"{fpath.stem}:{line_no}"
                    )
                )
                items.append(
                    InputItem(
                        id=item_id,
                        text=text,
                        metadata={
                            "source_file": str(fpath),
                            "line_number": line_no,
                            "original_record": record,
                        },
                    )
                )
                loaded += 1
                if loaded % 500 == 0:
                    logger.info("Loaded %d / %d items …", loaded, target)

        return items

    # ------------------------------------------------------------------
    # Legacy full-read API (BaseReader interface)
    # ------------------------------------------------------------------

    def read(self) -> list[InputItem]:
        all_ids = set(self._build_index().keys())
        return self.read_by_ids(all_ids)
