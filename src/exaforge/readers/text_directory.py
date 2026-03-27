"""Reader that bulk-loads plain-text files from a directory.

Designed to minimise metadata operations on Lustre: file paths are
collected in a single ``glob`` sweep, then contents are read in one
pass per file.

Supports a two-phase workflow via :meth:`scan` / :meth:`read_by_ids`
so the orchestrator can discover item IDs cheaply, filter through the
checkpoint, and only load the files that will actually be processed.
On a Lustre directory with 58 000 files this avoids tens of minutes of
unnecessary I/O.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from exaforge.config import TextDirectoryReaderConfig

from .base import BaseReader, InputItem

logger = logging.getLogger(__name__)


class TextDirectoryReader(BaseReader):
    """Load every matching text file from a directory as an InputItem."""

    def __init__(self, config: TextDirectoryReaderConfig) -> None:
        self.config = config
        self._path_map: Optional[dict[str, Path]] = None

    def _discover(self) -> dict[str, Path]:
        """Build an ordered mapping of ``{stem: path}`` (cached).

        Only performs a directory listing + dedup — no file content is
        read, so this stays cheap even on Lustre with tens of thousands
        of files.
        """
        if self._path_map is not None:
            return self._path_map

        input_dir = self.config.input_dir
        if not input_dir.is_dir():
            raise FileNotFoundError(
                f"Input directory does not exist: {input_dir}"
            )

        logger.info("Scanning %s …", input_dir)

        paths: list[Path] = []
        for pattern in self.config.glob_patterns:
            paths.extend(sorted(input_dir.glob(pattern)))

        seen: set[Path] = set()
        path_map: dict[str, Path] = {}
        for p in paths:
            if p not in seen and p.is_file():
                seen.add(p)
                path_map[p.stem] = p

        logger.info("Discovered %d items", len(path_map))
        self._path_map = path_map
        return path_map

    # ------------------------------------------------------------------
    # Two-phase API (preferred by the orchestrator)
    # ------------------------------------------------------------------

    def scan(self) -> list[str]:
        """Return item IDs without reading file content."""
        return list(self._discover().keys())

    def read_by_ids(self, ids: set[str]) -> list[InputItem]:
        """Load only items whose ID is in *ids*."""
        path_map = self._discover()
        items: list[InputItem] = []
        loaded = 0
        target = len(ids)
        for item_id, path in path_map.items():
            if item_id not in ids:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Skipping %s: %s", path, exc)
                continue
            items.append(
                InputItem(
                    id=item_id,
                    text=text,
                    metadata={"source_file": str(path)},
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
        all_ids = set(self._discover().keys())
        return self.read_by_ids(all_ids)
