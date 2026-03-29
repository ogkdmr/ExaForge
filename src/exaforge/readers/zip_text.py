"""Reader that loads text files from batched ZIP archives.

Designed for the workflow where ``exaforge preprocess --format zip`` has
packed a large directory of small files into a handful of ZIP archives.
At runtime, archives can optionally be extracted to fast node-local
storage (e.g. ``/tmp`` on Aurora) before reading, avoiding repeated
decompression and Lustre metadata overhead.

Supports the two-phase scan / read_by_ids workflow: file listings come
from the ZIP central directory (one seek per archive), and only the
requested entries are extracted.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from exaforge.config import ZipTextReaderConfig

from .base import BaseReader, InputItem

logger = logging.getLogger(__name__)


class ZipTextReader(BaseReader):
    """Load text files packed inside ZIP archives.

    If ``stage_dir`` is set in the config, each archive is extracted
    there once on first access.  Subsequent reads hit the fast local
    filesystem instead of the original archive / Lustre path.
    """

    def __init__(self, config: ZipTextReaderConfig) -> None:
        self.config = config
        self._index: Optional[dict[str, tuple[Path, str]]] = None
        self._staged_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def _discover_archives(self) -> list[Path]:
        """Return deduplicated, sorted list of matching ZIP files."""
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

    def _build_index(self) -> dict[str, tuple[Path, str]]:
        """Build ``{item_id: (archive_path, entry_name)}`` from ZIP central dirs.

        This only reads the ZIP table of contents — no file content is
        decompressed — so it stays cheap even for thousands of archives.
        """
        if self._index is not None:
            return self._index

        archives = self._discover_archives()
        index: dict[str, tuple[Path, str]] = {}

        for archive_path in archives:
            try:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    for entry in zf.namelist():
                        # Skip directories
                        if entry.endswith("/"):
                            continue
                        item_id = Path(entry).stem
                        index[item_id] = (archive_path, entry)
            except (zipfile.BadZipFile, OSError) as exc:
                logger.warning("Skipping bad archive %s: %s", archive_path, exc)

        logger.info(
            "Indexed %d items from %d ZIP archive(s)", len(index), len(archives)
        )
        self._index = index
        return index

    # ------------------------------------------------------------------
    # Staging (optional: extract to fast local storage)
    # ------------------------------------------------------------------

    def _ensure_staged(self) -> Optional[Path]:
        """Extract all archives to the staging directory (once).

        Returns the staging root so reads can use the extracted files
        instead of decompressing on every access.  Returns ``None`` if
        staging is disabled.
        """
        if self.config.stage_dir is None:
            return None

        if self._staged_dir is not None:
            return self._staged_dir

        stage_root = Path(
            tempfile.mkdtemp(
                prefix="exaforge_staged_",
                dir=str(self.config.stage_dir),
            )
        )
        logger.info("Staging archives to %s …", stage_root)

        for archive_path in self._discover_archives():
            try:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(stage_root)
            except (zipfile.BadZipFile, OSError) as exc:
                logger.warning(
                    "Skipping bad archive during staging %s: %s",
                    archive_path,
                    exc,
                )

        self._staged_dir = stage_root
        logger.info("Staging complete: %s", stage_root)
        return stage_root

    # ------------------------------------------------------------------
    # Two-phase API
    # ------------------------------------------------------------------

    def scan(self) -> list[str]:
        """Return item IDs from ZIP central directories (no decompression)."""
        return list(self._build_index().keys())

    def read_by_ids(self, ids: set[str]) -> list[InputItem]:
        """Load only items whose ID is in *ids*.

        If staging is enabled, reads from the pre-extracted directory.
        Otherwise, reads directly from the ZIP archives (grouping reads
        by archive to minimise open/close overhead).
        """
        index = self._build_index()
        stage_root = self._ensure_staged()

        items: list[InputItem] = []
        loaded = 0
        target = len(ids)

        if stage_root is not None:
            # Fast path: read from pre-extracted files
            for item_id in ids:
                entry = index.get(item_id)
                if entry is None:
                    logger.warning("ID %r not in index — skipping", item_id)
                    continue
                _, entry_name = entry
                staged_path = stage_root / entry_name
                try:
                    text = staged_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError as exc:
                    logger.warning("Skipping %s: %s", staged_path, exc)
                    continue

                items.append(
                    InputItem(
                        id=item_id,
                        text=text,
                        metadata={
                            "source_file": str(staged_path),
                            "archive_entry": entry_name,
                        },
                    )
                )
                loaded += 1
                if loaded % 500 == 0:
                    logger.info("Loaded %d / %d items …", loaded, target)
        else:
            # Direct path: read from ZIP archives, grouped by archive
            by_archive: dict[Path, list[tuple[str, str]]] = {}
            for item_id in ids:
                entry = index.get(item_id)
                if entry is None:
                    logger.warning("ID %r not in index — skipping", item_id)
                    continue
                archive_path, entry_name = entry
                by_archive.setdefault(archive_path, []).append(
                    (item_id, entry_name)
                )

            for archive_path, entries in by_archive.items():
                try:
                    with zipfile.ZipFile(archive_path, "r") as zf:
                        for item_id, entry_name in entries:
                            try:
                                raw = zf.read(entry_name)
                                text = raw.decode("utf-8", errors="replace")
                            except (KeyError, OSError) as exc:
                                logger.warning(
                                    "Skipping %s in %s: %s",
                                    entry_name,
                                    archive_path,
                                    exc,
                                )
                                continue

                            items.append(
                                InputItem(
                                    id=item_id,
                                    text=text,
                                    metadata={
                                        "source_file": str(archive_path),
                                        "archive_entry": entry_name,
                                    },
                                )
                            )
                            loaded += 1
                            if loaded % 500 == 0:
                                logger.info(
                                    "Loaded %d / %d items …", loaded, target
                                )
                except (zipfile.BadZipFile, OSError) as exc:
                    logger.warning(
                        "Skipping bad archive %s: %s", archive_path, exc
                    )

        return items

    # ------------------------------------------------------------------
    # Legacy full-read API
    # ------------------------------------------------------------------

    def read(self) -> list[InputItem]:
        all_ids = set(self._build_index().keys())
        return self.read_by_ids(all_ids)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_staging(self) -> None:
        """Remove the staging directory if it was created."""
        if self._staged_dir is not None and self._staged_dir.exists():
            logger.info("Cleaning up staging dir: %s", self._staged_dir)
            shutil.rmtree(self._staged_dir, ignore_errors=True)
            self._staged_dir = None
