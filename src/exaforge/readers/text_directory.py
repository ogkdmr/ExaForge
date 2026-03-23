"""Reader that bulk-loads plain-text files from a directory.

Designed to minimise metadata operations on Lustre: file paths are
collected in a single ``glob`` sweep, then contents are read in one
pass per file.
"""

from __future__ import annotations

from pathlib import Path

from exaforge.config import TextDirectoryReaderConfig

from .base import BaseReader, InputItem


class TextDirectoryReader(BaseReader):
    """Load every matching text file from a directory as an InputItem."""

    def __init__(self, config: TextDirectoryReaderConfig) -> None:
        self.config = config

    def read(self) -> list[InputItem]:
        input_dir = self.config.input_dir
        if not input_dir.is_dir():
            raise FileNotFoundError(
                f"Input directory does not exist: {input_dir}"
            )

        paths: list[Path] = []
        for pattern in self.config.glob_patterns:
            paths.extend(sorted(input_dir.glob(pattern)))

        # De-duplicate (overlapping globs) while preserving order
        seen: set[Path] = set()
        unique: list[Path] = []
        for p in paths:
            if p not in seen and p.is_file():
                seen.add(p)
                unique.append(p)

        items: list[InputItem] = []
        for p in unique:
            text = p.read_text(encoding="utf-8", errors="replace")
            items.append(
                InputItem(
                    id=p.stem,
                    text=text,
                    metadata={"source_file": str(p)},
                )
            )
        return items
