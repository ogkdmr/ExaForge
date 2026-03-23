"""Reader that bulk-loads records from JSONL files.

Each line of a JSONL file is expected to be a JSON object.  The reader
extracts the ``text`` and ``id`` fields (configurable) and carries the
full original record as metadata so it can be merged back into the
output.
"""

from __future__ import annotations

import json
from pathlib import Path

from exaforge.config import JsonlReaderConfig

from .base import BaseReader, InputItem


class JsonlReader(BaseReader):
    """Load JSONL files from a directory, one InputItem per JSON line."""

    def __init__(self, config: JsonlReaderConfig) -> None:
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

        seen: set[Path] = set()
        unique: list[Path] = []
        for p in paths:
            if p not in seen and p.is_file():
                seen.add(p)
                unique.append(p)

        items: list[InputItem] = []
        for p in unique:
            raw = p.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(raw.splitlines(), start=1):
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
                        self.config.id_field, f"{p.stem}:{line_no}"
                    )
                )
                items.append(
                    InputItem(
                        id=item_id,
                        text=text,
                        metadata={
                            "source_file": str(p),
                            "line_number": line_no,
                            "original_record": record,
                        },
                    )
                )
        return items
