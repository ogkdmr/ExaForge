"""Preprocessing utilities for ExaForge.

Converts a directory of individual text files into batched JSONL or ZIP
archives for more efficient I/O on parallel file systems like Lustre.

**JSONL mode** — packs K files per ``.jsonl`` shard::

    {"id": "file_stem", "text": "...", "source_file": "/original/path.mmd"}

**ZIP mode** — packs K files per ``.zip`` archive, preserving original
filenames.  At runtime the :class:`ZipTextReader` can optionally stage
archives to fast node-local storage (e.g. ``/tmp``) before reading.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
import sys
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _discover_files(
    input_dir: Path,
    glob_patterns: list[str],
) -> list[Path]:
    print(f"Discovering files in {input_dir} with patterns {glob_patterns}",
    file=sys.stdout,
    flush=True)
    """Collect and deduplicate matching files in deterministic order."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    paths: list[Path] = []
    for pattern in glob_patterns:
        paths.extend(input_dir.glob(pattern))

    print(f"Found {len(paths)} files", file=sys.stdout, flush=True)

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        if p not in seen and p.is_file():
            seen.add(p)
            unique.append(p)

    print(f"Removed {len(paths) - len(unique)} duplicate files", file=sys.stdout, flush=True)
    print(f"Batching {len(unique)} unique files", file=sys.stdout, flush=True)

    return unique


def preprocess_to_jsonl(
    input_dir: Path,
    output_dir: Path,
    glob_patterns: list[str],
    batch_size: int = 1000,
    *,
    base_name: str = "batch",
) -> int:
    """Pack text files into batched JSONL shards.

    Parameters
    ----------
    input_dir : Path
        Directory containing source text files.
    output_dir : Path
        Where to write the JSONL shard files.
    glob_patterns : list[str]
        Glob patterns to match source files (e.g. ``["*.mmd"]``).
    batch_size : int
        Number of records per JSONL file.
    base_name : str
        Prefix for shard filenames (``{base_name}_{0000}.jsonl``).

    Returns
    -------
    int
        Total number of files packed.
    """
    files = _discover_files(input_dir, glob_patterns)
    if not files:
        logger.warning("No files matched in %s", input_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(files)
    shard_idx = 0
    written = 0

    for start in tqdm(range(0, total, batch_size), total=total, desc="Preprocessing to JSONL"):
        batch = files[start : start + batch_size]
        shard_path = output_dir / f"{base_name}_{shard_idx:04d}.jsonl"

        with open(shard_path, "w", encoding="utf-8") as fp:
            for path in batch:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    logger.warning("Skipping %s: %s", path, exc)
                    continue

                record = {
                    "id": path.stem,
                    "text": text,
                    "source_file": str(path),
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        logger.info(
            "Shard %d: %d items -> %s", shard_idx, len(batch), shard_path
        )
        shard_idx += 1

    logger.info(
        "Preprocessing complete: %d files -> %d JSONL shard(s) in %s",
        written,
        shard_idx,
        output_dir,
    )
    return written


def preprocess_to_zip(
    input_dir: Path,
    output_dir: Path,
    glob_patterns: list[str],
    batch_size: int = 1000,
    *,
    base_name: str = "batch",
) -> int:
    """Pack text files into batched ZIP archives.

    Each archive contains up to *batch_size* files stored with their
    original basenames.  At runtime, the ZipTextReader can extract them
    to fast node-local storage for low-latency reads.

    Parameters
    ----------
    input_dir : Path
        Directory containing source text files.
    output_dir : Path
        Where to write the ZIP archive files.
    glob_patterns : list[str]
        Glob patterns to match source files.
    batch_size : int
        Number of files per ZIP archive.
    base_name : str
        Prefix for archive filenames (``{base_name}_{0000}.zip``).

    Returns
    -------
    int
        Total number of files packed.
    """
    files = _discover_files(input_dir, glob_patterns)
    if not files:
        logger.warning("No files matched in %s", input_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(files)
    shard_idx = 0
    written = 0

    for start in tqdm(range(0, total, batch_size), total=total, desc="Preprocessing to ZIP"):
        batch = files[start : start + batch_size]
        archive_path = output_dir / f"{base_name}_{shard_idx:04d}.zip"

        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for path in batch:
                try:
                    zf.write(path, arcname=path.name)
                    written += 1
                except OSError as exc:
                    logger.warning("Skipping %s: %s", path, exc)

        logger.info(
            "Archive %d: %d items -> %s", shard_idx, len(batch), archive_path
        )
        shard_idx += 1

    logger.info(
        "Preprocessing complete: %d files -> %d ZIP archive(s) in %s",
        written,
        shard_idx,
        output_dir,
    )
    return written
