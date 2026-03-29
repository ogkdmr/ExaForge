"""Preprocessing utilities for ExaForge.

Converts a directory of individual text files into batched JSONL or ZIP
archives for more efficient I/O on parallel file systems like Lustre.

**JSONL mode** — packs K files per ``.jsonl`` shard::

    {"id": "file_stem", "text": "...", "source_file": "/original/path.mmd"}

**ZIP mode** — packs K files per ``.zip`` archive, preserving original
filenames.  At runtime the :class:`ZipTextReader` can optionally stage
archives to fast node-local storage (e.g. ``/tmp``) before reading.

Both modes support concurrent shard writing via a thread pool
(``workers`` parameter / ``-w`` CLI flag).  Each worker independently
reads its assigned files and writes one shard, so the thread pool
saturates Lustre I/O bandwidth across multiple concurrent streams.
"""

from __future__ import annotations

import json
import logging
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_files(
    input_dir: Path,
    glob_patterns: list[str],
    deduplicate: bool = False,
) -> list[Path]:
    print(f"Discovering files in {input_dir} with patterns {glob_patterns}",
          file=sys.stdout, flush=True)
    """Collect matching files in deterministic order, optionally deduplicated."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    paths: list[Path] = []
    for pattern in glob_patterns:
        paths.extend(input_dir.glob(pattern))

    print(f"Found {len(paths)} files", file=sys.stdout, flush=True)

    if not deduplicate:
        print(f"Skipping deduplication for {len(paths)} files",
              file=sys.stdout, flush=True)
        return paths

    print("Deduplicating files", file=sys.stdout, flush=True)
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        if p not in seen and p.is_file():
            seen.add(p)
            unique.append(p)

    print(f"Removed {len(paths) - len(unique)} duplicate files",
          file=sys.stdout, flush=True)
    print(f"Batching {len(unique)} unique files", file=sys.stdout, flush=True)
    return unique


# ---------------------------------------------------------------------------
# Per-shard worker functions (called inside threads)
# ---------------------------------------------------------------------------

def _write_jsonl_shard(
    shard_path: Path,
    batch: list[Path],
) -> int:
    """Read *batch* files and write one JSONL shard.  Returns records written."""
    written = 0
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
    return written


def _write_zip_shard(
    archive_path: Path,
    batch: list[Path],
) -> int:
    """Pack *batch* files into one ZIP archive.  Returns files written."""
    written = 0
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in batch:
            try:
                zf.write(path, arcname=path.name)
                written += 1
            except OSError as exc:
                logger.warning("Skipping %s: %s", path, exc)
    return written


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_to_jsonl(
    input_dir: Path,
    output_dir: Path,
    glob_patterns: list[str],
    batch_size: int = 1000,
    deduplicate: bool = False,
    workers: int = 1,
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
    deduplicate : bool
        Remove duplicate paths before batching.
    workers : int
        Number of concurrent writer threads.  Each thread writes one
        shard at a time.  Use ``1`` for serial behaviour.
    base_name : str
        Prefix for shard filenames (``{base_name}_{0000}.jsonl``).

    Returns
    -------
    int
        Total number of records written.
    """
    files = _discover_files(input_dir, glob_patterns, deduplicate)
    if not files:
        logger.warning("No files matched in %s", input_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(files)
    num_shards = ceil(total / batch_size)

    # Build work items: (shard_idx, shard_path, batch_slice)
    work = [
        (
            shard_idx,
            output_dir / f"{base_name}_{shard_idx:04d}.jsonl",
            files[start : start + batch_size],
        )
        for shard_idx, start in enumerate(range(0, total, batch_size))
    ]

    written = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_write_jsonl_shard, shard_path, batch): shard_idx
            for shard_idx, shard_path, batch in work
        }
        with tqdm(total=num_shards, desc="Preprocessing to JSONL") as pbar:
            for future in as_completed(futures):
                shard_idx = futures[future]
                count = future.result()
                written += count
                logger.info("Shard %d: %d records written", shard_idx, count)
                pbar.update(1)

    logger.info(
        "Preprocessing complete: %d files -> %d JSONL shard(s) in %s",
        written, num_shards, output_dir,
    )
    return written


def preprocess_to_zip(
    input_dir: Path,
    output_dir: Path,
    glob_patterns: list[str],
    batch_size: int = 1000,
    deduplicate: bool = False,
    workers: int = 1,
    *,
    base_name: str = "batch",
) -> int:
    """Pack text files into batched ZIP archives.

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
    deduplicate : bool
        Remove duplicate paths before batching.
    workers : int
        Number of concurrent writer threads.
    base_name : str
        Prefix for archive filenames (``{base_name}_{0000}.zip``).

    Returns
    -------
    int
        Total number of files packed.
    """
    files = _discover_files(input_dir, glob_patterns, deduplicate)
    if not files:
        logger.warning("No files matched in %s", input_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(files)
    num_shards = ceil(total / batch_size)

    work = [
        (
            shard_idx,
            output_dir / f"{base_name}_{shard_idx:04d}.zip",
            files[start : start + batch_size],
        )
        for shard_idx, start in enumerate(range(0, total, batch_size))
    ]

    written = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_write_zip_shard, archive_path, batch): shard_idx
            for shard_idx, archive_path, batch in work
        }
        with tqdm(total=num_shards, desc="Preprocessing to ZIP") as pbar:
            for future in as_completed(futures):
                shard_idx = futures[future]
                count = future.result()
                written += count
                logger.info("Archive %d: %d files written", shard_idx, count)
                pbar.update(1)

    logger.info(
        "Preprocessing complete: %d files -> %d ZIP archive(s) in %s",
        written, num_shards, output_dir,
    )
    return written
