"""Lustre-aware I/O helpers.

Provides utilities for working efficiently with Lustre parallel
filesystems (e.g. *flare* on Aurora).  Key principles:

* Minimise metadata operations — avoid many small open/close cycles.
* Use buffered writes — accumulate data in memory then flush.
* Atomic renames for crash-safe checkpointing.
* Optional stripe tuning via ``lfs setstripe``.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def atomic_write(path: Path, data: str, *, fsync: bool = True) -> None:
    """Write *data* to *path* atomically via a temp-file rename.

    On Lustre (and POSIX in general) ``os.rename`` within the same
    directory is atomic, so readers never see a half-written file.

    Parameters
    ----------
    path : Path
        Destination file path.
    data : str
        UTF-8 string to write.
    fsync : bool
        If True, call ``os.fsync`` before renaming to ensure data
        is durable on disk (recommended for checkpoints).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as fp:
            fp.write(data)
            fp.flush()
            if fsync:
                os.fsync(fp.fileno())
        os.rename(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def set_stripe(
    path: Path,
    count: int = -1,
    size: str = "4M",
) -> bool:
    """Set Lustre striping on a directory.

    Parameters
    ----------
    path : Path
        Directory to stripe (must already exist).
    count : int
        Number of OSTs to stripe across.  ``-1`` = all available.
    size : str
        Stripe size (e.g. ``"1M"``, ``"4M"``).

    Returns
    -------
    bool
        True if the command succeeded, False otherwise (e.g. not on
        Lustre, ``lfs`` not available).
    """
    try:
        result = subprocess.run(
            ["lfs", "setstripe", "-c", str(count), "-S", size, str(path)],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def ensure_output_dir(
    path: Path,
    *,
    stripe_count: int = 0,
    stripe_size: str = "4M",
) -> Path:
    """Create an output directory and optionally set Lustre striping.

    Parameters
    ----------
    path : Path
        Directory to create.
    stripe_count : int
        Lustre stripe count (0 = skip striping).
    stripe_size : str
        Lustre stripe size.

    Returns
    -------
    Path
        The resolved, absolute directory path.
    """
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    if stripe_count:
        set_stripe(path, count=stripe_count, size=stripe_size)
    return path
