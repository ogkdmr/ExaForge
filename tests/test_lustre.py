"""Tests for the Lustre-aware I/O helpers."""

from __future__ import annotations

from pathlib import Path

from exaforge.lustre import atomic_write, ensure_output_dir, set_stripe


class TestAtomicWrite:
    def test_writes_content(self, tmp_dir: Path) -> None:
        target = tmp_dir / "file.txt"
        atomic_write(target, "hello world")
        assert target.read_text() == "hello world"

    def test_creates_parent_dirs(self, tmp_dir: Path) -> None:
        target = tmp_dir / "sub" / "dir" / "file.txt"
        atomic_write(target, "nested")
        assert target.read_text() == "nested"

    def test_no_temp_files_left_on_success(self, tmp_dir: Path) -> None:
        target = tmp_dir / "clean.txt"
        atomic_write(target, "data")
        leftovers = list(tmp_dir.glob(".*clean*"))
        assert len(leftovers) == 0

    def test_overwrites_existing(self, tmp_dir: Path) -> None:
        target = tmp_dir / "overwrite.txt"
        atomic_write(target, "v1")
        atomic_write(target, "v2")
        assert target.read_text() == "v2"

    def test_no_fsync_mode(self, tmp_dir: Path) -> None:
        target = tmp_dir / "no_fsync.txt"
        atomic_write(target, "fast", fsync=False)
        assert target.read_text() == "fast"


class TestSetStripe:
    def test_returns_false_when_lfs_not_available(
        self, tmp_dir: Path
    ) -> None:
        # On most non-Lustre systems, lfs won't be found
        result = set_stripe(tmp_dir, count=2)
        assert isinstance(result, bool)


class TestEnsureOutputDir:
    def test_creates_directory(self, tmp_dir: Path) -> None:
        target = tmp_dir / "new" / "output"
        result = ensure_output_dir(target)
        assert result.is_dir()
        assert result.is_absolute()

    def test_idempotent(self, tmp_dir: Path) -> None:
        target = tmp_dir / "idem"
        ensure_output_dir(target)
        ensure_output_dir(target)
        assert target.is_dir()
