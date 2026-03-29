"""Tests for the ExaForge preprocessing pipeline and new readers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exaforge.cli import app
from exaforge.config import JsonlReaderConfig, ZipTextReaderConfig
from exaforge.preprocess import preprocess_to_jsonl, preprocess_to_zip
from exaforge.readers import ZipTextReader, get_reader
from exaforge.readers.jsonl import JsonlReader

runner = CliRunner()


# ------------------------------------------------------------------
# preprocess_to_jsonl
# ------------------------------------------------------------------


class TestPreprocessToJsonl:
    def test_creates_shards(self, sample_mmd_dir: Path, tmp_dir: Path) -> None:
        out = tmp_dir / "jsonl_out"
        total = preprocess_to_jsonl(sample_mmd_dir, out, ["*.mmd"], batch_size=5)
        assert total == 12
        shards = sorted(out.glob("batch_*.jsonl"))
        # 12 files / 5 per shard = 3 shards (5 + 5 + 2)
        assert len(shards) == 3

    def test_shard_contents_valid_jsonl(
        self, sample_mmd_dir: Path, tmp_dir: Path
    ) -> None:
        out = tmp_dir / "jsonl_out"
        preprocess_to_jsonl(sample_mmd_dir, out, ["*.mmd"], batch_size=5)
        for shard in out.glob("batch_*.jsonl"):
            for line in shard.read_text().strip().splitlines():
                record = json.loads(line)
                assert "id" in record
                assert "text" in record
                assert "source_file" in record
                assert len(record["text"]) > 0

    def test_custom_base_name(
        self, sample_mmd_dir: Path, tmp_dir: Path
    ) -> None:
        out = tmp_dir / "jsonl_out"
        preprocess_to_jsonl(
            sample_mmd_dir, out, ["*.mmd"], batch_size=100, base_name="papers"
        )
        shards = list(out.glob("papers_*.jsonl"))
        assert len(shards) == 1
        assert shards[0].name == "papers_0000.jsonl"

    def test_empty_dir_returns_zero(self, tmp_dir: Path) -> None:
        empty = tmp_dir / "empty"
        empty.mkdir()
        out = tmp_dir / "out"
        total = preprocess_to_jsonl(empty, out, ["*.mmd"])
        assert total == 0

    def test_missing_dir_raises(self, tmp_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            preprocess_to_jsonl(tmp_dir / "nope", tmp_dir / "out", ["*.mmd"])


# ------------------------------------------------------------------
# preprocess_to_zip
# ------------------------------------------------------------------


class TestPreprocessToZip:
    def test_creates_archives(
        self, sample_mmd_dir: Path, tmp_dir: Path
    ) -> None:
        out = tmp_dir / "zip_out"
        total = preprocess_to_zip(sample_mmd_dir, out, ["*.mmd"], batch_size=5)
        assert total == 12
        archives = sorted(out.glob("batch_*.zip"))
        assert len(archives) == 3

    def test_archive_contents(
        self, sample_mmd_dir: Path, tmp_dir: Path
    ) -> None:
        out = tmp_dir / "zip_out"
        preprocess_to_zip(sample_mmd_dir, out, ["*.mmd"], batch_size=5)
        all_entries: list[str] = []
        for archive in out.glob("batch_*.zip"):
            with zipfile.ZipFile(archive, "r") as zf:
                all_entries.extend(zf.namelist())
        assert len(all_entries) == 12
        assert "paper_000.mmd" in all_entries

    def test_empty_dir_returns_zero(self, tmp_dir: Path) -> None:
        empty = tmp_dir / "empty"
        empty.mkdir()
        out = tmp_dir / "out"
        total = preprocess_to_zip(empty, out, ["*.mmd"])
        assert total == 0


# ------------------------------------------------------------------
# Enhanced JsonlReader (scan / read_by_ids)
# ------------------------------------------------------------------


class TestJsonlReaderTwoPhase:
    def test_scan_returns_ids(self, preprocessed_jsonl_dir: Path) -> None:
        cfg = JsonlReaderConfig(
            input_dir=preprocessed_jsonl_dir,
            glob_patterns=["*.jsonl"],
        )
        reader = JsonlReader(cfg)
        ids = reader.scan()
        assert len(ids) == 12
        assert "paper_000" in ids

    def test_read_by_ids_loads_subset(
        self, preprocessed_jsonl_dir: Path
    ) -> None:
        cfg = JsonlReaderConfig(
            input_dir=preprocessed_jsonl_dir,
            glob_patterns=["*.jsonl"],
        )
        reader = JsonlReader(cfg)
        items = reader.read_by_ids({"paper_001", "paper_005", "paper_010"})
        assert len(items) == 3
        ids = {it.id for it in items}
        assert ids == {"paper_001", "paper_005", "paper_010"}

    def test_scan_then_read_all_matches_full_read(
        self, preprocessed_jsonl_dir: Path
    ) -> None:
        cfg = JsonlReaderConfig(
            input_dir=preprocessed_jsonl_dir,
            glob_patterns=["*.jsonl"],
        )
        reader = JsonlReader(cfg)
        all_ids = set(reader.scan())
        selective = reader.read_by_ids(all_ids)
        full = JsonlReader(cfg).read()
        assert {it.id for it in selective} == {it.id for it in full}

    def test_read_by_ids_empty_set(
        self, preprocessed_jsonl_dir: Path
    ) -> None:
        cfg = JsonlReaderConfig(
            input_dir=preprocessed_jsonl_dir,
            glob_patterns=["*.jsonl"],
        )
        items = JsonlReader(cfg).read_by_ids(set())
        assert items == []

    def test_index_is_cached(self, preprocessed_jsonl_dir: Path) -> None:
        cfg = JsonlReaderConfig(
            input_dir=preprocessed_jsonl_dir,
            glob_patterns=["*.jsonl"],
        )
        reader = JsonlReader(cfg)
        reader.scan()
        reader.scan()
        assert reader._index is not None


# ------------------------------------------------------------------
# ZipTextReader
# ------------------------------------------------------------------


class TestZipTextReader:
    def test_scan_returns_ids(self, preprocessed_zip_dir: Path) -> None:
        cfg = ZipTextReaderConfig(
            input_dir=preprocessed_zip_dir,
            glob_patterns=["*.zip"],
        )
        reader = ZipTextReader(cfg)
        ids = reader.scan()
        assert len(ids) == 12
        assert "paper_000" in ids

    def test_read_by_ids_direct(self, preprocessed_zip_dir: Path) -> None:
        """Read directly from zip archives (no staging)."""
        cfg = ZipTextReaderConfig(
            input_dir=preprocessed_zip_dir,
            glob_patterns=["*.zip"],
        )
        reader = ZipTextReader(cfg)
        items = reader.read_by_ids({"paper_002", "paper_007"})
        assert len(items) == 2
        ids = {it.id for it in items}
        assert ids == {"paper_002", "paper_007"}
        for it in items:
            assert "Paper" in it.text
            assert "archive_entry" in it.metadata

    def test_read_by_ids_staged(
        self, preprocessed_zip_dir: Path, tmp_dir: Path
    ) -> None:
        """Read via staging to a temp directory."""
        stage = tmp_dir / "staging"
        stage.mkdir()
        cfg = ZipTextReaderConfig(
            input_dir=preprocessed_zip_dir,
            glob_patterns=["*.zip"],
            stage_dir=stage,
        )
        reader = ZipTextReader(cfg)
        items = reader.read_by_ids({"paper_003", "paper_011"})
        assert len(items) == 2
        # Verify staged files exist
        assert reader._staged_dir is not None
        assert reader._staged_dir.exists()
        reader.cleanup_staging()
        assert reader._staged_dir is None

    def test_full_read(self, preprocessed_zip_dir: Path) -> None:
        cfg = ZipTextReaderConfig(
            input_dir=preprocessed_zip_dir,
            glob_patterns=["*.zip"],
        )
        items = ZipTextReader(cfg).read()
        assert len(items) == 12

    def test_scan_then_read_matches_full_read(
        self, preprocessed_zip_dir: Path
    ) -> None:
        cfg = ZipTextReaderConfig(
            input_dir=preprocessed_zip_dir,
            glob_patterns=["*.zip"],
        )
        reader = ZipTextReader(cfg)
        all_ids = set(reader.scan())
        selective = reader.read_by_ids(all_ids)
        full = ZipTextReader(cfg).read()
        assert {it.id for it in selective} == {it.id for it in full}

    def test_read_by_ids_empty_set(
        self, preprocessed_zip_dir: Path
    ) -> None:
        cfg = ZipTextReaderConfig(
            input_dir=preprocessed_zip_dir,
            glob_patterns=["*.zip"],
        )
        items = ZipTextReader(cfg).read_by_ids(set())
        assert items == []


# ------------------------------------------------------------------
# Reader registry with zip_text
# ------------------------------------------------------------------


class TestReaderRegistryZipText:
    def test_get_reader_zip_text(self, tmp_dir: Path) -> None:
        reader = get_reader(
            {"name": "zip_text", "input_dir": str(tmp_dir)}
        )
        assert isinstance(reader, ZipTextReader)

    def test_get_reader_from_config_object(self, tmp_dir: Path) -> None:
        cfg = ZipTextReaderConfig(input_dir=tmp_dir)
        reader = get_reader(cfg)
        assert isinstance(reader, ZipTextReader)


# ------------------------------------------------------------------
# CLI preprocess command
# ------------------------------------------------------------------


class TestPreprocessCommand:
    def test_preprocess_jsonl(
        self, sample_mmd_dir: Path, tmp_dir: Path
    ) -> None:
        out = tmp_dir / "cli_jsonl"
        result = runner.invoke(
            app,
            [
                "preprocess",
                "-i", str(sample_mmd_dir),
                "-o", str(out),
                "-f", "jsonl",
                "-g", "*.mmd",
                "-b", "5",
            ],
        )
        assert result.exit_code == 0
        assert "Preprocessed 12" in result.output
        assert len(list(out.glob("batch_*.jsonl"))) == 3

    def test_preprocess_zip(
        self, sample_mmd_dir: Path, tmp_dir: Path
    ) -> None:
        out = tmp_dir / "cli_zip"
        result = runner.invoke(
            app,
            [
                "preprocess",
                "-i", str(sample_mmd_dir),
                "-o", str(out),
                "-f", "zip",
                "-g", "*.mmd",
                "-b", "5",
            ],
        )
        assert result.exit_code == 0
        assert "Preprocessed 12" in result.output
        assert len(list(out.glob("batch_*.zip"))) == 3

    def test_preprocess_jsonl_multithreaded(
        self, sample_mmd_dir: Path, tmp_dir: Path
    ) -> None:
        out = tmp_dir / "cli_jsonl_mt"
        result = runner.invoke(
            app,
            [
                "preprocess",
                "-i", str(sample_mmd_dir),
                "-o", str(out),
                "-f", "jsonl",
                "-g", "*.mmd",
                "-b", "5",
                "-w", "4",
            ],
        )
        assert result.exit_code == 0
        assert len(list(out.glob("batch_*.jsonl"))) == 3

    def test_preprocess_unknown_format(
        self, sample_mmd_dir: Path, tmp_dir: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "preprocess",
                "-i", str(sample_mmd_dir),
                "-o", str(tmp_dir / "out"),
                "-f", "parquet",
                "-g", "*.mmd",
            ],
        )
        assert result.exit_code != 0
