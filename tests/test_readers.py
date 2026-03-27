"""Tests for the ExaForge input readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exaforge.config import JsonlReaderConfig, TextDirectoryReaderConfig
from exaforge.readers import (
    InputItem,
    JsonlReader,
    TextDirectoryReader,
    get_reader,
)


class TestTextDirectoryReader:
    def test_reads_all_txt_files(self, sample_texts: list[Path]) -> None:
        cfg = TextDirectoryReaderConfig(
            input_dir=sample_texts[0].parent,
            glob_patterns=["*.txt"],
        )
        reader = TextDirectoryReader(cfg)
        items = reader.read()
        assert len(items) == 5
        assert all(isinstance(i, InputItem) for i in items)

    def test_ids_are_stems(self, sample_texts: list[Path]) -> None:
        cfg = TextDirectoryReaderConfig(
            input_dir=sample_texts[0].parent,
        )
        items = TextDirectoryReader(cfg).read()
        ids = {i.id for i in items}
        assert "paper_000" in ids
        assert "paper_004" in ids

    def test_metadata_contains_source_file(
        self, sample_texts: list[Path]
    ) -> None:
        cfg = TextDirectoryReaderConfig(
            input_dir=sample_texts[0].parent,
        )
        items = TextDirectoryReader(cfg).read()
        for item in items:
            assert "source_file" in item.metadata

    def test_missing_dir_raises(self, tmp_dir: Path) -> None:
        cfg = TextDirectoryReaderConfig(
            input_dir=tmp_dir / "nonexistent",
        )
        with pytest.raises(FileNotFoundError):
            TextDirectoryReader(cfg).read()

    def test_deduplicates_overlapping_globs(
        self, sample_texts: list[Path]
    ) -> None:
        cfg = TextDirectoryReaderConfig(
            input_dir=sample_texts[0].parent,
            glob_patterns=["*.txt", "paper_*.txt"],
        )
        items = TextDirectoryReader(cfg).read()
        assert len(items) == 5

    def test_empty_dir_returns_empty_list(self, tmp_dir: Path) -> None:
        cfg = TextDirectoryReaderConfig(input_dir=tmp_dir)
        items = TextDirectoryReader(cfg).read()
        assert items == []

    def test_scan_returns_ids_only(self, sample_texts: list[Path]) -> None:
        cfg = TextDirectoryReaderConfig(
            input_dir=sample_texts[0].parent,
            glob_patterns=["*.txt"],
        )
        reader = TextDirectoryReader(cfg)
        ids = reader.scan()
        assert len(ids) == 5
        assert all(isinstance(i, str) for i in ids)
        assert "paper_000" in ids

    def test_read_by_ids_loads_subset(
        self, sample_texts: list[Path]
    ) -> None:
        cfg = TextDirectoryReaderConfig(
            input_dir=sample_texts[0].parent,
            glob_patterns=["*.txt"],
        )
        reader = TextDirectoryReader(cfg)
        items = reader.read_by_ids({"paper_001", "paper_003"})
        assert len(items) == 2
        ids = {it.id for it in items}
        assert ids == {"paper_001", "paper_003"}
        assert all(it.text for it in items)
        assert all("source_file" in it.metadata for it in items)

    def test_scan_then_read_by_ids_matches_full_read(
        self, sample_texts: list[Path]
    ) -> None:
        cfg = TextDirectoryReaderConfig(
            input_dir=sample_texts[0].parent,
            glob_patterns=["*.txt"],
        )
        reader = TextDirectoryReader(cfg)
        all_ids = set(reader.scan())
        items_selective = reader.read_by_ids(all_ids)
        items_full = TextDirectoryReader(cfg).read()
        assert {it.id for it in items_selective} == {it.id for it in items_full}
        for sel in items_selective:
            full = next(f for f in items_full if f.id == sel.id)
            assert sel.text == full.text

    def test_read_by_ids_empty_set(
        self, sample_texts: list[Path]
    ) -> None:
        cfg = TextDirectoryReaderConfig(
            input_dir=sample_texts[0].parent,
            glob_patterns=["*.txt"],
        )
        items = TextDirectoryReader(cfg).read_by_ids(set())
        assert items == []

    def test_discovery_is_cached(
        self, sample_texts: list[Path]
    ) -> None:
        cfg = TextDirectoryReaderConfig(
            input_dir=sample_texts[0].parent,
            glob_patterns=["*.txt"],
        )
        reader = TextDirectoryReader(cfg)
        ids1 = reader.scan()
        ids2 = reader.scan()
        assert ids1 == ids2
        assert reader._path_map is not None


class TestJsonlReader:
    def test_reads_all_records(self, sample_jsonl: Path) -> None:
        cfg = JsonlReaderConfig(
            input_dir=sample_jsonl.parent,
            glob_patterns=["*.jsonl"],
        )
        items = JsonlReader(cfg).read()
        assert len(items) == 10

    def test_extracts_text_field(self, sample_jsonl: Path) -> None:
        cfg = JsonlReaderConfig(
            input_dir=sample_jsonl.parent,
            text_field="text",
        )
        items = JsonlReader(cfg).read()
        assert all("Sample document text" in i.text for i in items)

    def test_extracts_id_field(self, sample_jsonl: Path) -> None:
        cfg = JsonlReaderConfig(
            input_dir=sample_jsonl.parent,
            id_field="id",
        )
        items = JsonlReader(cfg).read()
        ids = {i.id for i in items}
        assert "doc-0" in ids
        assert "doc-9" in ids

    def test_fallback_id_when_missing(self, tmp_dir: Path) -> None:
        p = tmp_dir / "no_id.jsonl"
        p.write_text(json.dumps({"text": "hello"}) + "\n")
        cfg = JsonlReaderConfig(input_dir=tmp_dir)
        items = JsonlReader(cfg).read()
        assert items[0].id == "no_id:1"

    def test_skips_blank_lines(self, tmp_dir: Path) -> None:
        p = tmp_dir / "with_blanks.jsonl"
        p.write_text(
            json.dumps({"id": "a", "text": "one"})
            + "\n\n"
            + json.dumps({"id": "b", "text": "two"})
            + "\n"
        )
        cfg = JsonlReaderConfig(input_dir=tmp_dir)
        items = JsonlReader(cfg).read()
        assert len(items) == 2

    def test_skips_malformed_json(self, tmp_dir: Path) -> None:
        p = tmp_dir / "bad.jsonl"
        p.write_text('{"id":"ok","text":"valid"}\nnot json\n')
        cfg = JsonlReaderConfig(input_dir=tmp_dir)
        items = JsonlReader(cfg).read()
        assert len(items) == 1

    def test_metadata_has_original_record(
        self, sample_jsonl: Path
    ) -> None:
        cfg = JsonlReaderConfig(input_dir=sample_jsonl.parent)
        items = JsonlReader(cfg).read()
        for item in items:
            assert "original_record" in item.metadata
            assert isinstance(item.metadata["original_record"], dict)


class TestReaderRegistry:
    def test_get_reader_text_directory(self, tmp_dir: Path) -> None:
        reader = get_reader(
            {"name": "text_directory", "input_dir": str(tmp_dir)}
        )
        assert isinstance(reader, TextDirectoryReader)

    def test_get_reader_jsonl(self, tmp_dir: Path) -> None:
        reader = get_reader(
            {"name": "jsonl", "input_dir": str(tmp_dir)}
        )
        assert isinstance(reader, JsonlReader)

    def test_get_reader_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reader"):
            get_reader({"name": "parquet"})

    def test_get_reader_from_config_object(self, tmp_dir: Path) -> None:
        cfg = TextDirectoryReaderConfig(input_dir=tmp_dir)
        reader = get_reader(cfg)
        assert isinstance(reader, TextDirectoryReader)
