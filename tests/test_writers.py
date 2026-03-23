"""Tests for the ExaForge output writers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exaforge.config import JsonlWriterConfig
from exaforge.writers import JsonlWriter, OutputRecord, get_writer


class TestJsonlWriter:
    def _make_writer(
        self, tmp_dir: Path, buffer_size: int = 500
    ) -> JsonlWriter:
        cfg = JsonlWriterConfig(
            output_dir=tmp_dir / "out",
            buffer_size=buffer_size,
            base_name="test",
        )
        return JsonlWriter(cfg)

    def _make_records(self, n: int) -> list[OutputRecord]:
        return [
            OutputRecord(
                id=f"item-{i}",
                response=f"Response {i}",
                metadata={"score": i * 0.1},
            )
            for i in range(n)
        ]

    def test_write_and_flush(self, tmp_dir: Path) -> None:
        w = self._make_writer(tmp_dir, buffer_size=100)
        w.write(self._make_records(3))
        w.flush()

        files = list((tmp_dir / "out").glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().splitlines()
        assert len(lines) == 3

    def test_auto_flush_at_buffer_size(self, tmp_dir: Path) -> None:
        w = self._make_writer(tmp_dir, buffer_size=5)
        # Write 4 — stays in buffer (under threshold)
        w.write(self._make_records(4))
        files = list((tmp_dir / "out").glob("*.jsonl"))
        assert len(files) == 0

        # Write 3 more — total 7 >= 5, triggers auto-flush of all
        w.write(self._make_records(3))
        files = list((tmp_dir / "out").glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().splitlines()
        assert len(lines) == 7

    def test_close_flushes(self, tmp_dir: Path) -> None:
        w = self._make_writer(tmp_dir, buffer_size=100)
        w.write(self._make_records(2))
        w.close()
        files = list((tmp_dir / "out").glob("*.jsonl"))
        assert len(files) == 1

    def test_output_is_valid_jsonl(self, tmp_dir: Path) -> None:
        w = self._make_writer(tmp_dir, buffer_size=100)
        w.write(self._make_records(3))
        w.close()

        path = list((tmp_dir / "out").glob("*.jsonl"))[0]
        for line in path.read_text().strip().splitlines():
            record = json.loads(line)
            assert "id" in record
            assert "response" in record

    def test_metadata_merged_into_output(self, tmp_dir: Path) -> None:
        w = self._make_writer(tmp_dir, buffer_size=100)
        w.write(self._make_records(1))
        w.close()

        path = list((tmp_dir / "out").glob("*.jsonl"))[0]
        record = json.loads(path.read_text().strip())
        assert "score" in record

    def test_rotate_creates_new_file(self, tmp_dir: Path) -> None:
        w = self._make_writer(tmp_dir, buffer_size=100)
        w.write(self._make_records(2))
        w.rotate()
        w.write(self._make_records(2))
        w.close()

        files = sorted((tmp_dir / "out").glob("*.jsonl"))
        assert len(files) == 2
        assert "0000" in files[0].name
        assert "0001" in files[1].name

    def test_empty_flush_is_noop(self, tmp_dir: Path) -> None:
        w = self._make_writer(tmp_dir, buffer_size=100)
        w.flush()
        files = list((tmp_dir / "out").glob("*.jsonl"))
        assert len(files) == 0


class TestWriterRegistry:
    def test_get_writer_jsonl(self, tmp_dir: Path) -> None:
        w = get_writer(
            {"name": "jsonl", "output_dir": str(tmp_dir / "out")}
        )
        assert isinstance(w, JsonlWriter)

    def test_get_writer_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown writer"):
            get_writer({"name": "csv"})

    def test_get_writer_from_config_object(self, tmp_dir: Path) -> None:
        cfg = JsonlWriterConfig(output_dir=tmp_dir / "out")
        w = get_writer(cfg)
        assert isinstance(w, JsonlWriter)
