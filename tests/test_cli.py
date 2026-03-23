"""Tests for the ExaForge CLI."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from exaforge.cli import app

runner = CliRunner()


def _write_config(tmp_dir: Path, endpoints_file: Path) -> Path:
    """Write a minimal ExaForge config for testing."""
    data = {
        "aegis": {
            "auto_launch": False,
            "endpoints_file": str(endpoints_file),
        },
        "task": {"name": "generation", "max_tokens": 10},
        "reader": {
            "name": "text_directory",
            "input_dir": str(tmp_dir),
        },
        "writer": {
            "name": "jsonl",
            "output_dir": str(tmp_dir / "output"),
            "buffer_size": 100,
        },
        "client": {
            "max_concurrent_requests": 2,
            "timeout": 5.0,
            "max_retries": 0,
        },
        "checkpoint": {
            "enabled": False,
        },
    }
    path = tmp_dir / "config.yaml"
    path.write_text(yaml.dump(data))
    return path


class TestMergeCommand:
    def test_merge_creates_merged_file(self, tmp_dir: Path) -> None:
        shard_dir = tmp_dir / "shards"
        shard_dir.mkdir()
        (shard_dir / "shard_0000.jsonl").write_text(
            json.dumps({"id": "a", "response": "one"}) + "\n"
        )
        (shard_dir / "shard_0001.jsonl").write_text(
            json.dumps({"id": "b", "response": "two"}) + "\n"
        )

        result = runner.invoke(app, ["merge", str(shard_dir)])
        assert result.exit_code == 0
        assert "Merged 2 shard(s)" in result.output

        merged = shard_dir / "merged.jsonl"
        assert merged.is_file()
        lines = merged.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_merge_with_custom_output(self, tmp_dir: Path) -> None:
        shard_dir = tmp_dir / "shards"
        shard_dir.mkdir()
        (shard_dir / "data.jsonl").write_text(
            json.dumps({"id": "x"}) + "\n"
        )

        out = tmp_dir / "custom.jsonl"
        result = runner.invoke(
            app, ["merge", str(shard_dir), "--output", str(out)]
        )
        assert result.exit_code == 0
        assert out.is_file()

    def test_merge_missing_dir(self, tmp_dir: Path) -> None:
        result = runner.invoke(
            app, ["merge", str(tmp_dir / "nonexistent")]
        )
        assert result.exit_code != 0

    def test_merge_no_files(self, tmp_dir: Path) -> None:
        result = runner.invoke(app, ["merge", str(tmp_dir)])
        assert result.exit_code == 0
        assert "No matching" in result.output


class TestStatusCommand:
    def test_status_missing_endpoints(self, tmp_dir: Path) -> None:
        cfg_path = _write_config(
            tmp_dir, tmp_dir / "nonexistent.txt"
        )
        result = runner.invoke(app, ["status", "--config", str(cfg_path)])
        assert result.exit_code == 1


class TestRunCommand:
    def test_run_missing_config(self) -> None:
        result = runner.invoke(
            app, ["run", "--config", "/tmp/does_not_exist.yaml"]
        )
        assert result.exit_code != 0
