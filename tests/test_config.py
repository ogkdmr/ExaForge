"""Tests for the ExaForge configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from exaforge.config import (
    AegisConfig,
    CardExtractionTaskConfig,
    CheckpointConfig,
    ClientConfig,
    ExaForgeConfig,
    GenerationTaskConfig,
    JsonlReaderConfig,
    JsonlWriterConfig,
    MonitorConfig,
    TextDirectoryReaderConfig,
)


class TestBaseConfigSerde:
    """YAML and JSON round-trip tests."""

    def test_write_and_read_yaml(self, tmp_dir: Path) -> None:
        cfg = ClientConfig(max_concurrent_requests=32, timeout=120.0)
        path = tmp_dir / "client.yaml"
        cfg.write_yaml(path)
        loaded = ClientConfig.from_yaml(path)
        assert loaded.max_concurrent_requests == 32
        assert loaded.timeout == 120.0

    def test_write_and_read_json(self, tmp_dir: Path) -> None:
        cfg = MonitorConfig(log_file=Path("/tmp/test.log"))
        path = tmp_dir / "monitor.json"
        cfg.write_json(path)
        import json

        data = json.loads(path.read_text())
        assert data["log_file"] == "/tmp/test.log"


class TestSubConfigs:
    """Individual sub-config validation."""

    def test_aegis_defaults(self) -> None:
        cfg = AegisConfig()
        assert cfg.auto_launch is False
        assert cfg.endpoints_file == Path("aegis_endpoints.txt")

    def test_client_defaults(self) -> None:
        cfg = ClientConfig()
        assert cfg.max_concurrent_requests == 64
        assert cfg.load_balance_strategy == "round_robin"

    def test_checkpoint_defaults(self) -> None:
        cfg = CheckpointConfig()
        assert cfg.enabled is True

    def test_generation_task(self) -> None:
        cfg = GenerationTaskConfig(max_tokens=500)
        assert cfg.name == "generation"
        assert cfg.max_tokens == 500

    def test_card_extraction_task(self) -> None:
        cfg = CardExtractionTaskConfig(mode="data_card")
        assert cfg.name == "card_extraction"
        assert cfg.mode == "data_card"

    def test_text_directory_reader_resolves_path(self) -> None:
        cfg = TextDirectoryReaderConfig(input_dir=Path("relative/dir"))
        assert cfg.input_dir.is_absolute()

    def test_jsonl_reader(self) -> None:
        cfg = JsonlReaderConfig(text_field="body")
        assert cfg.text_field == "body"

    def test_jsonl_writer(self) -> None:
        cfg = JsonlWriterConfig(buffer_size=100, base_name="cards")
        assert cfg.buffer_size == 100


class TestExaForgeConfig:
    """Top-level config composition and YAML round-trip."""

    def _minimal_yaml(self, tmp_dir: Path) -> Path:
        data = {
            "task": {"name": "generation", "max_tokens": 1024},
            "reader": {
                "name": "text_directory",
                "input_dir": str(tmp_dir),
            },
        }
        path = tmp_dir / "config.yaml"
        path.write_text(yaml.dump(data))
        return path

    def test_load_minimal(self, tmp_dir: Path) -> None:
        path = self._minimal_yaml(tmp_dir)
        cfg = ExaForgeConfig.from_yaml(path)
        assert isinstance(cfg.task, GenerationTaskConfig)
        assert cfg.task.max_tokens == 1024
        assert isinstance(cfg.reader, TextDirectoryReaderConfig)

    def test_full_round_trip(self, tmp_dir: Path) -> None:
        data = {
            "aegis": {"auto_launch": True, "endpoints_file": "ep.txt"},
            "task": {
                "name": "card_extraction",
                "mode": "agent_card",
                "max_tokens": 3000,
            },
            "reader": {
                "name": "jsonl",
                "input_dir": str(tmp_dir),
                "text_field": "body",
            },
            "writer": {
                "name": "jsonl",
                "output_dir": str(tmp_dir / "out"),
                "buffer_size": 200,
            },
            "client": {"max_concurrent_requests": 128, "timeout": 600},
            "monitor": {"enable_rich": False},
            "checkpoint": {"enabled": False},
        }
        path = tmp_dir / "full.yaml"
        path.write_text(yaml.dump(data))
        cfg = ExaForgeConfig.from_yaml(path)

        assert cfg.aegis.auto_launch is True
        assert isinstance(cfg.task, CardExtractionTaskConfig)
        assert cfg.task.mode == "agent_card"
        assert isinstance(cfg.reader, JsonlReaderConfig)
        assert cfg.reader.text_field == "body"
        assert cfg.writer.buffer_size == 200
        assert cfg.client.max_concurrent_requests == 128
        assert cfg.monitor.enable_rich is False
        assert cfg.checkpoint.enabled is False

    def test_write_yaml_and_reload(self, tmp_dir: Path) -> None:
        cfg = ExaForgeConfig(
            task=GenerationTaskConfig(),
            reader=TextDirectoryReaderConfig(input_dir=tmp_dir),
        )
        path = tmp_dir / "written.yaml"
        cfg.write_yaml(path)
        reloaded = ExaForgeConfig.from_yaml(path)
        assert reloaded.task.name == "generation"

    def test_invalid_task_name_rejected(self, tmp_dir: Path) -> None:
        data = {
            "task": {"name": "nonexistent"},
            "reader": {"name": "text_directory", "input_dir": str(tmp_dir)},
        }
        path = tmp_dir / "bad.yaml"
        path.write_text(yaml.dump(data))
        with pytest.raises(Exception):
            ExaForgeConfig.from_yaml(path)
