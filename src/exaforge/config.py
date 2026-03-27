"""YAML-driven configuration for ExaForge.

Follows the BaseConfig pattern from distllm: Pydantic models with
``from_yaml`` / ``write_yaml`` helpers and discriminated-union sub-configs
selected by a ``name`` literal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional, TypeVar, Union

import yaml
from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")
PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Base config with YAML / JSON serialisation
# ---------------------------------------------------------------------------

class BaseConfig(BaseModel):
    """Pydantic base with YAML and JSON serialisation helpers."""

    name: Literal[""] = ""

    def write_yaml(self, path: PathLike) -> None:
        """Serialise the model to a YAML file."""
        with open(path, "w") as fp:
            yaml.dump(
                json.loads(self.model_dump_json()),
                fp,
                indent=4,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls: type[T], path: PathLike) -> T:
        """Deserialise from a YAML file."""
        with open(path) as fp:
            raw = yaml.safe_load(fp) or {}
        return cls(**raw)  # type: ignore[return-value]

    def write_json(self, path: PathLike) -> None:
        """Serialise the model to a JSON file."""
        with open(path, "w") as fp:
            fp.write(self.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Aegis sub-config
# ---------------------------------------------------------------------------

class AegisConfig(BaseConfig):
    """How ExaForge interacts with Aegis."""

    config_path: Optional[Path] = None
    auto_launch: bool = False
    wait_for_endpoints: bool = True
    endpoints_file: Path = Path("aegis_endpoints.txt")


# ---------------------------------------------------------------------------
# Client sub-config
# ---------------------------------------------------------------------------

class ClientConfig(BaseConfig):
    """Settings for the async HTTP client that talks to vLLM endpoints."""

    model: str = "default"
    max_concurrent_requests: int = 64
    timeout: float = 300.0
    max_retries: int = 3
    retry_backoff: float = 2.0
    load_balance_strategy: Literal["round_robin", "least_loaded"] = (
        "round_robin"
    )


# ---------------------------------------------------------------------------
# Monitor sub-config
# ---------------------------------------------------------------------------

class MonitorConfig(BaseConfig):
    """Settings for progress monitoring and logging."""

    log_file: Optional[Path] = None
    progress_interval: float = 30.0
    enable_rich: bool = True


# ---------------------------------------------------------------------------
# Checkpoint sub-config
# ---------------------------------------------------------------------------

class CheckpointConfig(BaseConfig):
    """Settings for job checkpointing / resume."""

    enabled: bool = True
    checkpoint_file: Path = Path("exaforge_checkpoint.json")


# ---------------------------------------------------------------------------
# Task sub-configs (discriminated union populated by tasks/ package)
# ---------------------------------------------------------------------------

class GenerationTaskConfig(BaseConfig):
    """Config for the generic text-generation task."""

    name: Literal["generation"] = "generation"  # type: ignore[assignment]
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0


class CardExtractionTaskConfig(BaseConfig):
    """Config for the paper card-extraction task."""

    name: Literal["card_extraction"] = "card_extraction"  # type: ignore[assignment]
    mode: Literal["model_card", "agent_card", "data_card"] = "model_card"
    system_prompt: str = (
        "You are a helpful assistant specialised in analysing "
        "scientific texts for bioinformatics workflows."
    )
    temperature: float = 0.3
    max_tokens: int = 4000
    top_p: float = 1.0
    chunk_size: int = 0
    character_limit: int = 0


TaskConfigs = Union[GenerationTaskConfig, CardExtractionTaskConfig]


# ---------------------------------------------------------------------------
# Reader sub-configs
# ---------------------------------------------------------------------------

class TextDirectoryReaderConfig(BaseConfig):
    """Read plain-text files from a directory."""

    name: Literal["text_directory"] = "text_directory"  # type: ignore[assignment]
    input_dir: Path = Path(".")
    glob_patterns: list[str] = Field(default=["*.txt"])

    @field_validator("input_dir")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return v.resolve()


class JsonlReaderConfig(BaseConfig):
    """Read records from one or more JSONL files."""

    name: Literal["jsonl"] = "jsonl"  # type: ignore[assignment]
    input_dir: Path = Path(".")
    glob_patterns: list[str] = Field(default=["*.jsonl"])
    text_field: str = "text"
    id_field: str = "id"

    @field_validator("input_dir")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return v.resolve()


ReaderConfigs = Union[TextDirectoryReaderConfig, JsonlReaderConfig]


# ---------------------------------------------------------------------------
# Writer sub-configs
# ---------------------------------------------------------------------------

class JsonlWriterConfig(BaseConfig):
    """Buffered JSONL output writer."""

    name: Literal["jsonl"] = "jsonl"  # type: ignore[assignment]
    output_dir: Path = Path("output")
    buffer_size: int = 500
    base_name: str = "results"

    @field_validator("output_dir")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return v.resolve()


WriterConfigs = JsonlWriterConfig


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

class ExaForgeConfig(BaseConfig):
    """Root configuration for an ExaForge run."""

    aegis: AegisConfig = Field(default_factory=AegisConfig)
    task: TaskConfigs = Field(discriminator="name")
    reader: ReaderConfigs = Field(discriminator="name")
    writer: WriterConfigs = Field(default_factory=JsonlWriterConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    max_items: int = Field(
        default=0,
        description="Cap the number of items processed. 0 means no limit (process all).",
    )
    batch_size: int = Field(
        default=1000,
        description=(
            "Number of items loaded from disk and held in memory at one time. "
            "Each batch is fully processed and written before the next is loaded. "
            "Controls memory footprint independently of max_concurrent_requests."
        ),
    )
