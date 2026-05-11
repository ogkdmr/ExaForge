"""YAML-driven configuration for ExaForge.

Follows the BaseConfig pattern from distllm: Pydantic models with
``from_yaml`` / ``write_yaml`` helpers and discriminated-union sub-configs
selected by a ``name`` literal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional, TypeVar, Union, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

# Absolute path to the repository root (src/exaforge/config.py → ../../..)
_REPO_ROOT: Path = Path(__file__).parents[2]

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
    local_runs_dir: Path = _REPO_ROOT / "local_runs"
    endpoints_file: Optional[Path] = Field(
        default=None,
        description=(
            "Path to the Aegis endpoints file. "
            "When auto_launch is true this can be omitted — ExaForge will "
            "write endpoints to a timestamped sub-directory of local_runs_dir. "
            "When auto_launch is false this must point to an existing file. "
            "If set together with auto_launch and the Aegis config also "
            "specifies endpoints_file, the two paths must match."
        ),
    )


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


class QAGenerationTaskConfig(BaseConfig):
    """Config for the novel Q/A generation task."""

    name: Literal["qa_generation"] = "qa_generation"  # type: ignore[assignment]
    system_prompt: str = (
        "You are an expert reading comprehension analyst. "
        "You generate high-quality questions from fiction novels."
    )
    temperature: float = 0.7
    max_tokens: int = 16000
    top_p: float = 1.0
    questions_per_novel: int = 20
    max_input_tokens: int = 110000
    min_text_chars: int = 5000


class JournalCardExtractionTaskConfig(BaseConfig):
    """Config for the journal / publication card-extraction task."""

    name: Literal["journal_card_extraction"] = "journal_card_extraction"  # type: ignore[assignment]
    system_prompt: str = (
        "You are a helpful assistant specialised in analysing "
        "scientific texts and extracting structured publication metadata "
        "for bioinformatics and life-science research."
    )
    temperature: float = 0.3
    max_tokens: int = 4000
    top_p: float = 1.0
    chunk_size: int = 0
    character_limit: int = 0


TaskConfigs = Union[
    GenerationTaskConfig,
    CardExtractionTaskConfig,
    QAGenerationTaskConfig,
    JournalCardExtractionTaskConfig,
]


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
    id_field: Optional[str] = "id"

    @field_validator("input_dir")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return v.resolve()


class ZipTextReaderConfig(BaseConfig):
    """Read text files from batched ZIP archives.

    Produced by ``exaforge preprocess --format zip``.  When *stage_dir*
    is set, archives are extracted to fast node-local storage before
    reading (e.g. ``/tmp`` on Aurora).
    """

    name: Literal["zip_text"] = "zip_text"  # type: ignore[assignment]
    input_dir: Path = Path(".")
    glob_patterns: list[str] = Field(default=["*.zip"])
    stage_dir: Optional[Path] = Field(
        default=None,
        description=(
            "If set, archives are extracted here before reading. "
            "Use a fast local filesystem (e.g. /tmp) for best performance."
        ),
    )

    @field_validator("input_dir")
    @classmethod
    def _resolve_input(cls, v: Path) -> Path:
        return v.resolve()

    @field_validator("stage_dir")
    @classmethod
    def _resolve_stage(cls, v: Optional[Path]) -> Optional[Path]:
        return v.resolve() if v is not None else None


ReaderConfigs = Union[TextDirectoryReaderConfig, JsonlReaderConfig, ZipTextReaderConfig]


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
