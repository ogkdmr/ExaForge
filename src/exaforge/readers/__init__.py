"""Input readers for ExaForge.

Uses a name-based registry (following the distllm pattern) so that the
reader implementation is selected by the ``name`` field in the YAML config.
"""

from __future__ import annotations

from typing import Union

from exaforge.config import JsonlReaderConfig, ReaderConfigs, TextDirectoryReaderConfig

from .base import BaseReader, InputItem
from .jsonl import JsonlReader
from .text_directory import TextDirectoryReader

__all__ = [
    "BaseReader",
    "InputItem",
    "JsonlReader",
    "TextDirectoryReader",
    "get_reader",
]

_STRATEGIES: dict[str, tuple[type, type[BaseReader]]] = {
    "text_directory": (TextDirectoryReaderConfig, TextDirectoryReader),
    "jsonl": (JsonlReaderConfig, JsonlReader),
}


def get_reader(config: Union[ReaderConfigs, dict]) -> BaseReader:
    """Instantiate a reader from a config object or raw dict."""
    if isinstance(config, dict):
        name = config.get("name", "")
    else:
        name = config.name

    entry = _STRATEGIES.get(name)  # type: ignore[arg-type]
    if entry is None:
        raise ValueError(
            f"Unknown reader name: {name!r}. "
            f"Available: {set(_STRATEGIES)}"
        )

    config_cls, reader_cls = entry
    if isinstance(config, dict):
        cfg = config_cls(**config)
    else:
        cfg = config
    return reader_cls(cfg)
