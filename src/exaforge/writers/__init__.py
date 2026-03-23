"""Output writers for ExaForge.

Uses a name-based registry so the writer implementation is selected
by the ``name`` field in the YAML config.
"""

from __future__ import annotations

from typing import Union

from exaforge.config import JsonlWriterConfig, WriterConfigs

from .base import BaseWriter, OutputRecord
from .jsonl import JsonlWriter

__all__ = [
    "BaseWriter",
    "JsonlWriter",
    "OutputRecord",
    "get_writer",
]

_STRATEGIES: dict[str, tuple[type, type[BaseWriter]]] = {
    "jsonl": (JsonlWriterConfig, JsonlWriter),
}


def get_writer(config: Union[WriterConfigs, dict]) -> BaseWriter:
    """Instantiate a writer from a config object or raw dict."""
    if isinstance(config, dict):
        name = config.get("name", "")
    else:
        name = config.name

    entry = _STRATEGIES.get(name)  # type: ignore[arg-type]
    if entry is None:
        raise ValueError(
            f"Unknown writer name: {name!r}. "
            f"Available: {set(_STRATEGIES)}"
        )

    config_cls, writer_cls = entry
    if isinstance(config, dict):
        cfg = config_cls(**config)
    else:
        cfg = config
    return writer_cls(cfg)
