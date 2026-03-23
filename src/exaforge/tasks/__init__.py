"""Pluggable task definitions for ExaForge.

Uses a name-based registry so the task implementation is selected
by the ``name`` field in the YAML config.
"""

from __future__ import annotations

from typing import Union

from exaforge.config import (
    CardExtractionTaskConfig,
    GenerationTaskConfig,
    TaskConfigs,
)

from .base import BaseTask
from .card_extraction import CardExtractionTask
from .generation import GenerationTask

__all__ = [
    "BaseTask",
    "CardExtractionTask",
    "GenerationTask",
    "get_task",
]

_STRATEGIES: dict[str, tuple[type, type[BaseTask]]] = {
    "generation": (GenerationTaskConfig, GenerationTask),
    "card_extraction": (CardExtractionTaskConfig, CardExtractionTask),
}


def get_task(config: Union[TaskConfigs, dict]) -> BaseTask:
    """Instantiate a task from a config object or raw dict."""
    if isinstance(config, dict):
        name = config.get("name", "")
    else:
        name = config.name

    entry = _STRATEGIES.get(name)  # type: ignore[arg-type]
    if entry is None:
        raise ValueError(
            f"Unknown task name: {name!r}. "
            f"Available: {set(_STRATEGIES)}"
        )

    config_cls, task_cls = entry
    if isinstance(config, dict):
        cfg = config_cls(**config)
    else:
        cfg = config
    return task_cls(cfg)
