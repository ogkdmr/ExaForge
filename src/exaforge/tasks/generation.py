"""Generic text-generation task.

The simplest task: wraps the input text in a system + user message
pair and returns the model output as-is.
"""

from __future__ import annotations

from typing import Any

from exaforge.config import GenerationTaskConfig
from exaforge.readers.base import InputItem

from .base import BaseTask


class GenerationTask(BaseTask):
    """Prompt an LLM with a system prompt + the input text."""

    def __init__(self, config: GenerationTaskConfig) -> None:
        self.config = config

    def prepare_messages(
        self, item: InputItem
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": item.text},
        ]

    def parse_response(self, raw: str) -> dict[str, Any]:
        return {"generated_text": raw}
