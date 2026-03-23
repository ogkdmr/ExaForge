"""Abstract base class for inference tasks.

A task is responsible for two things:

1. **prepare_messages** — turning an :class:`InputItem` into a list of
   OpenAI-format chat messages.
2. **parse_response** — extracting structured output from the raw model
   response text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from exaforge.readers.base import InputItem


class BaseTask(ABC):
    """Interface that every task must implement."""

    @abstractmethod
    def prepare_messages(
        self, item: InputItem
    ) -> list[dict[str, str]]:
        """Convert an input item into OpenAI chat messages.

        Parameters
        ----------
        item : InputItem
            The input to transform.

        Returns
        -------
        list[dict[str, str]]
            A list of ``{"role": ..., "content": ...}`` dicts.
        """
        ...

    @abstractmethod
    def parse_response(self, raw: str) -> dict[str, Any]:
        """Parse the model's response into a structured dict.

        Parameters
        ----------
        raw : str
            The raw text returned by the model.

        Returns
        -------
        dict[str, Any]
            Parsed output to be merged into the output record.
        """
        ...
