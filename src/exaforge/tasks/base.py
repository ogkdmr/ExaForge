"""Abstract base class for inference tasks.

A task is responsible for two things:

1. **prepare_messages** — turning an :class:`InputItem` into a list of
   OpenAI-format chat messages.
2. **parse_response** — extracting structured output from the raw model
   response text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from exaforge.readers.base import InputItem

if TYPE_CHECKING:
    from exaforge.writers.base import OutputRecord


class ItemSkipped(Exception):
    """Raised by a task to signal that an item should be skipped.

    The orchestrator catches this exception, writes the item to a
    type-specific skip file, and increments the matching counter.

    Parameters
    ----------
    reason : str
        Human-readable reason the item was skipped.
    skip_type : str
        Category used by the orchestrator to route to the right file
        and counter.  Conventional values: ``"too_long"``, ``"too_short"``.
    """

    def __init__(self, reason: str, skip_type: str = "skipped") -> None:
        self.reason = reason
        self.skip_type = skip_type
        super().__init__(reason)


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

    def extract_item_metadata(self, item: InputItem) -> dict[str, Any]:
        """Extract extra metadata from the input item for the output.

        Override this to promote fields from the input record into
        the top-level output metadata.  The default returns an empty
        dict (no extra fields).
        """
        return {}

    def build_records(
        self,
        item: InputItem,
        response_text: str,
        parsed: dict[str, Any],
    ) -> list[OutputRecord]:
        """Build the output records for one completed inference.

        The default produces a single :class:`OutputRecord` that merges
        ``item.metadata``, :meth:`extract_item_metadata`, and ``parsed``.

        Override this method to fan out one inference result into
        multiple records (e.g. one record per Q/A pair).
        """
        from exaforge.writers.base import OutputRecord

        extra = self.extract_item_metadata(item)
        return [
            OutputRecord(
                id=item.id,
                response=response_text,
                metadata={**item.metadata, **extra, **parsed},
            )
        ]
