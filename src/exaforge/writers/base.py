"""Abstract base class for output writers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OutputRecord:
    """A single inference result ready to be written.

    Attributes
    ----------
    id : str
        Matches the :class:`~exaforge.readers.base.InputItem` id.
    response : str
        The raw model response text.
    metadata : dict
        Arbitrary key-value pairs (carried from the input item or
        added during processing).
    """

    id: str
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseWriter(ABC):
    """Interface that every writer must implement."""

    @abstractmethod
    def write(self, records: list[OutputRecord]) -> None:
        """Write a batch of output records to persistent storage.

        Implementations should buffer internally if needed and handle
        Lustre-friendly I/O patterns.
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Force any buffered data to disk."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Flush and release any resources."""
        ...
