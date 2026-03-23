"""Abstract base class for input readers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InputItem:
    """A single unit of work to be sent through the inference pipeline.

    Attributes
    ----------
    id : str
        A unique, deterministic identifier for this item (used by the
        checkpoint system to track completion).
    text : str
        The raw text that will be transformed into a prompt by the task.
    metadata : dict
        Arbitrary key-value pairs carried through the pipeline and
        written alongside the model response.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseReader(ABC):
    """Interface that every reader must implement.

    A reader is responsible for bulk-loading input data from disk into a
    list of :class:`InputItem` objects.  Implementations should favour
    few, large I/O operations over many small ones (important on Lustre).
    """

    @abstractmethod
    def read(self) -> list[InputItem]:
        """Read all inputs and return them as a flat list.

        Returns
        -------
        list[InputItem]
            Every item that should be processed by the pipeline.
        """
        ...
