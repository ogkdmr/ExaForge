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

    Subclasses *may* override :meth:`scan` and :meth:`read_by_ids` to
    enable a two-phase workflow where the orchestrator discovers IDs
    cheaply, filters via the checkpoint, and only then loads the items
    that will actually be processed.  The default implementations fall
    back to :meth:`read`.
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

    def scan(self) -> list[str]:
        """Return item IDs without loading content.

        The default implementation calls :meth:`read` and extracts IDs.
        Subclasses that can list items cheaply (e.g. a directory listing
        without reading file contents) should override this.
        """
        return [item.id for item in self.read()]

    def read_by_ids(self, ids: set[str]) -> list[InputItem]:
        """Load only items whose ID is in *ids*.

        The default implementation calls :meth:`read` and filters.
        Subclasses should override when selective loading is cheaper
        than a full read (e.g. reading 128 files vs. 58 000).
        """
        return [item for item in self.read() if item.id in ids]
